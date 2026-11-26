//! GPU-side water *rendering*: the heightfield-displaced translucent surface plane
//! (`WaterSurfaceMaterial`, samples the per-cell `[surface, depth, 0, 0]` stream) and the
//! SPH particle sprites (`ParticleMaterial`, reads the live sim position buffer through a
//! `ShaderBuffer` mirrored GPU-side every frame). Both are fully GPU-side: the sim
//! publishes its cubecl buffers into `gpu::FieldVisGpuData`, render-world systems copy
//! them GPU-side, and the shaders sample the results directly.

//! Phase 3: water surface rendering.
//!
//! The sim publishes a 256x256 `Rgba32Float` texture of per-cell
//! `[surface_height, depth, 0, 0]` (`water::surface_to_texels`), which the render world
//! copies GPU-side into [`terrain::TerrainAssets::water_surface_image`] every frame
//! (`gpu::copy_field_vis_to_texture`). This module renders it as a displaced, translucent
//! plane: the vertex shader looks up the surface height at each world position (manual
//! bilinear — `Rgba32Float` is not linearly filterable) and lifts the vertex; the fragment
//! shader colors the water by depth (shoreline transparency, deep-water tint).
//!
//! No CPU round-trips: the sim runs on the same wgpu device, the copy is GPU-only and the
//! material samples the resulting texture directly.

use bevy::{
    pbr::Material,
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

use crate::{domain, gpu, parameters, terrain};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::storage::ShaderBuffer;

/// Handle created in [`setup_water_surface`], kept for the uniform-refresh system.
#[derive(Resource, Default)]
pub struct WaterSurfaceAssets {
    pub material: Handle<WaterSurfaceMaterial>,
}

const WATER_SURFACE_SHADER_PATH: &str = "shaders/water_surface.wesl";

/// Material uniform. Must match the `WaterSurfaceUniform` struct in `water_surface.wesl`
/// (32 bytes, 16-byte aligned — WGSL uniform layout requires the padding quads).
#[derive(Clone, Copy, Default, Debug, ShaderType)]
pub struct WaterSurfaceUniform {
    /// Domain center (x, z): the vertex shader converts world position to texture UV
    /// `(world_pos.xz - center + size/2) / size`.
    pub center: Vec2,
    /// Domain extent (world units, square).
    pub world_size: f32,
    /// Water depth (m) at which the surface color saturates to "deep".
    pub depth_scale: f32,
    /// Water depth (m) at which the surface alpha saturates (shoreline transparency).
    pub alpha_scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct WaterSurfaceMaterial {
    #[uniform(0)]
    pub uniform: WaterSurfaceUniform,
    #[texture(1)]
    #[sampler(2)]
    /// `[surface_height, depth, 0, 0]` texels (Rgba32Float, nearest-sampled manually in
    /// the shader).
    pub surface_texture: Handle<Image>,
}

impl Material for WaterSurfaceMaterial {
    fn vertex_shader() -> ShaderRef {
        WATER_SURFACE_SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        WATER_SURFACE_SHADER_PATH.into()
    }

    /// Transparent pass, no depth writes (see `AlphaMode::Blend` semantics in Bevy).
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// Spawns the water surface plane (after `terrain::setup_terrain`: needs the
/// `water_surface_image` handle from `TerrainAssets`).
pub fn setup_water_surface(
    mut commands: Commands,
    terrain_assets: Res<terrain::TerrainAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
    params: Res<parameters::GeneralParameters>,
) {
    // 128x128 vertices (0.5 m cells): the shader lifts each vertex to the surface height;
    // the 256x256 sim texture is sampled with manual bilinear between vertex lookups.
    let mesh: Mesh = Plane3d::default()
        .mesh()
        .size(domain::SIZE_F32.x, domain::SIZE_F32.y)
        .subdivisions(127)
        .into();

    let material = materials.add(WaterSurfaceMaterial {
        uniform: WaterSurfaceUniform {
            center: Vec2::new(domain::CENTER.x, domain::CENTER.y),
            world_size: domain::SIZE_F32.x,
            depth_scale: params.water.surface_depth_scale,
            alpha_scale: params.water.surface_alpha_scale,
            ..default()
        },
        surface_texture: terrain_assets.water_surface_image.clone(),
    });

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(material.clone()),
        // Same placement as the terrain mesh (the plane is centered on the origin).
        Transform::from_xyz(domain::CENTER.x, 0.0, domain::CENTER.y),
        // CRITICAL: the plane spans the whole domain at y=0 and must NOT cast shadows —
        // as a default shadow-caster it occluded the sun and put the ENTIRE terrain
        // below y=0 in shadow ("darkened basins with a uniform, y=0 height everywhere",
        // static, from the start). Note: an empty surface texture makes the plane
        // invisible in those regions, yet the mesh still shadow-casts — the fix is
        // only here, not in the shader.
        NotShadowCaster,
    ));
    commands.insert_resource(WaterSurfaceAssets { material });
}

/// Refreshes the material uniform when the F4 tunables change. Mutating a render asset
/// marks it modified; Bevy re-extracts it and the bind group is rebuilt next frame.
pub fn update_water_surface_uniform(
    params: Res<parameters::GeneralParameters>,
    assets: Res<WaterSurfaceAssets>,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
) {
    let Some(mut material) = materials.get_mut(&assets.material) else {
        return;
    };
    let wp = &params.water;
    if material.uniform.depth_scale == wp.surface_depth_scale
        && material.uniform.alpha_scale == wp.surface_alpha_scale
    {
        return;
    }
    material.uniform.depth_scale = wp.surface_depth_scale;
    material.uniform.alpha_scale = wp.surface_alpha_scale;
}

// ---------------------------------------------------------------------------
// Phase 4: SPH particle sprites
// ---------------------------------------------------------------------------
//
// The sim publishes its current position buffer (`[x, y, z, 0]` per particle) into
// `gpu::FieldVisGpuData::particles`; a render-world system mirrors it GPU-side into the
// `ShaderBuffer` asset bound by [`ParticleMaterial`] (`gpu::copy_particles_to_shader_buffer`,
// `copy_buffer_to_buffer` — no CPU round-trips). The mesh is a `num * 4`-vertex quad
// list built once (quad and corner decoded from `vertex_index`), and the vertex shader
// reads the *live* position from the storage buffer and billboards the quad (camera
// right/up from the `View` uniform). Benefits: exact particle positions for debugging,
// and (with the soft round sprite + blending) a spray look.

const PARTICLE_SHADER_PATH: &str = "shaders/particles.wesl";

/// Uniform for `particles.wesl` (16 bytes, 16-byte aligned).
#[derive(Clone, Copy, Default, Debug, ShaderType)]
pub struct ParticleUniform {
    /// World diameter of the sprites (m).
    pub size: f32,
    /// Alpha of the sprites (0 = hidden).
    pub alpha: f32,
    /// Live particle count: the mesh is built at full capacity (the stream brush appends
    /// particles at runtime), and quads past this count are pushed outside the clip volume.
    pub count: f32,
    _pad: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct ParticleMaterial {
    #[uniform(0)]
    pub uniform: ParticleUniform,
    /// Live mirrored particle positions (`array<vec4<f32>>`, one per particle).
    #[storage(1, read_only)]
    pub particles: Handle<ShaderBuffer>,
}

impl Material for ParticleMaterial {
    fn vertex_shader() -> ShaderRef {
        PARTICLE_SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        PARTICLE_SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// Handle created in [`setup_particles`], kept for the uniform-refresh system.
#[derive(Resource, Default)]
pub struct ParticleAssets {
    pub material: Handle<ParticleMaterial>,
}

/// Spawns the particle sprite layer. Mirrors `water::setup_water`'s capacity math
/// (max of `max_particles` and the initial blob count) so the storage buffer and the
/// quad mesh always match the sim's allocation: the stream brush grows `num` up to the
/// capacity without re-uploading anything, and the shader discards quads past the live
/// count (`ParticleUniform::count`).
pub fn setup_particles(
    mut commands: Commands,
    params: Res<parameters::GeneralParameters>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ParticleMaterial>>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    mut field_vis_gpu: ResMut<gpu::FieldVisGpuData>,
) {
    let wp = &params.water;
    let s = wp.particle_spacing;
    let (nx, ny, nz) = (
        (wp.blob_size[0] / s).ceil() as usize,
        (wp.blob_size[1] / s).ceil() as usize,
        (wp.blob_size[2] / s).ceil() as usize,
    );
    let blob_num = nx * ny * nz;
    if blob_num == 0 {
        return;
    }
    let capacity = wp.max_particles.max(blob_num);

    // Quad list: 4 vertices per particle. The shader decodes quad = vertex_index / 4,
    // corner = vertex_index % 4, and displaces the corner onto the particle position.
    let corners: [[f32; 3]; 4] = [
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    let corner_uvs: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut positions = Vec::with_capacity(capacity * 4);
    let mut normals = Vec::with_capacity(capacity * 4);
    let mut uvs = Vec::with_capacity(capacity * 4);
    for _ in 0..capacity {
        for c in 0..4 {
            positions.push(corners[c]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push(corner_uvs[c]);
        }
    }
    let mut indices = Vec::with_capacity(capacity * 6);
    for q in 0..capacity {
        let b = (q * 4) as u32;
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));

    // The storage buffer the sim buffer is mirrored into every frame. Zero-initialized:
    // a frame where the copy hasn't run yet just draws off-screen/at the origin, and the
    // shader skips particles inside the terrain only by depth-testing (they're at y=0
    // initially, hidden under the terrain dome).
    let storage = shader_buffers.add(ShaderBuffer::new(
        vec![[0.0f32; 4]; capacity],
        RenderAssetUsages::default(),
    ));

    let material = materials.add(ParticleMaterial {
        uniform: ParticleUniform {
            size: wp.particle_size,
            alpha: if wp.particle_vis { wp.particle_alpha } else { 0.0 },
            count: blob_num as f32,
            ..default()
        },
        particles: storage.clone(),
    });

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(material.clone()),
        // Positions are displaced by the shader, so the CPU AABB (the unit quad at the
        // origin) is meaningless — never frustum-cull the sprite layer.
        NoFrustumCulling,
        // Invisible-without-this lesson: an always-present mesh must not shadow-cast.
        NotShadowCaster,
        Transform::from_xyz(domain::CENTER.x, 0.0, domain::CENTER.y),
    ));
    commands.insert_resource(ParticleAssets { material });

    field_vis_gpu.particles.storage = storage;
    field_vis_gpu.particles.count = blob_num as u32;
}

/// Refreshes the material uniform when the F4 tunables change (size / alpha / visibility)
/// or the stream brush grows the live particle count.
pub fn update_particle_uniform(
    params: Res<parameters::GeneralParameters>,
    assets: Res<ParticleAssets>,
    mut materials: ResMut<Assets<ParticleMaterial>>,
    particles: Option<Res<super::physics::WaterParticles>>,
) {
    let Some(mut material) = materials.get_mut(&assets.material) else {
        return;
    };
    let wp = &params.water;
    let want_size = if wp.particle_vis { wp.particle_size } else { 0.0 };
    let want_alpha = if wp.particle_vis { wp.particle_alpha } else { 0.0 };
    let want_count = particles.as_ref().map_or(0.0, |p| p.num as f32);
    if material.uniform.size == want_size
        && material.uniform.alpha == want_alpha
        && material.uniform.count == want_count
    {
        return;
    }
    material.uniform.size = want_size;
    material.uniform.alpha = want_alpha;
    material.uniform.count = want_count;
}
