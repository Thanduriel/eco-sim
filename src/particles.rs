//! Phase 4: render the SPH particles themselves as camera-facing round sprites.
//!
//! The sim publishes its current position buffer (`[x, y, z, 0]` per particle) into
//! [`gpu::FieldVisGpuData::particles`]; a render-world system mirrors it GPU-side into
//! the [`ShaderBuffer`] asset bound by [`ParticleMaterial`] (`gpu::copy_particles_to_shader_buffer`,
//! `copy_buffer_to_buffer` — no CPU round-trips). The mesh is a `num * 4`-vertex quad
//! list built once (the quad index and corner are decoded from `vertex_index`), and the
//! vertex shader reads the *live* position from the storage buffer and billboards the
//! quad (camera right/up from the `View` uniform). Benefits: exact particle positions
//! for debugging, and (with the soft round sprite + blending) a spray look.

use bevy::light::NotShadowCaster;
use bevy::pbr::Material;
use bevy::prelude::*;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::render::mesh::Indices;
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{AsBindGroup, PrimitiveTopology, ShaderType};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

use crate::{domain, gpu, parameters};

const SHADER_PATH: &str = "shaders/particles.wesl";

/// Uniform for `particles.wesl` (16 bytes, 16-byte aligned).
#[derive(Clone, Copy, Default, Debug, ShaderType)]
pub struct ParticleUniform {
    /// World diameter of the sprites (m).
    pub size: f32,
    /// Alpha of the sprites (0 = hidden).
    pub alpha: f32,
    _pad0: f32,
    _pad1: f32,
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
        SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER_PATH.into()
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

/// Spawns the particle sprite layer. Mirrors `water::setup_water`'s particle-count math
/// (blob extents / spacing) so the storage buffer always matches the sim.
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
    let num = nx * ny * nz;
    if num == 0 {
        return;
    }

    // Quad list: 4 vertices per particle. The shader decodes quad = vertex_index / 4,
    // corner = vertex_index % 4, and displaces the corner onto the particle position.
    let corners: [[f32; 3]; 4] = [
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    let corner_uvs: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut positions = Vec::with_capacity(num * 4);
    let mut normals = Vec::with_capacity(num * 4);
    let mut uvs = Vec::with_capacity(num * 4);
    for _ in 0..num {
        for c in 0..4 {
            positions.push(corners[c]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push(corner_uvs[c]);
        }
    }
    let mut indices = Vec::with_capacity(num * 6);
    for q in 0..num {
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
        vec![[0.0f32; 4]; num],
        RenderAssetUsages::default(),
    ));

    let material = materials.add(ParticleMaterial {
        uniform: ParticleUniform {
            size: wp.particle_size,
            alpha: if wp.particle_vis { wp.particle_alpha } else { 0.0 },
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
    field_vis_gpu.particles.count = num as u32;
}

/// Refreshes the material uniform when the F4 tunables change (size / alpha / visibility).
pub fn update_particle_uniform(
    params: Res<parameters::GeneralParameters>,
    assets: Res<ParticleAssets>,
    mut materials: ResMut<Assets<ParticleMaterial>>,
) {
    let Some(mut material) = materials.get_mut(&assets.material) else {
        return;
    };
    let wp = &params.water;
    let want_size = if wp.particle_vis { wp.particle_size } else { 0.0 };
    let want_alpha = if wp.particle_vis { wp.particle_alpha } else { 0.0 };
    if material.uniform.size == want_size && material.uniform.alpha == want_alpha {
        return;
    }
    material.uniform.size = want_size;
    material.uniform.alpha = want_alpha;
}
