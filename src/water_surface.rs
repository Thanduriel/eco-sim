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

use crate::{domain, parameters, terrain};
use bevy::light::NotShadowCaster;

/// Handle created in [`setup_water_surface`], kept for the uniform-refresh system.
#[derive(Resource, Default)]
pub struct WaterSurfaceAssets {
    pub material: Handle<WaterSurfaceMaterial>,
}

const SHADER_PATH: &str = "shaders/water_surface.wesl";

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
        SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER_PATH.into()
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
        // static, from the start). This was the mystery all along: with an empty
        // surface texture the plane is invisible yet still shadow-casting.
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
