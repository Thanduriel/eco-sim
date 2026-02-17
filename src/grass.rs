use bevy::{
    asset::RenderAssetUsages,
    mesh::Indices,
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::{AsBindGroup, PrimitiveTopology},
    shader::ShaderRef,
};

#[derive(Resource, egui_probe::EguiProbe)]
pub struct GrassParameters {
    pub max_age: f32,
    pub spawn_radius: f32,
    pub orientation_max_angle: f32,
    pub below_surface_depth: f32,
    pub surface_area: f32,
}

impl Default for GrassParameters {
    fn default() -> Self {
        GrassParameters {
            max_age: 60.0,
            spawn_radius: 1.0,
            orientation_max_angle: 0.25,
            below_surface_depth: 0.08,
            surface_area: 0.25,
        }
    }
}

#[derive(Resource, Default)]
pub struct GrassAssets {
    pub mesh: Handle<Mesh>,
    pub material: Handle<GrassMaterial>,
}

pub fn create_grass_mesh(segments: usize, base_width: f32) -> Mesh {
    let num_vertices = 2 * segments + 1;
    let width_2 = base_width / 2.0;
    const TIP_START: f32 = 0.7;
    let width_mod_fn = |h: f32| {
        if h < TIP_START {
            1.0
        } else {
            1.0 - (h - TIP_START) / (1.0 - TIP_START)
        }
    };
    let mut positions = Vec::with_capacity(num_vertices);
    // rectangular segments
    for i in 0..segments {
        let height = i as f32 / segments as f32;
        let w = width_2 * width_mod_fn(height);
        positions.push([-w, height, 0.0]);
        positions.push([w, height, 0.0]);
    }
    // tip
    positions.push([0.0, 1.0, 0.0]);

    // flat normals
    let normals = vec![[0.0, 0.0, 1.0]; num_vertices];

    // triangles
    let mut triangles = Vec::with_capacity((num_vertices - 2) * 3);
    let top_idx = (num_vertices - 1) as u32;
    for i in (0..top_idx - 3).step_by(2) {
        triangles.extend([i, i + 1, i + 2]);
        triangles.extend([i + 2, i + 1, i + 3]);
    }

    // tip
    triangles.extend([top_idx - 2, top_idx - 1, top_idx]);

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(triangles))
}

const SHADER_ASSET_PATH: &str = "shaders/grass.wgsl";

#[derive(AsBindGroup, Asset, TypePath, Default, Debug, Clone)]
pub struct GrassMaterialExtension {}

impl MaterialExtension for GrassMaterialExtension {
    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    /* fn deferred_vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }*/

    fn prepass_vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn enable_prepass() -> bool {
        true
    }

    fn enable_shadows() -> bool {
        true
    }
}

pub type GrassMaterial = ExtendedMaterial<StandardMaterial, GrassMaterialExtension>;

pub fn create_grass_material() -> GrassMaterial {
    ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::linear_rgb(0.0, 1.0, 0.0),
            alpha_mode: AlphaMode::Opaque,
            double_sided: true,
            perceptual_roughness: 0.6,
            reflectance: 0.8,
            cull_mode: None,
            ..default()
        },
        extension: GrassMaterialExtension {},
    }
}
