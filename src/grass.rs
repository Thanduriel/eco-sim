use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};

#[derive(Resource, Default)]
pub struct GrassAssets {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
}

pub fn create_grass_mesh(segments : usize) -> Mesh {
    let num_vertices = 2 * segments + 1;
    let mut positions = Vec::with_capacity(num_vertices * 2);
    // rectangular segments
    for i in 0..segments {
        let height = i as f32 / segments as f32;
        positions.push([-0.1, height, 0.0]);
        positions.push([0.1, height, 0.0]);
    }
    // tip
    positions.push([0.0, 1.0, 0.0]);
    // duplicate vertices for backside
    positions.append(&mut positions.clone());

    // flat normals
    let mut normals = vec![[0.0, 0.0, 1.0]; num_vertices];
    normals.resize(num_vertices * 2, [0.0, 0.0, -1.0]);

    let mut triangles = Vec::with_capacity(num_vertices * 2 * 3);
    let top_idx = (num_vertices - 1) as u32;
    for i in (0..top_idx).step_by(4) {
        triangles.extend([i, i+1, i+2]);    
        triangles.extend([i + 2, i + 1, i + 3]);

        // backside with opposite orientation
        let j = i + num_vertices as u32;
        triangles.extend([j, j+2, j+1]);    
        triangles.extend([j + 2, j + 3, j + 1]);
    }
    // tip
    let top_idx = (num_vertices - 1) as u32;
    triangles.extend([top_idx - 2, top_idx - 1, top_idx]);
    let top_idx_back = top_idx + num_vertices as u32;
    triangles.extend([top_idx_back - 2, top_idx_back, top_idx_back - 1]);

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        positions,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        normals,
    )
    .with_inserted_indices(Indices::U32(triangles))
}

pub fn create_grass_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::linear_rgb(0.0, 1.0, 0.0),
        alpha_mode: AlphaMode::Opaque,
        double_sided: true,
        perceptual_roughness: 0.6,
        reflectance: 0.8,
        cull_mode: None,
        ..default()
    }
}
