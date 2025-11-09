use bevy::color::{Color, Mix};
use bevy::prelude::*;
//use bevy::render::render_resource::PrimitiveTopology;
use bevy::mesh::VertexAttributeValues;

use noise::utils::{NoiseMap, NoiseMapBuilder};

// for now it just marks the entity
#[derive(Component)]
pub struct Terrain {}

fn get_terrain_height(noise_map: &NoiseMap, x: f32, y: f32) -> f32 {
    2.5 * noise_map.get_value(x as usize, y as usize) as f32
}

const COLOR_BEDROCK: Color = Color::linear_rgb(87. / 255., 105. / 255., 95. / 255.);
const COLOR_SAND: Color = Color::linear_rgb(194. / 255., 178. / 255., 128. / 255.);
const BOUNDARY_WIDTH: f32 = 0.1;
const BOUNDARY_POS: f32 = -0.2;

fn get_terrain_color(height: f32) -> Color {
    const BOUNDARY_START: f32 = BOUNDARY_POS - BOUNDARY_WIDTH;
    let t = (height - BOUNDARY_START).clamp(0.0, 1.0);
    COLOR_SAND.mix(&COLOR_BEDROCK, t)
}

pub fn generate_terrain_mesh(x: f32, z: f32, size: f32, subdivisions: u32) -> Mesh {
    //let fbm = noise::Fbm::<noise::Fbm<noise::Perlin>>::default();
    let noise_fn = noise::HybridMulti::<noise::Perlin>::default();

    let noise_map: NoiseMap = noise::utils::PlaneMapBuilder::new(noise_fn)
        .set_size(subdivisions as usize, subdivisions as usize)
        .set_x_bounds(-1.0, 1.0) //(x + size) as f64)
        .set_y_bounds(-1.0, 1.0) //(z + size) as f64)
        .build();

    let num_vertices: usize = (subdivisions as usize + 2) * (subdivisions as usize + 2);
    //let mut uvs: Vec<[f32;2]> = Vec::with_capacity(num_vertices);
    let mut vertex_colors: Vec<[f32; 4]> = Vec::with_capacity(num_vertices);
    let mut mesh: Mesh = Plane3d::default()
        .mesh()
        .size(size, size)
        .subdivisions(subdivisions)
        .into();
    // get positions
    let pos_attr = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION).unwrap();
    let VertexAttributeValues::Float32x3(pos_attr_vec) = pos_attr else {
        panic!("Unexpected vertex format, expected Float32x3");
    };
    // modify y with height sampling
    let r = (subdivisions as f32) / size;
    for pos in pos_attr_vec.iter_mut() {
        pos[1] = get_terrain_height(&noise_map, r * (pos[0] - x), r * (pos[2] - z));
        vertex_colors.push(get_terrain_color(pos[1]).to_linear().to_f32_array());
    }

    //mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);

    let _ = mesh.generate_tangents();

    mesh
}
