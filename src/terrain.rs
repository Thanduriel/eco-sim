use bevy::color::{Color, Mix};
use bevy::prelude::*;
//use bevy::render::render_resource::PrimitiveTopology;
use bevy::mesh::VertexAttributeValues;

use crate::{color_map, domain};
use noise::utils::{NoiseMap, NoiseMapBuilder};

#[derive(Component)]
pub struct Terrain {
    pub height_map: domain::Field<f32>,
}

#[derive(Component)]
pub struct Surface {
    pub veg_density: domain::Field<f32>,
}

fn get_terrain_height(noise_map: &NoiseMap, x: usize, y: usize) -> f32 {
    2.5 * noise_map.get_value(x, y) as f32
}

const COLOR_BEDROCK: Color = Color::linear_rgb(87. / 255., 105. / 255., 95. / 255.);
const COLOR_SAND: Color = Color::linear_rgb(194. / 255., 178. / 255., 128. / 255.);
const BOUNDARY_WIDTH: f32 = 0.1;
const BOUNDARY_POS: f32 = -0.2;

impl Terrain {
    pub fn new(subdivisions: i32) -> Self {
        let mut height_map = domain::Field::new(subdivisions);

        //let fbm = noise::Fbm::<noise::Fbm<noise::Perlin>>::default();
        let noise_fn = noise::HybridMulti::<noise::Perlin>::default();

        let noise_map: NoiseMap = noise::utils::PlaneMapBuilder::new(noise_fn)
            .set_size(height_map.size.x, height_map.size.y)
            .set_x_bounds(0.0, domain::SIZE.x as f64 / 32.0) // bounds just determine the frequency
            .set_y_bounds(0.0, domain::SIZE.y as f64 / 32.0)
            .build();

        for y in 0..height_map.size.y {
            for x in 0..height_map.size.x {
                height_map[[x, y]] = get_terrain_height(&noise_map, x, y);
            }
        }

        Terrain {
            height_map: height_map,
        }
    }
}

fn get_terrain_color(height: f32) -> Color {
    const BOUNDARY_START: f32 = BOUNDARY_POS - BOUNDARY_WIDTH;
    let t = (height - BOUNDARY_START).clamp(0.0, 1.0);
    COLOR_SAND.mix(&COLOR_BEDROCK, t)
}

pub fn reset_terrain_color(mesh: &mut Mesh) {
    let mut color_attr = mesh.remove_attribute(Mesh::ATTRIBUTE_COLOR).unwrap();
    let VertexAttributeValues::Float32x4(ref mut col_attr_vec) = color_attr else {
        panic!("Unexpected vertex format, expected Float32x4");
    };

    let pos_attr = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
    let VertexAttributeValues::Float32x3(pos_attr_vec) = pos_attr else {
        panic!("Unexpected vertex format, expected Float32x3");
    };

    for (pos, col) in pos_attr_vec.iter().zip(col_attr_vec.iter_mut()) {
        *col = get_terrain_color(pos[1]).to_linear().to_f32_array();
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_attr);
}

pub fn set_terrain_color(mesh: &mut Mesh, color_values: &domain::Field<f32>) {
    let mut color_attr = mesh.remove_attribute(Mesh::ATTRIBUTE_COLOR).unwrap();
    let VertexAttributeValues::Float32x4(ref mut col_attr_vec) = color_attr else {
        panic!("Unexpected vertex format, expected Float32x4");
    };

    let pos_attr = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
    let VertexAttributeValues::Float32x3(pos_attr_vec) = pos_attr else {
        panic!("Unexpected vertex format, expected Float32x3");
    };

    let (min, max) = color_values.compute_min_max();
    let cmap = color_map::ColorMap::new(min, max, color_map::ColorScheme::Incandescent);

    for (pos, col) in pos_attr_vec.iter().zip(col_attr_vec.iter_mut()) {
        let pos_domain = Vec2::new(
            pos[0] + domain::HALF_SIZE.x as f32,
            pos[2] + domain::HALF_SIZE.y as f32,
        );

        *col = cmap
            .get_color(color_values.get_nearest(pos_domain))
            .to_f32_array();
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_attr);
}

pub fn generate_terrain_mesh(
    height_map: &domain::Field<f32>,
) -> Mesh {
    let num_vertices: usize = (height_map.size.x + 2) * (height_map.size.y + 2);
    //let mut uvs: Vec<[f32;2]> = Vec::with_capacity(num_vertices);
    let mut vertex_colors: Vec<[f32; 4]> = Vec::with_capacity(num_vertices);
    let mut mesh: Mesh = Plane3d::default()
        .mesh()
        .size(domain::SIZE.x as f32, domain::SIZE.y as f32)
        .subdivisions((height_map.size.x - 1) as u32)
        .into();
    // get positions
    let pos_attr = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION).unwrap();
    let VertexAttributeValues::Float32x3(pos_attr_vec) = pos_attr else {
        panic!("Unexpected vertex format, expected Float32x3");
    };

    // modify y with height sampling
    for pos in pos_attr_vec.iter_mut() {
        let pos_domain = Vec2::new(
            pos[0] + domain::HALF_SIZE.x as f32,
            pos[2] + domain::HALF_SIZE.y as f32,
        );
        let h = height_map.get_nearest(pos_domain);
        pos[1] = h;

        vertex_colors.push(get_terrain_color(h).to_linear().to_f32_array());
    }

    //mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);

    let _ = mesh.generate_tangents();

    mesh
}
