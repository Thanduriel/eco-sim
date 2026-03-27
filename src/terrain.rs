use bevy::asset::RenderAssetUsages;
use bevy::color::{Color, Mix};
use bevy::image::{
    Image, ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor,
};
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, Face, TextureDimension, TextureFormat};
use bevy::tasks::{ComputeTaskPool, ParallelSliceMut};

use crate::{color_map, domain};
use noise::utils::{NoiseMap, NoiseMapBuilder};

#[derive(Component)]
pub struct Terrain {
    pub height_map: domain::Field<f32>,
}

#[derive(Resource, Default)]
pub struct TerrainAssets {
    pub ground_material: Handle<StandardMaterial>,
    pub field_vis_material: Handle<StandardMaterial>,
    pub field_vis_image: Handle<Image>,
}

#[derive(Component)]
pub struct Surface {
    pub veg_density: domain::Field<f32>,
}

fn get_terrain_height(noise_map: &NoiseMap, x: usize, y: usize) -> f32 {
    2.5 * noise_map.get_value(x, y) as f32
}

//const COLOR_BEDROCK: Color = Color::linear_rgb(87. / 255., 105. / 255., 95. / 255.);
//const COLOR_SAND: Color = Color::linear_rgb(194. / 255., 178. / 255., 128. / 255.);
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
    //COLOR_SAND.mix(&COLOR_BEDROCK, t)
    Color::linear_rgb(0.75, 0.75, 0.75).mix(&Color::WHITE, t)
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

// Set the color at each vertex to the one in field.
// If no range is provided, min and max values of the field are used.
pub fn set_terrain_color(mesh: &mut Mesh, field: &domain::Field<f32>, range: Option<(f32, f32)>) {
    let mut color_attr = mesh.remove_attribute(Mesh::ATTRIBUTE_COLOR).unwrap();
    let VertexAttributeValues::Float32x4(ref mut col_attr_vec) = color_attr else {
        panic!("Unexpected vertex format, expected Float32x4");
    };

    let pos_attr = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
    let VertexAttributeValues::Float32x3(pos_attr_vec) = pos_attr else {
        panic!("Unexpected vertex format, expected Float32x3");
    };

    let (min, max) = if let Some(min_max) = range {
        min_max
    } else {
        field.compute_min_max()
    };
    let cmap = color_map::ColorMap::new(min, max, color_map::ColorScheme::Incandescent);

    let task_pool = ComputeTaskPool::get();
    // Creating significantly more tasks than the available threads leads to more consistent timings.
    // todo: investigate again when there is more simulation work
    let chunk_size = 2048;
    col_attr_vec.par_chunk_map_mut(task_pool, chunk_size, |index, chunk| {
        let mut idx = index * chunk_size;
        for col in chunk {
            let pos = pos_attr_vec[idx];
            let pos_domain = Vec2::new(
                pos[0] + domain::HALF_SIZE.x as f32,
                pos[2] + domain::HALF_SIZE.y as f32,
            );
            *col = cmap
                .get_color(field.get_bilinear(pos_domain))
                .to_f32_array();
            idx += 1;
        }
    });

    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_attr);
}

pub fn generate_terrain_mesh(height_map: &domain::Field<f32>) -> Mesh {
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
        let h = height_map.get_bilinear(pos_domain);
        pos[1] = h;

        vertex_colors.push(get_terrain_color(h).to_linear().to_f32_array());
    }

    //mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);

    let _ = mesh.generate_tangents();

    mesh
}

// Sets the image to represent the field. The Image is resized if the sizes don't match.
// If no range is provided, min and max values of the field are used.
#[allow(dead_code)]
pub fn set_image_from_field(
    image: &mut Image,
    field: &domain::Field<f32>,
    range: Option<(f32, f32)>,
) {
    let field_size = UVec2::new(field.size.x as u32, field.size.y as u32);
    if image.size() != field_size {
        image.resize(Extent3d {
            width: field_size.x,
            height: field_size.y,
            depth_or_array_layers: 1,
        });
    }

    let (min, max) = if let Some(min_max) = range {
        min_max
    } else {
        field.compute_min_max()
    };
    let cmap = color_map::ColorMap::new(min, max, color_map::ColorScheme::Incandescent);

    const BYTES_PER_PIXEL: usize = 4;

    if let Some(data) = &mut image.data {
        let task_pool = ComputeTaskPool::get();
        // Creating significantly more tasks than the available threads leads to more consistent timings.
        // todo: investigate again when there is more simulation work
        //let chunk_size = field.num_elem() / task_pool.thread_num() * BYTES_PER_PIXEL;
        let chunk_size = 2048 * BYTES_PER_PIXEL;
        data.par_chunk_map_mut(task_pool, chunk_size, |index, chunk| {
            let mut idx = index * chunk_size / BYTES_PER_PIXEL;
            let (sub_chunks, []) = chunk.as_chunks_mut::<BYTES_PER_PIXEL>() else {
                unreachable!()
            };
            for bytes in sub_chunks {
                let color = cmap.get_color(field[idx]).to_u8_array();
                *bytes = color;
                idx += 1;
            }
        });
    }

    /*   if let Some(data) = &mut image.data {
        for y in 0..field.size.y {
            for x in 0..field.size.x {
                let offset = (x + y * field.size.x) * BYTES_PER_PIXEL;
                let color = cmap
                    .get_color(field[[x, y]])
                    .to_u8_array();
                data[offset..offset+4].copy_from_slice(&color);
            }
        }
    }*/

    // for some reason this variant is much faster in (single threaded)
    /*   for y in 0..field_size.y {
        for x in 0..field_size.x {
            image
                .set_color_at(
                    x,
                    y,
                    Color::LinearRgba(
                        cmap.get_color(field[[x as usize, y as usize]]),
                    ),
                )
                .unwrap();
        }
    }*/
}

pub fn setup_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut terrain_assets: ResMut<TerrainAssets>,
    asset_server: Res<AssetServer>,
) {
    let repeated = |settings: &mut ImageLoaderSettings| {
        settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            ..default()
        });
    };
    let disable_srgb = |settings: &mut ImageLoaderSettings| {
        settings.is_srgb = false;
        settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            ..default()
        });
    };

    // terrain
    let terrain_material = StandardMaterial {
        base_color_texture: Some(
            asset_server
                .load_with_settings("textures/ground/PX_Soil_Ground_24_albedo.jpg", repeated),
        ),
        normal_map_texture: Some(
            asset_server
                .load_with_settings("textures/ground/PX_Soil_Ground_24_normal.jpg", disable_srgb),
        ),
        /*    metallic_roughness_texture: Some(asset_server.load_with_settings(
            "textures/ground/PX_Soil_Ground_24_roughness.jpg",
            disable_srgb,
        )),*/
        occlusion_texture: Some(
            asset_server
                .load_with_settings("textures/ground/PX_Soil_Ground_24_ao.jpg", disable_srgb),
        ),
        //   alpha_mode: AlphaMode::Opaque,
        double_sided: false,
        ior: 1.45,
        perceptual_roughness: 0.7294,
        reflectance: 0.1, // in the blender material specular ior is set to 0.5 but his may be a different property
        // texture is designed for 2m x 2m but the checkerboard is very visible so we do 4x4 instead
        uv_transform: bevy::math::Affine2::from_scale(domain::SIZE_F32 * 0.25),
        cull_mode: Some(Face::Back),
        ..default()
    };
    terrain_assets.ground_material = materials.add(terrain_material);

    let terrain = Terrain::new(3);

    // (debug) visualize fields
    let mut field_vis_image = Image::new_fill(
        Extent3d {
            width: terrain.height_map.size.x as u32,
            height: terrain.height_map.size.y as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[255u8; 4],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    field_vis_image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::nearest());
    terrain_assets.field_vis_image = images.add(field_vis_image);

    let field_vis_material = StandardMaterial {
        base_color_texture: Some(terrain_assets.field_vis_image.clone()),
        double_sided: false,
        cull_mode: Some(Face::Back),
        ..default()
    };
    terrain_assets.field_vis_material = materials.add(field_vis_material);

    commands.spawn((
        Mesh3d(meshes.add(generate_terrain_mesh(&terrain.height_map))),
        MeshMaterial3d(terrain_assets.ground_material.clone()),
        Transform::from_xyz(domain::HALF_SIZE.x as f32, 0.0, domain::HALF_SIZE.y as f32),
        terrain,
        Surface {
            veg_density: domain::Field::new(3),
        },
    ));
}
