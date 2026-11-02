use bevy::asset::RenderAssetUsages;
use bevy::color::{Color, Mix};
use bevy::image::{
    Image, ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
    ImageSamplerDescriptor,
};
use bevy::mesh::VertexAttributeValues;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, Face, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;
use bevy::tasks::{ComputeTaskPool, ParallelSliceMut};

use crate::{color_map, domain};
use noise::utils::{NoiseMap, NoiseMapBuilder};

#[derive(Component)]
pub struct Terrain {
    pub height_map: domain::Field<f32>,
}

#[derive(Resource, Default)]
pub struct TerrainAssets {
    pub terrain_material: Handle<TerrainMaterial>,
    /// `Rgba8Unorm` texture holding the value of the currently visualized field (in the red
    /// channel).
    pub field_vis_image: Handle<Image>,
    /// Colormap LUT (256 x 1, linear) sampled by the field visualization shader.
    pub color_lut_image: Handle<Image>,
}

/// Whether a field is defined on grid *vertices* (point values, interpolated smoothly
/// between neighbors) or per *cell* (one constant value per cell, shaded flat).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum FieldKind {
    /// Value at grid point `i` lives at world position `i * h
    #[default]
    Vertex,
    /// One value per cell
    #[allow(dead_code)]
    Cell,
}

#[derive(Default, PartialEq, Eq, Copy, Clone, Debug)]
pub enum FieldType {
    #[default]
    None,
    VegDensity,   // [0,1]
    SoilMoisture, // [0, ∞)
}

impl FieldType {
    // mapping of
    pub fn kind(self) -> FieldKind {
        match self {
            FieldType::None | FieldType::VegDensity | FieldType::SoilMoisture => FieldKind::Vertex,
        }
    }
}

#[derive(Resource, Default)]
pub struct FieldVisState {
    pub field_type: FieldType,
}

/// GPU representation of the field-visualization parameters.
#[derive(Clone, Copy, Default, Debug, ShaderType)]
#[repr(C)]
pub struct FieldVisUniform {
    /// 0 = ground, 1 = field overlay (fragment shader mixes on this).
    pub weight: f32,
    /// Value range mapped onto the colormap.
    pub min: f32,
    pub max: f32,
    /// Field resolution along one axis (texel count).
    pub field_size: f32,
    /// World size of the domain along one axis (see `domain::SIZE`).
    pub domain_size: f32,
    /// Half-texel shift used to align grid points with texel centers (0.0 for cell fields).
    pub half_texel: f32,
    /// `FieldKind` as `u32` (0 = Vertex, 1 = Cell).
    pub kind: u32,
    // Padding to keep the uniform at a 16-byte multiple.
    _pad: u32,
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug, Default)]
pub struct FieldVisExtension {
    #[uniform(100)]
    pub uniform: FieldVisUniform,
    #[texture(101)]
    #[sampler(102)]
    pub field_texture: Option<Handle<Image>>,
    #[texture(103)]
    #[sampler(104)]
    pub color_lut: Option<Handle<Image>>,
}

pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, FieldVisExtension>;

impl MaterialExtension for FieldVisExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/field_vis.wesl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/field_vis.wesl".into()
    }
}

#[derive(Component)]
pub struct Surface {
    pub veg_density: domain::Field<f32>,
    pub soil_moisture: domain::Field<f32>,
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
        let mut height_map = domain::Field::new(subdivisions, 0.0);

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

#[allow(dead_code)]
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
#[allow(dead_code)]
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
            let pos_domain = Vec2::new(pos[0] + domain::CENTER.x, pos[2] + domain::CENTER.y);
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
        .size(domain::SIZE_F32.x, domain::SIZE_F32.y)
        .subdivisions((height_map.size.x - 1) as u32)
        .into();
    // get positions
    let pos_attr = mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION).unwrap();
    let VertexAttributeValues::Float32x3(pos_attr_vec) = pos_attr else {
        panic!("Unexpected vertex format, expected Float32x3");
    };

    // modify y with height sampling
    for pos in pos_attr_vec.iter_mut() {
        let pos_domain = Vec2::new(pos[0] + domain::CENTER.x, pos[2] + domain::CENTER.y);
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
    mut materials: ResMut<Assets<TerrainMaterial>>,
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

    // ground base material (also the base of `TerrainMaterial`)
    let ground_material = StandardMaterial {
        base_color_texture: Some(
            asset_server
                .load_builder()
                .with_settings(repeated)
                .load("textures/ground/PX_Soil_Ground_24_albedo.jpg"),
        ),
        normal_map_texture: Some(
            asset_server
                .load_builder()
                .with_settings(disable_srgb)
                .load("textures/ground/PX_Soil_Ground_24_normal.jpg"),
        ),
        /*    metallic_roughness_texture: Some(asset_server.load_with_settings(
            "textures/ground/PX_Soil_Ground_24_roughness.jpg",
            disable_srgb,
        )),*/
        occlusion_texture: Some(
            asset_server
                .load_builder()
                .with_settings(disable_srgb)
                .load("textures/ground/PX_Soil_Ground_24_ao.jpg"),
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

    let terrain = Terrain::new(3);

    // (debug) visualize fields.
    // The active field is stored in the red channel as an `Rgba8Unorm` texture. Unlike
    // `R32Float`, `Rgba8Unorm` is filterable on every adapter (it does not need the
    // `float32-filterable` feature), so a linear sampler always binds — otherwise one bad
    // binding would break the whole material, including the ground. Fields are in [0, 1]
    // and are colormapped on the GPU, so 8-bit precision is plenty for a debug view.
    let mut field_vis_image = Image::new_fill(
        Extent3d {
            width: terrain.height_map.size.x as u32,
            height: terrain.height_map.size.y as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 4],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    field_vis_image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    terrain_assets.field_vis_image = images.add(field_vis_image);

    // Colormap LUT sampled by the shader (linear space to match `color_map::ColorMap`).
    terrain_assets.color_lut_image = images.add(build_field_color_lut());

    let terrain_material = TerrainMaterial {
        base: ground_material,
        extension: FieldVisExtension {
            uniform: FieldVisUniform {
                weight: 0.0,
                min: 0.0,
                max: 1.0,
                field_size: terrain.height_map.size.x as f32,
                domain_size: domain::SIZE.x as f32,
                half_texel: 0.5 / terrain.height_map.size.x as f32,
                ..default()
            },
            field_texture: Some(terrain_assets.field_vis_image.clone()),
            color_lut: Some(terrain_assets.color_lut_image.clone()),
        },
    };
    terrain_assets.terrain_material = materials.add(terrain_material);

    commands.spawn((
        Mesh3d(meshes.add(generate_terrain_mesh(&terrain.height_map))),
        MeshMaterial3d(terrain_assets.terrain_material.clone()),
        Transform::from_xyz(domain::CENTER.x, 0.0, domain::CENTER.y),
        terrain,
        Surface {
            veg_density: domain::Field::new(3, 0.0),
            soil_moisture: domain::Field::new(2, 0.5),
        },
    ));
}

/// Builds the 256 x 1 linear-space colormap LUT used by the field visualization shader.
fn build_field_color_lut() -> Image {
    const LUT_SIZE: u32 = 256;
    let cmap = color_map::ColorMap::new(0.0, 1.0, color_map::ColorScheme::Incandescent);
    let mut data = Vec::with_capacity((LUT_SIZE * 4) as usize);
    for i in 0..LUT_SIZE {
        let t = i as f32 / (LUT_SIZE - 1) as f32;
        data.extend_from_slice(&cmap.get_color(t).to_u8_array());
    }

    let mut image = Image::new_fill(
        Extent3d {
            width: LUT_SIZE,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 4],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    image.data = Some(data);
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    image
}

/// Returns the field that the given visualization type displays, if any.
fn visible_field(field_type: FieldType, surface: &Surface) -> Option<&domain::Field<f32>> {
    match field_type {
        FieldType::None => None,
        FieldType::VegDensity => Some(&surface.veg_density),
        FieldType::SoilMoisture => Some(&surface.soil_moisture),
    }
}

/// Writes the field values into the `Rgba8Unorm` texture (value in the red channel),
/// resizing it if the field's resolution changed. Fields are clamped to [0, 1]; the shader
/// maps them through the colormap LUT. This is the only per-tick CPU work the visualization
/// needs.
fn write_field_to_image(image: &mut Image, field: &domain::Field<f32>) {
    let n = field.size.x as u32;
    let extent = Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: 1,
    };
    if image.size() != UVec2::new(n, n) {
        image.resize(extent);
    }
    let Some(data) = image.data.as_mut() else {
        return;
    };
    for (dst, &v) in data.chunks_exact_mut(4).zip(field.buffer.iter()) {
        let v = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        dst[0] = v;
        dst[1] = v;
        dst[2] = v;
        dst[3] = 255;
    }
}

/// Applies the current visualization mode to the terrain material and field texture. This is the
/// single place that writes to the field image; it is invoked by [`field_vis_refresh_system`]
/// whenever the selected field changes or the simulation mutates [`Surface`].
fn configure_field_vis(
    state: &FieldVisState,
    terrain_assets: &TerrainAssets,
    materials: &mut Assets<TerrainMaterial>,
    images: &mut Assets<Image>,
    surface: &Surface,
) {
    let Some(field) = visible_field(state.field_type, surface) else {
        // Hide the overlay but leave the texture resized/buffered as-is. Cheap and rare.
        if let Some(mut mat) = materials.get_mut(&terrain_assets.terrain_material) {
            mat.extension.uniform.weight = 0.0;
        }
        return;
    };

    let kind = state.field_type.kind();
    let n = field.size.x as f32;
    let filter = match kind {
        FieldKind::Vertex => ImageFilterMode::Linear,
        FieldKind::Cell => ImageFilterMode::Nearest,
    };

    if let Some(mut image) = images.get_mut(&terrain_assets.field_vis_image) {
        let mut desc = ImageSamplerDescriptor::linear();
        desc.set_filter(filter);
        image.sampler = ImageSampler::Descriptor(desc);
        write_field_to_image(&mut image, field);
    }

    if let Some(mut mat) = materials.get_mut(&terrain_assets.terrain_material) {
        let u = &mut mat.extension.uniform;
        u.weight = 1.0;
        u.min = 0.0;
        u.max = 1.0;
        u.field_size = n;
        u.domain_size = domain::SIZE.x as f32;
        u.half_texel = match kind {
            FieldKind::Vertex => 0.5 / n,
            FieldKind::Cell => 0.0,
        };
        u.kind = kind as u32;
    }
}

/// It refreshes when either:
///   * the selected field changed, or
///   * the simulation mutated `Surface` (change detection, so we skip idle frames and don't
///     re-upload on every `FixedUpdate` tick, which can run several times per frame at high
///     game speed).
/// This is what feeds Bevy's main-world -> render-world `RenderAsset<Image>` extraction; all
/// writes to the field image happen in `configure_field_vis`.
pub fn field_vis_refresh_system(
    surface_query: Query<Ref<Surface>>,
    state: Res<FieldVisState>,
    terrain_assets: Res<TerrainAssets>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut last_field_type: Local<Option<FieldType>>,
) {
    let Ok(surface) = surface_query.single() else {
        return;
    };

    // user selected another field or current field values changed
    let field_type_changed = *last_field_type != Some(state.field_type);
    if !field_type_changed && !surface.is_changed() {
        return;
    }
    *last_field_type = Some(state.field_type);

    configure_field_vis(
        &state,
        &terrain_assets,
        &mut materials,
        &mut images,
        &surface,
    );
}
