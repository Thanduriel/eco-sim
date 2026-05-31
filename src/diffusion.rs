use bevy::prelude::*;
use cubecl::bytes::Bytes;
//use cubecl::num_traits::{One, Zero};
use cubecl::future;
use cubecl::prelude::*;
//uuse std::ops::Mul;

use crate::domain;
use crate::parameters;
use crate::parameters::{BoundaryCondition};
use crate::terrain;


// Finite Difference diffusion kernel
#[cube(launch_unchecked)]
fn diffusion_kernel_fe<F: Float + cubecl::CubeElement>(
    input: &Array<F>,
    output: &mut Array<F>,
    dt: F,
    h: F,
    diffusion_coefficient: F,
    bc_type: i32,
    bc_dirichlet: F,
    //bc: BoundaryCondition,
    size_x: usize,
    size_y: usize,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    let x = ABSOLUTE_POS_X as usize;
    let y = ABSOLUTE_POS_Y as usize;

    let center = input[x + y * size_x];

    let boundary_val = if bc_type == 0 {
      bc_dirichlet
     } else {
        center};
    
    //#[comptime]
 /*   let boundary_val = match bc {
        BoundaryCondition::Dirichlet(_) => F::new(0.0),
        BoundaryCondition::Closed => center,
    };*/

    let left = if x > 0 {
        input[x - 1 + y * size_x]
    } else {
        boundary_val
    };
    let right = if x + 1 < size_x {
        input[x + 1 + y * size_x]
    } else {
        boundary_val
    };
    let down = if y > 0 {
        input[x + (y - 1) * size_x]
    } else {
        boundary_val
    };
    let up = if y + 1 < size_y {
        input[x + (y + 1) * size_x]
    } else {
        boundary_val
    };

    output[ABSOLUTE_POS] = center
        + dt * diffusion_coefficient * (left + right + down + up - F::new(4.0) * center) / (h * h);
    //(input[ABSOLUTE_POS] + dt).clamp(F::zero(), F::one());
}

fn launch<R: Runtime, F: Float + cubecl::CubeElement>(
    device: &R::Device,
    field: &mut domain::Field<F>,
    dt: F,
    diffusion_params: &parameters::DiffusionParameters,
) {
    let start_upload = std::time::Instant::now();
    let client = R::client(device);
    let num_elem = field.buffer.len();
    let output_handle = client.empty(num_elem * core::mem::size_of::<F>());
    let input_handle = client.create(Bytes::from_elems(field.buffer.clone()));
    let cube_dim = 32;
    let _ = future::block_on(client.sync());
    println!("upload: {}s", start_upload.elapsed().as_secs_f64());

    let start_computation = std::time::Instant::now();
    unsafe {
        diffusion_kernel_fe::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(
                field.size.x as u32 / cube_dim,
                field.size.y as u32 / cube_dim,
                1,
            ),
            CubeDim::new_2d(cube_dim, cube_dim),
            ArrayArg::from_raw_parts(input_handle, num_elem),
            ArrayArg::from_raw_parts(output_handle.clone(), num_elem),
            dt,
            F::new(field.h),
            F::new(diffusion_params.diffusivity),
            match diffusion_params.bc{
                BoundaryCondition::Dirichlet(_) => 0,
                BoundaryCondition::Closed => 1,
                
            },
            match diffusion_params.bc{
                BoundaryCondition::Dirichlet(val) => F::new(val),
                BoundaryCondition::Closed => F::zero(),
            },
            field.size.x,
            field.size.y,
        )
    };
    let _ = future::block_on(client.sync());
    println!("compute: {}s", start_computation.elapsed().as_secs_f64());

    let start_download = std::time::Instant::now();
    let bytes = client.read_one(output_handle).unwrap();
    let output = F::from_bytes(&bytes);
    field.buffer = Vec::from(output);
    //field.buffer = bytes.try_into_vec().unwrap();

    println!("download: {}s", start_download.elapsed().as_secs_f64());
}

pub fn run_cubecl(
    time: Res<Time>,
    mut surface_query: Query<&mut terrain::Surface>,
    params: Res<parameters::GeneralParameters>,
) {
    let mut surface = surface_query.single_mut().unwrap();
    launch::<cubecl::cuda::CudaRuntime, f32>(
        &Default::default(),
        &mut surface.soil_moisture,
        time.delta_secs(),
        &params.ground.soil_water,
    );
}

/*
use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::GpuImage,
};
use std::borrow::Cow;

use crate::domain;
use crate::terrain::*;

#[derive(Resource, Clone, ExtractResource)]
pub struct DeviceFieldTextures(Vec<[Handle<Image>; 2]>);
/*
 * texture
 * + already has 2d structure
 * + can be rendered directly
 * - requires additional host buffer (image in main world)
 * ShaderBuffer
 * + can be directly integrated with domain::Field
 * + no dealing with interpolation (texel fetch), texture format
 */

fn setup_device_fields(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    surface_query: Query<&Surface>,
) {
    let surface = surface_query.single().unwrap();

    let mut image = Image::new_target_texture(
        surface.soil_moisture.size.x as u32,
        surface.soil_moisture.size.y as u32,
        TextureFormat::R32Float,
        None,
    );
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC;
    let image0 = images.add(image.clone());
    let image1 = images.add(image);

    commands.insert_resource(DeviceFieldTextures {
        0: vec![[image0, image1]],
    });

    commands.insert_resource(DeviceFieldUniforms {
        h: surface.soil_moisture.size.x as f32 / domain::SIZE_F32.x,
        dt: 1.0 / 60.0,
    });
}

#[derive(Resource)]
struct DeviceFieldBindGroups(Vec<[BindGroup; 2]>);

#[derive(Resource, Clone, ExtractResource, ShaderType)]
pub struct DeviceFieldUniforms {
    pub h: f32,
    pub dt: f32,
}

pub fn prepare_field_bind_groups(
    mut commands: Commands,
    pipeline: Res<FieldSimStepPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    device_fields: Res<DeviceFieldTextures>,
    device_field_uniforms: Res<DeviceFieldUniforms>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
) {
    let mut uniform_buffer = UniformBuffer::from(device_field_uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let mut bind_groups = Vec::with_capacity(device_fields.0.len());

    for [buf0, buf1] in &device_fields.0 {
        let view0 = gpu_images.get(buf0).unwrap();
        let view1 = gpu_images.get(buf1).unwrap();

        let bind_group0 = render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.texture_bind_group_layout),
            &BindGroupEntries::sequential((
                &view0.texture_view,
                &view1.texture_view, /*&uniform_buffer*/
            )),
        );
        let bind_group1 = render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.texture_bind_group_layout),
            &BindGroupEntries::sequential((&view0.texture_view, &view1.texture_view)),
        );
        bind_groups.push([bind_group0, bind_group1]);
    }
    commands.insert_resource(DeviceFieldBindGroups(bind_groups));
}

const DIFFUSION_SHADER_ASSET_PATH: &str = "shaders/diffusion.wgsl";

#[derive(Resource)]
struct FieldSimStepPipeline {
    texture_bind_group_layout: BindGroupLayoutDescriptor,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

fn init_field_sim_step_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let texture_bind_group_layout = BindGroupLayoutDescriptor::new(
        "FieldImages",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<DeviceFieldUniforms>(false),
            ),
        ),
    );
    let shader = asset_server.load(DIFFUSION_SHADER_ASSET_PATH);
    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("init")),
        ..default()
    });
    let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("update")),
        ..default()
    });

    commands.insert_resource(FieldSimStepPipeline {
        texture_bind_group_layout,
        init_pipeline,
        update_pipeline,
    });
}

#[derive(Resource, Default, PartialEq, Copy, Clone)]
enum DiffusionPipelineState {
    #[default]
    Loading,
    Init,
    Update(usize),
}

//pub fn diffusion_step() {}
pub fn update_diffusion_state(
    pipeline: Res<FieldSimStepPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut state: Res<DiffusionPipelineState>,
) {
    match *state {
        DiffusionPipelineState::Loading => {
            match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                CachedPipelineState::Ok(_) => {
                    *state = DiffusionPipelineState::Init;
                }
                // If the shader hasn't loaded yet, just wait.
                CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => {}
                CachedPipelineState::Err(err) => {
                    panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                }
                _ => {}
            }
        }
        DiffusionPipelineState::Init => {
            if let CachedPipelineState::Ok(_) =
                pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
            {
                *state = DiffusionPipelineState::Update(1);
            }
        }
        DiffusionPipelineState::Update(0) => {
            *state = DiffusionPipelineState::Update(1);
        }
        DiffusionPipelineState::Update(1) => {
            *state = DiffusionPipelineState::Update(0);
        }
        DiffusionPipelineState::Update(_) => unreachable!(),
    }
}

const WORKGROUP_SIZE: u32 = 8;

fn field_simulation_update(
    mut render_context: RenderContext,
    bind_groups: Res<DeviceFieldBindGroups>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FieldSimStepPipeline>,
    state: Res<DiffusionPipelineState>,
    mut surface_query: Query<&mut Surface>,
) {
    let mut surface = surface_query.single_mut().unwrap();

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    // select the pipeline based on the current state
    match *state {
        DiffusionPipelineState::Loading => {}
        DiffusionPipelineState::Init => {
            let init_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.init_pipeline)
                .unwrap();
            pass.set_bind_group(0, &bind_groups.0[0][0], &[]);
            pass.set_pipeline(init_pipeline);
            pass.dispatch_workgroups(
                surface.soil_moisture.size.x as u32 / WORKGROUP_SIZE,
                surface.soil_moisture.size.y as u32 / WORKGROUP_SIZE,
                1,
            );
        }
        DiffusionPipelineState::Update(index) => {
            let update_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.update_pipeline)
                .unwrap();
            pass.set_bind_group(0, &bind_groups.0[0][index], &[]);
            pass.set_pipeline(update_pipeline);
            pass.dispatch_workgroups(
                surface.soil_moisture.size.x as u32 / WORKGROUP_SIZE,
                surface.soil_moisture.size.y as u32 / WORKGROUP_SIZE,
                1,
            );
        }
    }
}

struct DiffusionPlugin;

impl Plugin for DiffusionPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<DeviceFieldTextures>::default(),
            ExtractResourcePlugin::<DeviceFieldUniforms>::default(),
        ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<DiffusionPipelineState>()
            .add_systems(RenderStartup, init_field_sim_step_pipeline)
            .add_systems(
                Render,
                prepare_field_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderGraph, field_simulation_update);
    }
}
*/
