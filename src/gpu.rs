//! Shared GPU setup: runs cubecl on Bevy's own wgpu device so simulation buffers
//! can be consumed by the renderer without CPU round-trips.
//!
//! Pipeline: cubecl (wgpu backend, same `wgpu::Device`/`Queue` as Bevy) writes
//! simulation output into a persistent buffer → a render-world system does a GPU-only
//! `copy_buffer_to_texture` into the field-visualization `Image` → the terrain material
//! samples it. No data ever leaves the GPU.

use bevy::asset::Handle;
use bevy::ecs::resource::Resource;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::{prepare_assets, RenderAssets};
use bevy::render::render_resource::{
    CommandEncoderDescriptor, Extent3d, Origin3d, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureFormat,
};
use bevy::render::renderer::{RenderAdapter, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use cubecl::prelude::*;
use cubecl::wgpu::wgpu as wgpu;

/// The cubecl client running on Bevy's wgpu device (and queue).
#[derive(Resource)]
pub struct GpuContext {
    pub client: Client,
}

/// Registers the cubecl wgpu runtime on Bevy's device. Must run after the renderer is
/// initialized (the `RenderDevice`/`RenderQueue`/`RenderAdapter` main-world resources
/// exist once [`bevy::render::RenderPlugin`] is finished).
pub fn setup_gpu_context(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    render_adapter: Res<RenderAdapter>,
) {
    let adapter = render_adapter.clone().into_inner();
    let setup = cubecl::wgpu::WgpuSetup {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()),
        adapter,
        device: render_device.wgpu_device().clone(),
        queue: render_queue.clone().into_inner(),
        backend: render_device.wgpu_device().adapter_info().backend,
    };
    let device =
        cubecl::wgpu::init_device(setup, cubecl::wgpu::RuntimeOptions::default());
    let client = cubecl::Device::from(device).client();
    commands.insert_resource(GpuContext { client });
}

/// One field value becomes one `Rgba32Float` texel (`[v, 0, 0, 0]`, value in the red
/// channel), which is exactly the `block_copy_size` layout of `Rgba32Float` so
/// `copy_buffer_to_texture` can move it to the visualization texture byte-for-byte.
///
/// `F` is generic so the kernel is reusable for any float scalar and stays testable on
/// the CPU runtime.
#[cube(launch_unchecked)]
pub fn field_values_to_texels<F: Float + LaunchArg>(
    input: &[F],
    output: &mut [F],
    size_x: usize,
    size_y: usize,
) {
    let x = ABSOLUTE_POS_X as usize;
    let y = ABSOLUTE_POS_Y as usize;

    if x >= size_x || y >= size_y {
        terminate!();
    }

    let idx = x + y * size_x;
    let out = idx * 4;
    output[out] = input[idx].clamp(F::new(0.0_f32), F::new(1.0_f32));
    output[out + 1] = F::new(0.0_f32);
    output[out + 2] = F::new(0.0_f32);
    output[out + 3] = F::new(0.0_f32);
}

/// Which GPU-resident buffer currently backs the field visualization.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FieldVisSource {
    #[default]
    Soil,
    Water,
}

/// A cubecl-allocated buffer publishing GPU-resident texels for rendering.
#[derive(Clone, Debug, Default)]
pub struct VisBuffer {
    /// Stream of `Rgba32Float` texels (value in the red channel); `None` until the
    /// first simulation tick ran.
    pub buffer: Option<wgpu::Buffer>,
    /// Byte offset into the buffer (cubecl may give back a slice of a larger block).
    pub offset: u64,
    pub width: u32,
    pub height: u32,
}

/// The sim-side handle to the renderer: which buffer feeds the visualization texture,
/// plus where (in which texture) a render-world system should copy it each frame.
///
/// Main world is the source; the render world receives a clone via
/// [`ExtractResource`] (cheap: holds an `Arc` into the wgpu buffer).
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct FieldVisGpuData {
    pub soil: VisBuffer,
    pub water: VisBuffer,
    /// Which of the above is currently displayed.
    pub source: FieldVisSource,
    /// Target visualization texture (must be `Rgba32Float`).
    pub image: Handle<bevy::image::Image>,
    /// Whether the visualization currently displays data from a GPU buffer; the copy is
    /// skipped when false (e.g. when a CPU-resident field is shown instead).
    pub active: bool,
    /// Phase 3: per-cell water surface stream `[surface_height, depth, 0, 0]` (texels), copied
    /// into `surface_image` every frame regardless of the F1/F3/F5 field selection.
    pub surface: VisBuffer,
    /// Target texture for the water surface (Rgba32Float, e.g. 256x256).
    pub surface_image: Handle<bevy::image::Image>,
}

impl Default for FieldVisGpuData {
    fn default() -> Self {
        Self {
            soil: VisBuffer::default(),
            water: VisBuffer::default(),
            source: FieldVisSource::default(),
            image: Handle::default(),
            active: false,
            surface: VisBuffer::default(),
            surface_image: Handle::default(),
        }
    }
}

/// Copies the latest simulation texels into the visualization texture. Runs in the
/// render world, once per frame, entirely on the GPU.
fn copy_field_vis_to_texture(
    data: Option<Res<FieldVisGpuData>>,
    images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(data) = data else { return };

    // Phase 3: ALWAYS-copied water surface stream — MUST run before the field-vis
    // early-outs below. The `active` flag tracks the F1/F3/F5 field overlay; gating the
    // surface copy on it froze the water texture when the overlay was off (the plane
    // showed one stale frame: static, uniform, unresponsive to the surface tunables).
    let Some(surface_image) = images.get(&data.surface_image) else {
        return;
    };
    if surface_image.texture_descriptor.format != TextureFormat::Rgba32Float {
        return;
    }
    if data.surface.width == 0 || data.surface.height == 0 {
        return;
    }
    let Some(surface_buffer) = &data.surface.buffer else {
        return;
    };
    let surface_bpr = (data.surface.width * core::mem::size_of::<f32>() as u32 * 4).max(256);
    let mut surf_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("water_surface_to_texture"),
    });
    surf_encoder.copy_buffer_to_texture(
        TexelCopyBufferInfo {
            buffer: surface_buffer,
            layout: TexelCopyBufferLayout {
                offset: data.surface.offset,
                bytes_per_row: Some(surface_bpr),
                rows_per_image: None,
            },
        },
        TexelCopyTextureInfo {
            texture: &surface_image.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        Extent3d {
            width: data.surface.width,
            height: data.surface.height,
            depth_or_array_layers: 1,
        },
    );
    let surf_cmd = surf_encoder.finish();
    render_queue.submit(Some(surf_cmd));

    // Field-vis stream (F1/F3/F5 overlay), gated by the overlay state.
    if !data.active {
        return;
    }
    let src = match data.source {
        FieldVisSource::Soil => &data.soil,
        FieldVisSource::Water => &data.water,
    };
    if src.width == 0 || src.height == 0 {
        return;
    }
    let Some(buffer) = &src.buffer else { return };
    let Some(gpu_image) = images.get(&data.image) else { return };
    if gpu_image.texture_descriptor.format != TextureFormat::Rgba32Float {
        return;
    }
    // 4 f32 per texel.
    let bytes_per_row = (src.width * core::mem::size_of::<f32>() as u32 * 4).max(256);
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("field_vis_tick_to_texture"),
    });
    encoder.copy_buffer_to_texture(
        TexelCopyBufferInfo {
            buffer,
            layout: TexelCopyBufferLayout {
                offset: src.offset,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        TexelCopyTextureInfo {
            texture: &gpu_image.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        Extent3d {
            width: src.width,
            height: src.height,
            depth_or_array_layers: 1,
        },
    );
    let command_buffer = encoder.finish();
    render_queue.submit(Some(command_buffer));
}

/// Extracts [`FieldVisGpuData`] into the render world and copies it into the
/// visualization texture each frame (after `GpuImage` assets are prepared).
pub struct GpuRenderCopyPlugin;

impl Plugin for GpuRenderCopyPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<FieldVisGpuData>::default());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return };
        render_app.add_systems(
            Render,
            copy_field_vis_to_texture
                .in_set(RenderSystems::Prepare)
                .after(prepare_assets::<GpuImage>),
        );
    }
}
