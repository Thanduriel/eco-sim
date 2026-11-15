use bevy::prelude::*;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::gpu;
use crate::parameters;
use crate::parameters::BoundaryCondition;
use crate::terrain;

// Finite Difference diffusion kernel
#[cube(launch_unchecked)]
fn diffusion_kernel_fe<F: Float + LaunchArg>(
    input: &[F],
    output: &mut [F],
    dt: F,
    h: F,
    diffusion_coefficient: F,
    bc_type: i32,
    bc_dirichlet: F,
    size_x: usize,
    size_y: usize,
) {
    let x = ABSOLUTE_POS_X as usize;
    let y = ABSOLUTE_POS_Y as usize;

    if x >= size_x || y >= size_y {
        terminate!();
    }

    let idx = x + y * size_x;
    let center = input[idx];

    let boundary_val = if bc_type == 0 { bc_dirichlet } else { center };

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

    output[idx] = center
        + dt * diffusion_coefficient * (left + right + down + up - F::new(4.0_f32) * center)
            / (h * h);
}

/// Persistent GPU buffers for the diffusion field, so no buffer is (re)allocated per
/// physics tick anymore; the sim result stays resident on the GPU and is shared with
/// the renderer (via [`gpu::field_values_to_texels`] + `copy_buffer_to_texture`).
pub struct DiffusionGpu {
    input: Handle,
    output: Handle,
    texels: Handle,
    size_x: usize,
    size_y: usize,
}

impl DiffusionGpu {
    fn new(client: &Client, num_elem: usize, size_x: usize, size_y: usize) -> Self {
        Self {
            input: client.empty(num_elem * core::mem::size_of::<f32>()),
            output: client.empty(num_elem * core::mem::size_of::<f32>()),
            // 4 f32 texels per field value (Rgba32Float layout: [v, 0, 0, 0]).
            texels: client.empty(num_elem * 4 * core::mem::size_of::<f32>()),
            size_x,
            size_y,
        }
    }
}

pub fn run_cubecl(
    time: Res<Time>,
    mut surface_query: Query<&mut terrain::Surface>,
    params: Res<parameters::GeneralParameters>,
    gpu_context: Res<gpu::GpuContext>,
    terrain_assets: Res<terrain::TerrainAssets>,
    mut field_vis_gpu: ResMut<gpu::FieldVisGpuData>,
    mut diffusion_gpu: Local<Option<DiffusionGpu>>,
) {
    if !params.ground.soil_water.is_active {
        return;
    }

    let mut surface = surface_query.single_mut().unwrap();
    let client = &gpu_context.client;

    let num_elem = surface.soil_moisture.buffer.len();
    let size_x = surface.soil_moisture.size.x;
    let size_y = surface.soil_moisture.size.y;
    if diffusion_gpu.is_none() {
        *diffusion_gpu = Some(DiffusionGpu::new(client, num_elem, size_x, size_y));
    }
    let gpu_bufs = diffusion_gpu.as_ref().unwrap();
    let cube_dim = 32;

    // Upload the CPU-authoritative field.
    let data = cubecl::bytes::Bytes::from_bytes_vec(
        f32::as_bytes(&surface.soil_moisture.buffer).to_vec(),
    );
    client.write(&gpu_bufs.input, data);

    unsafe {
        diffusion_kernel_fe::launch_unchecked::<f32>(
            client,
            CubeCount::Static(
                gpu_bufs.size_x as u32 / cube_dim,
                gpu_bufs.size_y as u32 / cube_dim,
                1,
            ),
            CubeDim::new_2d(cube_dim, cube_dim),
            BufferArg::from_raw_parts(gpu_bufs.input.clone(), num_elem),
            BufferArg::from_raw_parts(gpu_bufs.output.clone(), num_elem),
            time.delta_secs(),
            f32::new(surface.soil_moisture.h),
            f32::new(params.ground.soil_water.diffusivity),
            match params.ground.soil_water.bc {
                BoundaryCondition::Dirichlet(_) => 0,
                BoundaryCondition::Closed => 1,
            },
            match params.ground.soil_water.bc {
                BoundaryCondition::Dirichlet(val) => val,
                BoundaryCondition::Closed => 0.0,
            },
            gpu_bufs.size_x,
            gpu_bufs.size_y,
        )
    };

    // Back to CPU, as before (organisms read `soil_moisture` every tick).
    let bytes = client.read_one_unchecked(gpu_bufs.output.clone());
    surface.soil_moisture.buffer = f32::from_bytes(&bytes).to_vec();

    // GPU-only presentation: texels <- output, then hand the buffer to the renderer.
    unsafe {
        gpu::field_values_to_texels::launch_unchecked::<f32>(
            client,
            CubeCount::Static(
                gpu_bufs.size_x as u32 / cube_dim,
                gpu_bufs.size_y as u32 / cube_dim,
                1,
            ),
            CubeDim::new_2d(cube_dim, cube_dim),
            BufferArg::from_raw_parts(gpu_bufs.output.clone(), num_elem),
            BufferArg::from_raw_parts(gpu_bufs.texels.clone(), num_elem * 4),
            gpu_bufs.size_x,
            gpu_bufs.size_y,
        )
    };

    let managed = client
        .get_resource::<cubecl::wgpu::WgpuServer<cubecl::wgpu::AutoCompiler>>(
            gpu_bufs.texels.clone(),
        )
        .expect("cubecl buffer should still be alive");
    let resource = managed.resource();
    field_vis_gpu.soil.buffer = Some(resource.buffer.clone());
    field_vis_gpu.soil.offset = resource.offset;
    field_vis_gpu.soil.width = gpu_bufs.size_x as u32;
    field_vis_gpu.soil.height = gpu_bufs.size_y as u32;
    if field_vis_gpu.image != terrain_assets.field_vis_image {
        field_vis_gpu.image = terrain_assets.field_vis_image.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::{CubeCount, CubeDim};

    fn gaussian_field(size: usize, h: f32, sigma: f32, center: f32) -> Vec<f32> {
        let mut buf = vec![0.0; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = (x as f32 - center) * h;
                let dy = (y as f32 - center) * h;
                let r2 = dx * dx + dy * dy;
                buf[x + y * size] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
        buf
    }

    /// Runs one explicit diffusion step of `dt` on the given field via the cubecl CPU
    /// runtime (same `#[cube]` kernel as the sim).
    fn diffuse(client: &Client, field: &mut Vec<f32>, size: usize, h: f32, dt: f32, d: f32) {
        let n = field.len();
        let input = client.empty(n * core::mem::size_of::<f32>());
        let output = client.empty(n * core::mem::size_of::<f32>());
        let data = cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(field).to_vec());
        client.write(&input, data);
        let cube_dim = 32;
        unsafe {
            diffusion_kernel_fe::launch_unchecked::<f32>(
                client,
                CubeCount::Static(size as u32 / cube_dim, size as u32 / cube_dim, 1),
                CubeDim::new_2d(cube_dim, cube_dim),
                BufferArg::from_raw_parts(input, n),
                BufferArg::from_raw_parts(output.clone(), n),
                dt,
                f32::new(h),
                f32::new(d),
                0, // Dirichlet
                0.0,
                size,
                size,
            );
        }
        let bytes = client.read_one_unchecked(output);
        *field = f32::from_bytes(&bytes).to_vec();
    }

    /// 2D heat-equation solution: `u = (s0^2 / s^2) exp(-r^2 / 2 s^2)` with
    /// `s^2 = s0^2 + 2 D t`.
    fn analytic(size: usize, h: f32, sigma0: f32, d: f32, t: f32, center: f32) -> Vec<f32> {
        let s2 = sigma0 * sigma0 + 2.0 * d * t;
        let factor = sigma0 * sigma0 / s2;
        let mut buf = vec![0.0; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = (x as f32 - center) * h;
                let dy = (y as f32 - center) * h;
                let r2 = dx * dx + dy * dy;
                buf[x + y * size] = factor * (-r2 / (2.0 * s2)).exp();
            }
        }
        buf
    }

    fn relative_l2(sim: &[f32], analytic: &[f32]) -> f32 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (&a, &b) in sim.iter().zip(analytic.iter()) {
            num += (a - b) * (a - b);
            den += b * b;
        }
        (num / den.max(1e-12)).sqrt()
    }

    /// The explicit 5-point scheme is second-order in space: halving the grid spacing
    /// (keeping the same physical time and dt) should shrink the error ~4x. This verifies
    /// both the kernel logic and the cubecl CPU test harness.
    #[test]
    fn gaussian_diffusion_converges_second_order_in_h() {
        let client = cubecl::Device::cpu().unwrap().client();

        let d = 0.01;
        let dt = 1.0 / 60.0;
        const STEPS: usize = 60; // 1 s of simulated time

        let mut errors = Vec::new();
        for &size in &[64usize, 128usize] {
            let h = 64.0 / size as f32; // physical domain size is fixed
            // Same *physical* initial blob on both grids (in cells this differs).
            let sigma0 = 3.0 / h;
            let center = 32.0 / h;
            let mut field = gaussian_field(size, h, sigma0, center);
            for _ in 0..STEPS {
                diffuse(&client, &mut field, size, h, dt, d);
            }
            let exact = analytic(size, h, sigma0, d, STEPS as f32 * dt, center);
            errors.push(relative_l2(&field, &exact));
        }

        // coarser grid must have a clearly larger error, ratio bounded ~4 (2nd order).
        let ratio = errors[0] / errors[1];
        assert!(
            errors[0] > 0.0 && ratio > 2.0,
            "expected ~4x error drop when halving h, got ratio {ratio} (errors {errors:?})"
        );
        assert!(
            errors[1] < 0.02,
            "fine-grid L2 error should be small, got {:?}",
            errors
        );
    }
}
