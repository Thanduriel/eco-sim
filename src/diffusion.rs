use bevy::prelude::*;
use cubecl::prelude::*;

use crate::domain;
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
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }
    let x = ABSOLUTE_POS_X as usize;
    let y = ABSOLUTE_POS_Y as usize;

    let center = input[x + y * size_x];

    let boundary_val = if bc_type == 0 {
        bc_dirichlet
    } else {
        center
    };

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
        + dt * diffusion_coefficient * (left + right + down + up - F::new(4.0_f32) * center)
            / (h * h);
}

fn launch(
    client: &Client,
    field: &mut domain::Field<f32>,
    dt: f32,
    diffusion_params: &parameters::DiffusionParameters,
) {
    let num_elem = field.buffer.len();
    let output_handle = client.empty(num_elem * core::mem::size_of::<f32>());
    let input_handle = client.create_from_slice(f32::as_bytes(&field.buffer));
    let cube_dim = 32;

    unsafe {
        diffusion_kernel_fe::launch_unchecked::<f32>(
            client,
            CubeCount::Static(
                field.size.x as u32 / cube_dim,
                field.size.y as u32 / cube_dim,
                1,
            ),
            CubeDim::new_2d(cube_dim, cube_dim),
            BufferArg::from_raw_parts(input_handle, num_elem),
            BufferArg::from_raw_parts(output_handle.clone(), num_elem),
            dt,
            f32::new(field.h),
            f32::new(diffusion_params.diffusivity),
            match diffusion_params.bc {
                BoundaryCondition::Dirichlet(_) => 0,
                BoundaryCondition::Closed => 1,
            },
            match diffusion_params.bc {
                BoundaryCondition::Dirichlet(val) => val,
                BoundaryCondition::Closed => 0.0,
            },
            field.size.x,
            field.size.y,
        )
    };

    let bytes = client.read_one_unchecked(output_handle);
    field.buffer = f32::from_bytes(&bytes).to_vec();
}

pub fn run_cubecl(
    time: Res<Time>,
    mut surface_query: Query<&mut terrain::Surface>,
    params: Res<parameters::GeneralParameters>,
) {
    if !params.ground.soil_water.is_active {
        return;
    }

    let mut surface = surface_query.single_mut().unwrap();
    let client = cubecl::Device::cuda(0).unwrap().client();
    launch(
        &client,
        &mut surface.soil_moisture,
        time.delta_secs(),
        &params.ground.soil_water,
    );
}
