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
 //   let _ = future::block_on(client.sync());
  //  println!("upload: {}s", start_upload.elapsed().as_secs_f64());

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
    //let _ = future::block_on(client.sync());
    //println!("compute: {}s", start_computation.elapsed().as_secs_f64());

    let start_download = std::time::Instant::now();
    let bytes = client.read_one(output_handle).unwrap();
    let output = F::from_bytes(&bytes);
    field.buffer = Vec::from(output);
    //field.buffer = bytes.try_into_vec().unwrap();

    //println!("download: {}s", start_download.elapsed().as_secs_f64());
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
    launch::<cubecl::cuda::CudaRuntime, f32>(
        &Default::default(),
        &mut surface.soil_moisture,
        time.delta_secs(),
        &params.ground.soil_water,
    );
}
