//! Particle water skeleton (Phase 1): gravity + terrain contact + world walls.
//!
//! All state stays on the GPU via cubecl on Bevy's device (see `gpu.rs`). The integrator is
//! semi-implicit Euler (symplectic for position-independent forces) with a constant inner
//! step: each physics tick runs `params.water.substeps` fixed sub-steps of
//! `dt = (1 / PHYSICS_TICKS_PER_SEC) / substeps`.
//!
//! Phase 1 has no particle-particle forces (that is Phase 2: density/pressure, SPH
//! kernels, neighbor grid). The debug visualization is a per-grid-cell "max particle
//! height" field, written with atomics and displayed via the F5 field overlay
//! (existing `Rgba32Float` field-vis path).

use bevy::prelude::*;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::gpu;
use crate::parameters;
use crate::terrain;

/// Persistent GPU particle state (double-buffered for ping-pong between steps).
#[derive(Resource)]
pub struct WaterParticles {
    pub num: usize,
    /// 4 floats per particle: [x, y, z, unused].
    pub pos_a: Handle,
    pub pos_b: Handle,
    pub vel_a: Handle,
    pub vel_b: Handle,
    /// Terrain height map (grid points at `i * h`, Vertex convention), uploaded once.
    pub terrain: Handle,
    pub terrain_x: usize,
    pub terrain_y: usize,
    /// Debug splat: per-cell particle count (`u32`, atomics) ...
    pub splat_counts: Handle,
    /// ... converted into an `Rgba32Float` texel stream for the field overlay
    /// ([normalized count, 0, 0, 0]).
    pub splat: Handle,
    pub cells: usize,
    /// True when the *b* buffers hold the latest state.
    pub current_is_b: bool,
}

/// One fixed integration sub-step. No particle-particle interaction yet: gravity,
/// world walls and the terrain height field (bilinear height + gradient normal).
#[cube(launch_unchecked)]
fn particle_step<F: Float + LaunchArg>(
    pos_in: &[F],
    vel_in: &[F],
    pos_out: &mut [F],
    vel_out: &mut [F],
    terrain: &[F],
    dt: F,
    gravity: F,
    domain_size: F,
    friction: F,
    tx: usize,
    ty: usize,
    num: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }

    let p = i * 4;
    let mut x = pos_in[p];
    let mut y = pos_in[p + 1];
    let mut z = pos_in[p + 2];
    let mut vx = vel_in[p];
    let mut vy = vel_in[p + 1];
    let mut vz = vel_in[p + 2];

    // Semi-implicit Euler (y-up world).
    vy = vy - gravity * dt;
    x = x + vx * dt;
    y = y + vy * dt;
    z = z + vz * dt;

    // World walls (closed box).
    if x < F::new(0.0_f32) {
        x = F::new(0.0_f32);
        vx = F::new(0.0_f32);
    }
    if x > domain_size {
        x = domain_size;
        vx = F::new(0.0_f32);
    }
    if z < F::new(0.0_f32) {
        z = F::new(0.0_f32);
        vz = F::new(0.0_f32);
    }
    if z > domain_size {
        z = domain_size;
        vz = F::new(0.0_f32);
    }

    // Terrain: bilinear height and gradient at (x, z).
    let h = domain_size / F::cast_from(tx as f32);
    let max_x = F::cast_from((tx - 1) as f32);
    let max_z = F::cast_from((ty - 1) as f32);
    let fx = x / h;
    let fz = z / h;
    let frac_x = clamp(fx - fx.floor(), F::new(0.0_f32), F::new(1.0_f32));
    let frac_z = clamp(fz - fz.floor(), F::new(0.0_f32), F::new(1.0_f32));
    let x0 = usize::cast_from(clamp(fx, F::new(0.0_f32), max_x));
    let x1 = usize::cast_from(clamp(fx + F::new(1.0_f32), F::new(0.0_f32), max_x));
    let z0 = usize::cast_from(clamp(fz, F::new(0.0_f32), max_z));
    let z1 = usize::cast_from(clamp(fz + F::new(1.0_f32), F::new(0.0_f32), max_z));

    let b00 = terrain[x0 + z0 * tx];
    let b10 = terrain[x1 + z0 * tx];
    let b01 = terrain[x0 + z1 * tx];
    let b11 = terrain[x1 + z1 * tx];

    let b = b00 * (F::new(1.0_f32) - frac_x) * (F::new(1.0_f32) - frac_z)
        + b10 * frac_x * (F::new(1.0_f32) - frac_z)
        + b01 * (F::new(1.0_f32) - frac_x) * frac_z
        + b11 * frac_x * frac_z;

    // Surface gradient (zero at the domain edge where the indices clamp to one point).
    let gx = (b10 - b00) / h;
    let gz = (b01 - b00) / h;

    // Contact: push out of the terrain and remove the inward normal velocity.
    let eps = F::new(0.001_f32);
    if y < b + eps {
        y = b + eps;
        // Normal from the heightfield (terrain heights are scaled by 2.5 in world units,
        // so amplify the gradient slope).
        let nx = -F::new(2.0_f32) * gx;
        let nz = -F::new(2.0_f32) * gz;
        let n_mag = (nx * nx + F::new(1.0_f32) + nz * nz).sqrt();
        let nx = nx / n_mag;
        let ny = F::new(1.0_f32) / n_mag;
        let nz = nz / n_mag;

        let vn = vx * nx + vy * ny + vz * nz;
        if vn < F::new(0.0_f32) {
            vx = vx - vn * nx;
            vy = vy - vn * ny;
            vz = vz - vn * nz;
        }
        // Tangential friction on contact.
        let damp = (F::new(1.0_f32) - friction * dt).max(F::new(0.0_f32));
        vx = vx * damp;
        vy = vy * damp;
        vz = vz * damp;
    }

    pos_out[p] = x;
    pos_out[p + 1] = y;
    pos_out[p + 2] = z;
    pos_out[p + 3] = F::new(0.0_f32);
    vel_out[p] = vx;
    vel_out[p + 1] = vy;
    vel_out[p + 2] = vz;
    vel_out[p + 3] = F::new(0.0_f32);
}

/// Zero out the integer per-cell counters (one launch, one workgroup thread per cell).
#[cube(launch_unchecked)]
fn splat_clear<I: Int + LaunchArg>(splat: &mut [I], cells: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= cells * cells {
        terminate!();
    }
    splat[i] = I::new(0);
}

/// Scatter particles into cells: increment the per-cell counter with an integer atomic
/// (float atomics are not supported by naga/wgpu-30, integer adds are). This kernel is
/// not covered by the CPU-runtime unit tests (the CPU runtime does not implement
/// atomics either).
#[cube(launch_unchecked)]
fn particle_to_splat<I: Int + LaunchArg, F: Float + LaunchArg>(
    pos: &[F],
    atomics: &[Atomic<I>],
    domain_size: F,
    cells: F,
    num: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }

    let p = i * 4;
    let x = pos[p];
    let z = pos[p + 2];

    let cx = clamp(
        x / domain_size * cells,
        F::new(0.0_f32),
        cells - F::new(1.0_f32),
    );
    let cz = clamp(
        z / domain_size * cells,
        F::new(0.0_f32),
        cells - F::new(1.0_f32),
    );
    let cell = usize::cast_from(cx) + usize::cast_from(cz) * usize::cast_from(cells);
    atomics[cell].fetch_add(I::new(1));
}

/// Convert per-cell counts into the `Rgba32Float` texel stream shown by the overlay
/// ([min(count / scale, 1), 0, 0, 0]).
#[cube(launch_unchecked)]
fn counts_to_texels<I: Int + LaunchArg, F: Float + LaunchArg>(
    counts: &[I],
    texels: &mut [F],
    scale: F,
    cells: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= cells * cells {
        terminate!();
    }
    let p = i * 4;
    let v = clamp(
        F::cast_from(counts[i]) / scale,
        F::new(0.0_f32),
        F::new(1.0_f32),
    );
    texels[p] = v;
    texels[p + 1] = F::new(0.0_f32);
    texels[p + 2] = F::new(0.0_f32);
    texels[p + 3] = F::new(0.0_f32);
}

/// Spawns the initial water blob and uploads the (static) terrain height map to the GPU.
pub fn setup_water(
    mut commands: Commands,
    gpu_context: Res<gpu::GpuContext>,
    terrain_query: Query<&terrain::Terrain>,
    parameters: Res<parameters::GeneralParameters>,
) {
    let Ok(terrain) = terrain_query.single() else {
        return;
    };
    let params = &parameters.water;
    let client = &gpu_context.client;

    let (sx, sy, sz) = (params.blob_size[0], params.blob_size[1], params.blob_size[2]);
    let s = params.particle_spacing;
    let (nx, ny, nz) = (
        (sx / s).ceil() as usize,
        (sy / s).ceil() as usize,
        (sz / s).ceil() as usize,
    );
    let num = nx * ny * nz;
    let cx = crate::domain::CENTER.x;
    let cz = crate::domain::CENTER.y;
    let base_y = 1.5;

    let mut pos = Vec::with_capacity(num * 4);
    for y in 0..ny {
        for z in 0..nz {
            for x in 0..nx {
                pos.push(cx - sx / 2.0 + x as f32 * s);
                pos.push(base_y + y as f32 * s);
                pos.push(cz - sz / 2.0 + z as f32 * s);
                pos.push(0.0);
            }
        }
    }
    let vel = vec![0.0f32; num * 4];

    let bytes = |data: &Vec<f32>| {
        cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(data).to_vec())
    };

    let pos_a = client.empty((num * 4) as usize * core::mem::size_of::<f32>());
    let pos_b = client.empty((num * 4) as usize * core::mem::size_of::<f32>());
    let vel_a = client.empty((num * 4) as usize * core::mem::size_of::<f32>());
    let vel_b = client.empty((num * 4) as usize * core::mem::size_of::<f32>());
    client.write(&pos_a, bytes(&pos));
    client.write(&vel_a, bytes(&vel));

    let terrain_size = terrain.height_map.size;
    let terrain_bytes = terrain.height_map.buffer.len() * core::mem::size_of::<f32>();
    let terrain_handle = client.empty(terrain_bytes);
    client.write(
        &terrain_handle,
        bytes(&terrain.height_map.buffer),
    );

    let cells = 256usize;
    let splat_counts = client.empty(cells * cells * core::mem::size_of::<u32>());
    let splat = client.empty(cells * cells * 4 * core::mem::size_of::<f32>());

    commands.insert_resource(WaterParticles {
        num,
        pos_a,
        pos_b,
        vel_a,
        vel_b,
        terrain: terrain_handle,
        terrain_x: terrain_size.x,
        terrain_y: terrain_size.y,
        splat_counts,
        splat,
        cells,
        current_is_b: false,
    });
}

/// Phase 1 main loop: `substeps` fixed integration steps per physics tick, then refresh
/// the debug splat and publish the buffer for rendering.
pub fn run_water(
    params: Res<parameters::GeneralParameters>,
    gpu_context: Res<gpu::GpuContext>,
    terrain_assets: Res<terrain::TerrainAssets>,
    mut field_vis_gpu: ResMut<gpu::FieldVisGpuData>,
    particles: Option<ResMut<WaterParticles>>,
) {
    let Some(particles) = particles else { return };
    if !params.water.is_active {
        return;
    }
    let client = &gpu_context.client;
    let mut p = particles;

    let sub_dt =
        1.0 / parameters::PHYSICS_TICKS_PER_SEC as f32 / params.water.substeps.max(1) as f32;
    let dim = 64u32;

    let mut use_b = p.current_is_b;
    for _ in 0..params.water.substeps.max(1) {
        let (pos_in, vel_in, pos_out, vel_out) = if use_b {
            (&p.pos_b, &p.vel_b, &p.pos_a, &p.vel_a)
        } else {
            (&p.pos_a, &p.vel_a, &p.pos_b, &p.vel_b)
        };
        unsafe {
            particle_step::launch_unchecked::<f32>(
                client,
                CubeCount::Static(
                    p.num.div_ceil(dim as usize) as u32,
                    1,
                    1,
                ),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(pos_in.clone(), p.num * 4),
                BufferArg::from_raw_parts(vel_in.clone(), p.num * 4),
                BufferArg::from_raw_parts(pos_out.clone(), p.num * 4),
                BufferArg::from_raw_parts(vel_out.clone(), p.num * 4),
                BufferArg::from_raw_parts(p.terrain.clone(), p.terrain_x * p.terrain_y),
                sub_dt,
                9.81_f32,
                crate::domain::SIZE_F32.x,
                params.water.terrain_friction,
                p.terrain_x,
                p.terrain_y,
                p.num,
            )
        };
        use_b = !use_b;
    }
    p.current_is_b = use_b;

    // Debug splat: clear counts, scatter particles, convert to texels.
    let (cur_pos, _cur_vel) = if p.current_is_b {
        (&p.pos_b, &p.vel_b)
    } else {
        (&p.pos_a, &p.vel_a)
    };
    unsafe {
        splat_clear::launch_unchecked::<u32>(
            client,
            CubeCount::Static((p.cells * p.cells).div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.splat_counts.clone(), p.cells * p.cells),
            p.cells,
        );
    }
    unsafe {
        particle_to_splat::launch_unchecked::<u32, f32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(cur_pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.splat_counts.clone(), p.cells * p.cells),
            crate::domain::SIZE_F32.x,
            p.cells as f32,
            p.num,
        );
    }
    unsafe {
        counts_to_texels::launch_unchecked::<u32, f32>(
            client,
            CubeCount::Static((p.cells * p.cells).div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.splat_counts.clone(), p.cells * p.cells),
            BufferArg::from_raw_parts(p.splat.clone(), p.cells * p.cells * 4),
            params.water.debug_count_scale,
            p.cells,
        );
    }

    let managed = client
        .get_resource::<cubecl::wgpu::WgpuServer<cubecl::wgpu::AutoCompiler>>(
            p.splat.clone(),
        )
        .expect("cubecl buffer should still be alive");
    let resource = managed.resource();
    field_vis_gpu.water.buffer = Some(resource.buffer.clone());
    field_vis_gpu.water.offset = resource.offset;
    field_vis_gpu.water.width = p.cells as u32;
    field_vis_gpu.water.height = p.cells as u32;
    if field_vis_gpu.image != terrain_assets.field_vis_image {
        field_vis_gpu.image = terrain_assets.field_vis_image.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::{CubeCount, CubeDim};

    /// Runs `steps` integration sub-steps of `dt` on a single particle via the cubecl CPU
    /// runtime (same `#[cube]` kernel as the sim). Ping-pong buffers are used like the
    /// sim does. `terrain` is a `tx x ty` height map.
    fn integrate(
        client: &Client,
        pos: &mut [f32],
        vel: &mut [f32],
        terrain: &[f32],
        tx: usize,
        ty: usize,
        dt: f32,
        steps: usize,
    ) {
        let n = pos.len();
        let pos_a = client.empty(n * core::mem::size_of::<f32>());
        let pos_b = client.empty(n * core::mem::size_of::<f32>());
        let vel_a = client.empty(n * core::mem::size_of::<f32>());
        let vel_b = client.empty(n * core::mem::size_of::<f32>());
        let terr = client.empty(terrain.len() * core::mem::size_of::<f32>());
        let bytes = |d: &[f32]| {
            cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(d).to_vec())
        };
        client.write(&pos_a, bytes(pos));
        client.write(&vel_a, bytes(vel));
        client.write(&terr, bytes(terrain));

        let dim = 64u32;
        let mut use_b = false;
        for _ in 0..steps {
            let (pi, vi, po, vo) = if use_b {
                (&pos_b, &vel_b, &pos_a, &vel_a)
            } else {
                (&pos_a, &vel_a, &pos_b, &vel_b)
            };
            unsafe {
                particle_step::launch_unchecked::<f32>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(pi.clone(), n),
                    BufferArg::from_raw_parts(vi.clone(), n),
                    BufferArg::from_raw_parts(po.clone(), n),
                    BufferArg::from_raw_parts(vo.clone(), n),
                    BufferArg::from_raw_parts(terr.clone(), tx * ty),
                    dt,
                    9.81,
                    64.0,
                    3.0,
                    tx,
                    ty,
                    1,
                );
            }
            use_b = !use_b;
        }
        let out = if use_b { pos_b } else { pos_a };
        let vout = if use_b { vel_b } else { vel_a };
        let bytes = client.read_one_unchecked(out);
        pos.copy_from_slice(&f32::from_bytes(&bytes));
        let bytes = client.read_one_unchecked(vout);
        vel.copy_from_slice(&f32::from_bytes(&bytes));
    }

    /// Semi-implicit Euler free fall: y_n = y0 - g*dt^2*n(n+1)/2, v_n = -g*dt*n.
    #[test]
    fn free_fall_matches_closed_form() {
        let client = cubecl::Device::cpu().unwrap().client();
        let mut pos = [32.0f32, 100.0, 32.0, 0.0];
        let mut vel = [0.0f32; 4];
        let terrain = vec![0.0f32; 8 * 8];
        let dt = 0.01;
        let n = 40;

        integrate(&client, &mut pos, &mut vel, &terrain, 8, 8, dt, n);

        let expected_y = 100.0 - 9.81 * dt * dt * (n as f32 * (n as f32 + 1.0) / 2.0);
        let expected_vy = -9.81 * n as f32 * dt;
        assert!(
            (pos[1] - expected_y).abs() < 1e-3,
            "y={} expected {}",
            pos[1],
            expected_y
        );
        assert!(
            (vel[1] - expected_vy).abs() < 1e-3,
            "vy={} expected {}",
            vel[1],
            expected_vy
        );
        assert_eq!(pos[0], 32.0);
        assert_eq!(pos[2], 32.0);
    }

    /// A particle dropped on flat ground must come to rest at the surface with a tiny
    /// residual velocity (contact projection + friction).
    #[test]
    fn settles_on_flat_terrain() {
        let client = cubecl::Device::cpu().unwrap().client();
        let mut pos = [32.0f32, 3.0, 32.0, 0.0];
        let mut vel = [0.0f32; 4];
        // Flat ground at b = 1.0 (grid points every 8 world units; all same height).
        let terrain = vec![1.0f32; 8 * 8];
        let dt = 1.0 / 60.0 / 4.0;
        let steps = 4 * 60 * 4; // 4 simulated seconds

        integrate(&client, &mut pos, &mut vel, &terrain, 8, 8, dt, steps);

        assert!(
            (pos[1] - 1.0).abs() < 0.02,
            "rest y={} (expected ~1.0)",
            pos[1]
        );
        assert!(
            (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt() < 0.05,
            "residual speed {:?}",
            vel
        );
    }

    /// Particles never leave the closed world box.
    #[test]
    fn wall_keeps_particles_inside() {
        let client = cubecl::Device::cpu().unwrap().client();
        let mut pos = [0.5f32, 10.0, 63.5, 0.0];
        let mut vel = [-50.0f32, 0.0, 50.0, 0.0];
        let terrain = vec![0.0f32; 8 * 8];
        let dt = 1.0 / 60.0 / 4.0;
        let steps = 4 * 60 * 2;

        integrate(&client, &mut pos, &mut vel, &terrain, 8, 8, dt, steps);

        assert!(pos[0] >= 0.0, "x crossed the wall: {}", pos[0]);
        assert!(pos[2] <= 64.0, "z crossed the wall: {}", pos[2]);
        assert!(pos[1] >= 0.0, "fell through the floor: {}", pos[1]);
        assert!(
            (pos[0] - 0.0).abs() < 0.02 && (pos[2] - 64.0).abs() < 0.02,
            "should rest against the walls, pos={:?}",
            pos
        );
    }
}
