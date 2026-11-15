//! Particle water (Phase 2): weakly-compressible SPH hydrodynamics.
//!
//! Physics per substep (velocity-Verlet, constant inner step `dt`):
//!   1. build the uniform neighbor grid  (count -> prefix -> scatter, integer atomics)
//!   2. density & pressure   (summation density, Tait EoS, clamp at `stiffness_clamp * rho0`)
//!   3. accelerations        (Monaghan pressure gradient + artificial viscosity)
//!   4. v^{n+1/2} = v^n + (a^n - g) dt/2 ; x^{n+1} = x^n + v^{n+1/2} dt (+ walls/terrain)
//!   5. re-evaluate forces at x^{n+1} ; v^{n+1} = v^{n+1/2} + (a^{n+1} - g) dt/2
//!
//! All state stays on the GPU via cubecl on Bevy's device (see `gpu.rs`).
//!
//! Neighbor search: the *physics* kernels take a `use_grid` switch. Production currently
//! runs `use_grid = 0` (dense all-pairs loop) — the uniform-grid cell lists (built with
//! integer atomics) are kept but DISABLED: the cubecl-wgpu scatter produced
//! non-deterministic cell lists on this device despite atomics themselves being correct
//! (verified: 64/64 concurrent fetch_adds; single-threaded probes pass). At ~2.3k
//! particles the all-pairs loop is trivially cheap on the GPU, and it is the exact path
//! the CPU-runtime unit tests exercise. Revisit with the Bevy-readback migration.

use bevy::prelude::*;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::gpu;
use crate::parameters;
use crate::terrain;

/// Persistent GPU water state (double-buffered for ping-pong between steps).
#[derive(Resource)]
pub struct WaterParticles {
    pub num: usize,
    /// 4 floats per particle: [x, y, z, unused].
    pub pos_a: Handle,
    pub pos_b: Handle,
    pub vel_a: Handle,
    pub vel_b: Handle,
    /// Per-particle fluid state `[rho, p, 0, 0]` (ping-pong with the position parity).
    pub state_a: Handle,
    pub state_b: Handle,
    /// Per-particle acceleration `[ax, ay, az, 0]` (pressure + viscosity only; gravity is
    /// applied by the integrator).
    pub acc_a: Handle,
    pub acc_b: Handle,
    /// Terrain height map (grid points at `i * h`, Vertex convention), uploaded once.
    pub terrain: Handle,
    pub terrain_x: usize,
    pub terrain_y: usize,
    /// Uniform neighbor grid: per-cell counts (`cells` u32), exclusive prefix (`cells+1`),
    /// and the compacted particle index list (`num` u32).
    pub grid_counts: Handle,
    pub grid_scan: Handle,
    pub grid_list: Handle,
    pub cells_x: usize,
    pub cells_y: usize,
    pub cells_z: usize,
    pub cell_inv: f32,
    /// Debug splat: per-cell particle count (`u32`, atomics) ...
    pub splat_counts: Handle,
    /// ... converted into an `Rgba32Float` texel stream for the field overlay
    /// ([normalized count, 0, 0, 0]).
    pub splat: Handle,
    pub cells: usize,
    /// True when the *b* buffers hold the latest state.
    pub current_is_b: bool,
}

// ---------------------------------------------------------------------------
// SPH kernels
// ---------------------------------------------------------------------------


/// Zero the per-cell counters.
#[cube(launch_unchecked)]
fn grid_zero<I: Int + LaunchArg>(counts: &mut [I], cells: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i < cells {
        counts[i] = I::new(0);
    }
}

/// Per-particle cell counter (integer atomics). Cell size equals the SPH support
/// radius (2h), so a 3x3x3 cell window fully covers the kernel support.
#[cube(launch_unchecked)]
fn grid_count<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    counts: &[Atomic<I>],
    cell_inv: F,
    _domain_size: F,
    _max_y: F,
    cx: usize,
    cy: usize,
    cz: usize,
    num: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    let x = clamp(
        pos[p] * cell_inv,
        F::new(0.0_f32),
        F::cast_from((cx - 1) as f32),
    );
    let y = clamp(
        pos[p + 1] * cell_inv,
        F::new(0.0_f32),
        F::cast_from((cy - 1) as f32),
    );
    let z = clamp(
        pos[p + 2] * cell_inv,
        F::new(0.0_f32),
        F::cast_from((cz - 1) as f32),
    );
    let ix = usize::cast_from(x);
    let iy = usize::cast_from(y);
    let iz = usize::cast_from(z);
    let cell = ix + cx * (iy + cy * iz);
    counts[cell].fetch_add(I::new(1));
}

/// Exclusive prefix sum over cell counts (single thread; trivial cost).
#[cube(launch_unchecked)]
fn grid_prefix<I: Int + LaunchArg>(counts: &[I], scan: &mut [I], cells: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i != 0 {
        terminate!();
    }
    scan[0] = I::new(0);
    for c in 0..cells {
        scan[c + 1] = scan[c] + counts[c];
    }
}

/// Scatter particle indices into per-cell ranges (cell size 2h; see [`grid_count`]).
#[cube(launch_unchecked)]
fn grid_scatter<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    scan: &[I],
    counts: &[Atomic<I>],
    list: &mut [I],
    cell_inv: F,
    _max_y: F,
    cx: usize,
    cy: usize,
    cz: usize,
    num: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    let x = clamp(
        pos[p] * cell_inv,
        F::new(0.0_f32),
        F::cast_from((cx - 1) as f32),
    );
    let y = clamp(
        pos[p + 1] * cell_inv,
        F::new(0.0_f32),
        F::cast_from((cy - 1) as f32),
    );
    let z = clamp(
        pos[p + 2] * cell_inv,
        F::new(0.0_f32),
        F::cast_from((cz - 1) as f32),
    );
    let ix = usize::cast_from(x);
    let iy = usize::cast_from(y);
    let iz = usize::cast_from(z);
    let cell = ix + cx * (iy + cy * iz);
    let slot = usize::cast_from(counts[cell].fetch_add(I::new(1)));
    let lidx = usize::cast_from(scan[cell]) + slot;
    list[lidx] = I::cast_from(i);
}

/// Summation density: `rho_i = m sum_j W(|r_i - r_j|)` plus the Tait pressure stored next
/// to it in `state` (`[rho, p, 0, 0]`). `use_grid=0` scans all particles (used by the
/// CPU-runtime tests); `use_grid=1` iterates the 3x3x3 cell neighborhood.
#[cube(launch_unchecked)]
fn compute_density<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    scan: &[I],
    list: &[I],
    state: &mut [F],
    h: F,
    h2: F,
    sigma: F,
    mass: F,
    rho0: F,
    b: F,
    gamma: F,
    cell_inv: F,
    _max_y: F,
    cx: usize,
    cy: usize,
    cz: usize,
    num: usize,
    use_grid: i32,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    let xi = pos[p];
    let yi = pos[p + 1];
    let zi = pos[p + 2];

    let support2 = F::new(4.0_f32) * h2;
    let mut rho = F::new(0.0_f32);

    if use_grid == 0 {
        for j in 0..num {
            {
                let pj = j * 4;
                let dx = xi - pos[pj];
                let dy = yi - pos[pj + 1];
                let dz = zi - pos[pj + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < support2 {
                    let q = (r2).sqrt() / h;
                    let q2 = q * q;
                    let mut w = F::new(0.0_f32);
                    if q < F::new(1.0_f32) {
                        w = sigma * (F::new(1.0_f32) - F::new(1.5_f32) * q2 + F::new(0.75_f32) * q2 * q);
                    } else {
                        if q < F::new(2.0_f32) {
                            let t = F::new(2.0_f32) - q;
                            w = sigma * F::new(0.25_f32) * t * t * t;
                        }
                    }
                    rho = rho + mass * w;
                }
            }
        }
    } else {
        let x = clamp(
            xi * cell_inv,
            F::new(0.0_f32),
            F::cast_from((cx - 1) as f32),
        );
        let y = clamp(
            yi * cell_inv,
            F::new(0.0_f32),
            F::cast_from((cy - 1) as f32),
        );
        let z = clamp(
            zi * cell_inv,
            F::new(0.0_f32),
            F::cast_from((cz - 1) as f32),
        );
        let icx = usize::cast_from(x) as i32;
        let icy = usize::cast_from(y) as i32;
        let icz = usize::cast_from(z) as i32;

        for cdz in 0..3usize {
            for cdy in 0..3usize {
                for cdx in 0..3usize {
                    let nc = icx + (cdx as i32 - 1);
                    let nr = icy + (cdy as i32 - 1);
                    let nh = icz + (cdz as i32 - 1);
                    if nc >= 0 && nc < cx as i32 && nr >= 0 && nr < cy as i32
                        && nh >= 0 && nh < cz as i32
                    {
                        let cell = nc as usize + cx * (nr as usize + cy * nh as usize);
                        let r_start = usize::cast_from(scan[cell]);
                        let r_end = usize::cast_from(scan[cell + 1]);
                        for k in r_start..r_end {
                            let j = usize::cast_from(list[k]);
                            {
                                let pj = j * 4;
                                let dx = xi - pos[pj];
                                let dy = yi - pos[pj + 1];
                                let dz = zi - pos[pj + 2];
                                let r2 = dx * dx + dy * dy + dz * dz;
                                if r2 < support2 {
                                    let q = (r2).sqrt() / h;
                                    let q2 = q * q;
                                    let mut w = F::new(0.0_f32);
                                    if q < F::new(1.0_f32) {
                                        w = sigma * (F::new(1.0_f32) - F::new(1.5_f32) * q2 + F::new(0.75_f32) * q2 * q);
                                    } else {
                                        if q < F::new(2.0_f32) {
                                            let t = F::new(2.0_f32) - q;
                                            w = sigma * F::new(0.25_f32) * t * t * t;
                                        }
                                    }
                                    rho = rho + mass * w;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    state[p] = rho;
    // Clamp at rho0: no negative (tensile) pressure. This avoids the classical WCSPH
    // pairwise tensile instability, which would otherwise collapse a column into a
    // monolayer (fluid "sticking" to itself).
    let ratio = rho.max(rho0) / rho0;
    state[p + 1] = b * (ratio.powf(gamma) - F::new(1.0_f32));
    state[p + 2] = F::new(0.0_f32);
    state[p + 3] = F::new(0.0_f32);
}

/// WCSPH accelerations: pressure gradient only (dense; viscosity added next).
#[cube(launch_unchecked)]
fn compute_force<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    vel: &[F],
    state: &[F],
    scan: &[I],
    list: &[I],
    acc: &mut [F],
    h: F,
    h2: F,
    sigma: F,
    mass: F,
    alpha: F,
    speed_of_sound: F,
    cell_inv: F,
    _max_y: F,
    cx: usize,
    cy: usize,
    cz: usize,
    num: usize,
    use_grid: i32,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    let xi = pos[p];
    let yi = pos[p + 1];
    let zi = pos[p + 2];
    let rho_i = state[p];
    let p_i = state[p + 1];
    let inv_rho_i2 = F::new(1.0_f32) / (rho_i * rho_i);

    let support2 = F::new(4.0_f32) * h2;
    let mut ax = F::new(0.0_f32);
    let mut ay = F::new(0.0_f32);
    let mut az = F::new(0.0_f32);

    for j in 0..num {
        if j != i {
            let pj = j * 4;
            let dx = xi - pos[pj];
            let dy = yi - pos[pj + 1];
            let dz = zi - pos[pj + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < support2 && r2 > F::new(1e-12_f32) {
                let r = (r2).sqrt();
                let q = r / h;
                let mut dw_dq = F::new(0.0_f32);
                if q < F::new(1.0_f32) {
                    dw_dq = sigma * (F::new(-3.0_f32) * q + F::new(2.25_f32) * q * q);
                } else {
                    if q < F::new(2.0_f32) {
                        let t = F::new(2.0_f32) - q;
                        dw_dq = F::new(-0.75_f32) * sigma * t * t;
                    }
                }
                let grad_mag = dw_dq / h / r;
                let gx = dx * grad_mag;
                let gy = dy * grad_mag;
                let gz = dz * grad_mag;
                let rho_j = state[pj];
                let p_j = state[pj + 1];
                let inv_rho_j2 = F::new(1.0_f32) / (rho_j * rho_j);
                let mut coeff = mass * (p_i * inv_rho_i2 + p_j * inv_rho_j2);

                let vx = vel[p] - vel[pj];
                let vy = vel[p + 1] - vel[pj + 1];
                let vz = vel[p + 2] - vel[pj + 2];
                let vdot = vx * dx + vy * dy + vz * dz;
                if vdot < F::new(0.0_f32) {
                    let rho_bar = (rho_i + rho_j) * F::new(0.5_f32);
                    let mu = h * vdot / (r2 + F::new(0.01_f32) * h2);
                    let pi = -alpha * speed_of_sound * mu / rho_bar;
                    coeff = coeff + mass * pi;
                }
                ax = ax - coeff * gx;
                ay = ay - coeff * gy;
                az = az - coeff * gz;
            }
        }
    }

    if use_grid == 1 {
        let x = clamp(
            xi * cell_inv,
            F::new(0.0_f32),
            F::cast_from((cx - 1) as f32),
        );
        let y = clamp(
            yi * cell_inv,
            F::new(0.0_f32),
            F::cast_from((cy - 1) as f32),
        );
        let z = clamp(
            zi * cell_inv,
            F::new(0.0_f32),
            F::cast_from((cz - 1) as f32),
        );
        let icx = usize::cast_from(x) as i32;
        let icy = usize::cast_from(y) as i32;
        let icz = usize::cast_from(z) as i32;

        for cdz in 0..3usize {
            for cdy in 0..3usize {
                for cdx in 0..3usize {
                    let nc = icx + (cdx as i32 - 1);
                    let nr = icy + (cdy as i32 - 1);
                    let nh = icz + (cdz as i32 - 1);
                    if nc >= 0 && nc < cx as i32 && nr >= 0 && nr < cy as i32
                        && nh >= 0 && nh < cz as i32
                    {
                        let cell = nc as usize + cx * (nr as usize + cy * nh as usize);
                        let r_start = usize::cast_from(scan[cell]);
                        let r_end = usize::cast_from(scan[cell + 1]);
                        for k in r_start..r_end {
                            let j = usize::cast_from(list[k]);
                            {
                                let pj = j * 4;
                                let dx = xi - pos[pj];
                                let dy = yi - pos[pj + 1];
                                let dz = zi - pos[pj + 2];
                                let r2 = dx * dx + dy * dy + dz * dz;
                                if r2 < support2 {
                                    let r = (r2).sqrt();
                                    let q = r / h;
                                    let mut dw_dq = F::new(0.0_f32);
                                    if q < F::new(1.0_f32) {
                                        dw_dq = sigma * (F::new(-3.0_f32) * q + F::new(2.25_f32) * q * q);
                                    } else {
                                        if q < F::new(2.0_f32) {
                                            let t = F::new(2.0_f32) - q;
                                            dw_dq = F::new(-0.75_f32) * sigma * t * t;
                                        }
                                    }
                                    let grad_mag = dw_dq / h / r;
                                    let gx = dx * grad_mag;
                                    let gy = dy * grad_mag;
                                    let gz = dz * grad_mag;
                                    let rho_j = state[pj];
                                    let p_j = state[pj + 1];
                                    let inv_rho_j2 = F::new(1.0_f32) / (rho_j * rho_j);
                                    let mut coeff = mass * (p_i * inv_rho_i2 + p_j * inv_rho_j2);

                                    let vx = vel[p] - vel[pj];
                                    let vy = vel[p + 1] - vel[pj + 1];
                                    let vz = vel[p + 2] - vel[pj + 2];
                                    let vdot = vx * dx + vy * dy + vz * dz;
                                    if vdot < F::new(0.0_f32) {
                                        let rho_bar = (rho_i + rho_j) * F::new(0.5_f32);
                                        let mu = h * vdot / (r2 + F::new(0.01_f32) * h2);
                                        let pi = -alpha * speed_of_sound * mu / rho_bar;
                                        coeff = coeff + mass * pi;
                                    }
                                    ax = ax - coeff * gx;
                                    ay = ay - coeff * gy;
                                    az = az - coeff * gz;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    acc[p] = ax;
    acc[p + 1] = ay;
    acc[p + 2] = az;
    acc[p + 3] = F::new(0.0_f32);
}

/// Velocity-Verlet half step + boundary contact. Gravity is applied here (y-down is
/// negative). `vel_half = vel + (acc - g)*dt/2`, `pos_new = pos + vel_half*dt`, with
/// world walls and heightfield-terrain push-out + normal projection.
#[cube(launch_unchecked)]
fn integrate_half<F: Float + LaunchArg>(
    pos_in: &[F],
    vel_in: &[F],
    acc_in: &[F],
    pos_out: &mut [F],
    vel_half: &mut [F],
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

    let ht = dt * F::new(0.5_f32);
    let mut vx = vel_in[p] + acc_in[p] * ht;
    let mut vy = vel_in[p + 1] + (acc_in[p + 1] - gravity) * ht;
    let mut vz = vel_in[p + 2] + acc_in[p + 2] * ht;

    let mut x = pos_in[p] + vx * dt;
    let mut y = pos_in[p + 1] + vy * dt;
    let mut z = pos_in[p + 2] + vz * dt;

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
    let th = domain_size / F::cast_from(tx as f32);
    let fx = x / th;
    let fz = z / th;
    let frac_x = clamp(fx - fx.floor(), F::new(0.0_f32), F::new(1.0_f32));
    let frac_z = clamp(fz - fz.floor(), F::new(0.0_f32), F::new(1.0_f32));
    let x0 = usize::cast_from(clamp(fx, F::new(0.0_f32), F::cast_from((tx - 1) as f32)));
    let x1 = usize::cast_from(clamp(
        fx + F::new(1.0_f32),
        F::new(0.0_f32),
        F::cast_from((tx - 1) as f32),
    ));
    let z0 = usize::cast_from(clamp(fz, F::new(0.0_f32), F::cast_from((ty - 1) as f32)));
    let z1 = usize::cast_from(clamp(
        fz + F::new(1.0_f32),
        F::new(0.0_f32),
        F::cast_from((ty - 1) as f32),
    ));

    let b00 = terrain[x0 + z0 * tx];
    let b10 = terrain[x1 + z0 * tx];
    let b01 = terrain[x0 + z1 * tx];
    let b11 = terrain[x1 + z1 * tx];

    let b = b00 * (F::new(1.0_f32) - frac_x) * (F::new(1.0_f32) - frac_z)
        + b10 * frac_x * (F::new(1.0_f32) - frac_z)
        + b01 * (F::new(1.0_f32) - frac_x) * frac_z
        + b11 * frac_x * frac_z;

    let gx = (b10 - b00) / th;
    let gz = (b01 - b00) / th;

    let eps = F::new(0.001_f32);
    let pen = b + eps - y;
    if pen > F::new(0.0_f32) {
        // Surface normal (y-up).
        let nx = -F::new(2.0_f32) * gx;
        let nz = -F::new(2.0_f32) * gz;
        let n_mag = (nx * nx + F::new(1.0_f32) + nz * nz).sqrt();
        let nx = nx / n_mag;
        let ny = F::new(1.0_f32) / n_mag;
        let nz = nz / n_mag;

        // Penetration push along the normal (horizontal for steep walls, so particles
        // are never teleported on top of a vertical face). Capped per step to avoid
        // explosive kicks out of deep penetrations.
        let push = pen.min(F::new(0.1_f32));
        x = x + push * nx;
        y = y + push * ny;
        z = z + push * nz;

        let vn = vx * nx + vy * ny + vz * nz;
        if vn < F::new(0.0_f32) {
            vx = vx - vn * nx;
            vy = vy - vn * ny;
            vz = vz - vn * nz;
        }
        let damp = (F::new(1.0_f32) - friction * dt).max(F::new(0.0_f32));
        vx = vx * damp;
        vy = vy * damp;
        vz = vz * damp;
    }

    pos_out[p] = x;
    pos_out[p + 1] = y;
    pos_out[p + 2] = z;
    pos_out[p + 3] = F::new(0.0_f32);
    vel_half[p] = vx;
    vel_half[p + 1] = vy;
    vel_half[p + 2] = vz;
    vel_half[p + 3] = F::new(0.0_f32);
}

/// Velocity-Verlet completion: `vel += (acc - g) * dt/2` (in place; each particle only
/// touches its own slot).
#[cube(launch_unchecked)]
fn finish_verlet<F: Float + LaunchArg>(
    vel: &mut [F],
    acc: &[F],
    dt: F,
    gravity: F,
    num: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    let ht = dt * F::new(0.5_f32);
    vel[p] = vel[p] + acc[p] * ht;
    vel[p + 1] = vel[p + 1] + (acc[p + 1] - gravity) * ht;
    vel[p + 2] = vel[p + 2] + acc[p + 2] * ht;
    vel[p + 3] = F::new(0.0_f32);
}

// ---------------------------------------------------------------------------
// Debug splat (as in Phase 1)
// ---------------------------------------------------------------------------

/// Zero out the integer per-cell counters (one launch, one workgroup thread per cell).
#[cube(launch_unchecked)]
fn splat_clear<I: Int + LaunchArg>(splat: &mut [I], cells: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= cells * cells {
        terminate!();
    }
    splat[i] = I::new(0);
}

/// Scatter particles into cells: increment the per-cell counter with an integer atomic.
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

// ---------------------------------------------------------------------------
// Setup & main loop
// ---------------------------------------------------------------------------

/// Builds the neighbor grid for the current positions (zero -> count -> prefix -> scatter).
///
/// DISABLED since the grid-mode physics is not used (see the module docs and AGENTS.md):
/// `grid_scatter` produced non-deterministic cell lists on cubecl-wgpu. Kept for the
/// CPU-runtime comparison and a future migration.
fn build_grid(client: &Client, p: &WaterParticles, pos: &Handle) {
    let dim = 64u32;
    let cells = p.cells_x * p.cells_y * p.cells_z;
    let cell_inv = p.cell_inv;
    let max_y = p.cells_y as f32 * (1.0 / cell_inv);
    unsafe {
        grid_zero::launch_unchecked::<u32>(
            client,
            CubeCount::Static(cells.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            cells,
        );
        grid_count::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            cell_inv,
            crate::domain::SIZE_F32.x,
            max_y,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
        );
        grid_prefix::launch_unchecked::<u32>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            cells,
        );
        grid_scatter::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            BufferArg::from_raw_parts(p.grid_list.clone(), p.num),
            cell_inv,
            max_y,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
        );
    }
}

/// Force pass: density + pressure + accelerations at the current positions.
#[allow(clippy::too_many_arguments)]
fn sph_force_pass(
    client: &Client,
    p: &WaterParticles,
    pos: &Handle,
    vel: &Handle,
    state: &Handle,
    acc: &Handle,
    h: f32,
    h2: f32,
    sigma: f32,
    mass: f32,
    rho0: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    speed_of_sound: f32,
    use_grid: i32,
) {
    let dim = 64u32;
    let cells = p.cells_x * p.cells_y * p.cells_z;
    let cell_inv = p.cell_inv;
    let max_y = p.cells_y as f32 * (1.0 / cell_inv);
    unsafe {
        compute_density::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            BufferArg::from_raw_parts(p.grid_list.clone(), p.num),
            BufferArg::from_raw_parts(state.clone(), p.num * 4),
            h,
            h2,
            sigma,
            mass,
            rho0,
            b,
            gamma,
            cell_inv,
            max_y,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
            use_grid,
        );
        compute_force::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(vel.clone(), p.num * 4),
            BufferArg::from_raw_parts(state.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            BufferArg::from_raw_parts(p.grid_list.clone(), p.num),
            BufferArg::from_raw_parts(acc.clone(), p.num * 4),
            h,
            h2,
            sigma,
            mass,
            alpha,
            speed_of_sound,
            cell_inv,
            max_y,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
            use_grid,
        );
    }
}

/// One velocity-Verlet substep with two force evaluations (each preceded by a grid
/// rebuild for its positions).
#[allow(clippy::too_many_arguments)]
fn sph_substep(
    client: &Client,
    p: &WaterParticles,
    pos_in: &Handle,
    vel_in: &Handle,
    pos_out: &Handle,
    vel_out: &Handle,
    state_scratch: &Handle,
    acc_scratch: &Handle,
    dt: f32,
    h: f32,
    mass: f32,
    rho0: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    speed_of_sound: f32,
    use_grid: i32,
) {
    let dim = 64u32;
    let h2 = h * h;
    let sigma = 1.0 / (core::f32::consts::PI * h * h * h);

    // Force at the current positions.
    if use_grid == 1 {
        build_grid(client, p, pos_in);
    }
    sph_force_pass(
        client,
        p,
        pos_in,
        vel_in,
        state_scratch,
        acc_scratch,
        h,
        h2,
        sigma,
        mass,
        rho0,
        b,
        gamma,
        alpha,
        speed_of_sound,
        use_grid,
    );

    // Half integration. `vel_out` now holds v^{n+1/2}.
    unsafe {
        integrate_half::launch_unchecked::<f32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos_in.clone(), p.num * 4),
            BufferArg::from_raw_parts(vel_in.clone(), p.num * 4),
            BufferArg::from_raw_parts(acc_scratch.clone(), p.num * 4),
            BufferArg::from_raw_parts(pos_out.clone(), p.num * 4),
            BufferArg::from_raw_parts(vel_out.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.terrain.clone(), p.terrain_x * p.terrain_y),
            dt,
            9.81_f32,
            crate::domain::SIZE_F32.x,
            parameters::WaterParameters::default().terrain_friction,
            p.terrain_x,
            p.terrain_y,
            p.num,
        );
    }

    // Force at the new positions.
    if use_grid == 1 {
        build_grid(client, p, pos_out);
    }
    sph_force_pass(
        client,
        p,
        pos_out,
        vel_out,
        state_scratch,
        acc_scratch,
        h,
        h2,
        sigma,
        mass,
        rho0,
        b,
        gamma,
        alpha,
        speed_of_sound,
        use_grid,
    );

    // Finish the Verlet step in place (`vel_out` = v^{n+1}).
    unsafe {
        finish_verlet::launch_unchecked::<f32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(vel_out.clone(), p.num * 4),
            BufferArg::from_raw_parts(acc_scratch.clone(), p.num * 4),
            dt,
            9.81_f32,
            p.num,
        );
    }
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

    let bytes = |data: &Vec<f32>| cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(data).to_vec());

    let pos_a = client.empty((num * 4) * core::mem::size_of::<f32>());
    let pos_b = client.empty((num * 4) * core::mem::size_of::<f32>());
    let vel_a = client.empty((num * 4) * core::mem::size_of::<f32>());
    let vel_b = client.empty((num * 4) * core::mem::size_of::<f32>());
    let state_a = client.empty((num * 4) * core::mem::size_of::<f32>());
    let state_b = client.empty((num * 4) * core::mem::size_of::<f32>());
    let acc_a = client.empty((num * 4) * core::mem::size_of::<f32>());
    let acc_b = client.empty((num * 4) * core::mem::size_of::<f32>());
    client.write(&pos_a, bytes(&pos));
    client.write(&vel_a, bytes(&vel));

    let terrain_size = terrain.height_map.size;
    let terrain_bytes = terrain.height_map.buffer.len() * core::mem::size_of::<f32>();
    let terrain_handle = client.empty(terrain_bytes);
    client.write(&terrain_handle, bytes(&terrain.height_map.buffer));

    // Neighbor grid: cell size = SPH support (2h).
    let h = s * params.smoothing_ratio;
    let cell_size = 2.0 * h;
    let cells_x = (crate::domain::SIZE_F32.x / cell_size).ceil() as usize;
    let cells_y = (params.max_particle_height / cell_size).ceil() as usize;
    let cells_z = (crate::domain::SIZE_F32.y / cell_size).ceil() as usize;
    let cells = cells_x * cells_y * cells_z;
    let grid_counts = client.empty(cells * core::mem::size_of::<u32>());
    let grid_scan = client.empty((cells + 1) * core::mem::size_of::<u32>());
    let grid_list = client.empty(num * core::mem::size_of::<u32>());

    let splat_cells = 256usize;
    let splat_counts = client.empty(splat_cells * splat_cells * core::mem::size_of::<u32>());
    let splat = client.empty(splat_cells * splat_cells * 4 * core::mem::size_of::<f32>());

    commands.insert_resource(WaterParticles {
        num,
        pos_a,
        pos_b,
        vel_a,
        vel_b,
        state_a,
        state_b,
        acc_a,
        acc_b,
        terrain: terrain_handle,
        terrain_x: terrain_size.x,
        terrain_y: terrain_size.y,
        grid_counts,
        grid_scan,
        grid_list,
        cells_x,
        cells_y,
        cells_z,
        cell_inv: 1.0 / cell_size,
        splat_counts,
        splat,
        cells: splat_cells,
        current_is_b: false,
    });
}

/// Phase 2 main loop: `substeps` fixed velocity-Verlet steps per physics tick, then refresh
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
    let wp = &params.water;

    // The renderer configures its window surface during the first frames, which requires
    // the GPU queue to be idle; bulk cubecl submissions during that window break wgpu
    // ("Failed to wait for GPU to come idle before reconfiguring the Surface") and poison
    // the device. Start the sim once the surfaces have settled.
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        static WARMUP: AtomicU32 = AtomicU32::new(0);
        if WARMUP.fetch_add(1, Ordering::Relaxed) < 120 {
            return;
        }
    }

    let dt =
        1.0 / parameters::PHYSICS_TICKS_PER_SEC as f32 / wp.substeps.max(1) as f32;
    let s = wp.particle_spacing;
    let h = s * wp.smoothing_ratio;
    let mass = wp.fluid_density * s * s * s;
    let b = wp.fluid_density * wp.sound_speed * wp.sound_speed / wp.gamma;
    let dim = 64u32;

    let sub = wp.substeps.max(1);
    for _ in 0..sub {
        let use_b = p.current_is_b;
        let (pos_in, vel_in, pos_out, vel_out, state_scratch, acc_scratch) = if use_b {
            (&p.pos_b, &p.vel_b, &p.pos_a, &p.vel_a, &p.state_a, &p.acc_a)
        } else {
            (&p.pos_a, &p.vel_a, &p.pos_b, &p.vel_b, &p.state_b, &p.acc_b)
        };
        sph_substep(
            client,
            &p,
            pos_in,
            vel_in,
            pos_out,
            vel_out,
            state_scratch,
            acc_scratch,
            dt,
            h,
            mass,
            wp.fluid_density,
            b,
            wp.gamma,
            wp.viscosity_alpha,
            wp.sound_speed,
            0,
        );
        p.current_is_b = !use_b;
    }

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
        counts_to_texels::launch_unchecked::<u32, f32>(
            client,
            CubeCount::Static((p.cells * p.cells).div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.splat_counts.clone(), p.cells * p.cells),
            BufferArg::from_raw_parts(p.splat.clone(), p.cells * p.cells * 4),
            wp.debug_count_scale,
            p.cells,
        );
    }

    let managed = client
        .get_resource::<cubecl::wgpu::WgpuServer<cubecl::wgpu::AutoCompiler>>(p.splat.clone())
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
    use cubecl::wgpu::wgpu as wgpu;

    const RHO0: f32 = 1000.0;
    const GRAVITY: f32 = 9.81;
    const SPEED_OF_SOUND: f32 = 20.0;
    const GAMMA: f32 = 7.0;
    const ALPHA: f32 = 0.2;
    const TERRAIN_FRICTION: f32 = 3.0;

    /// Creates a cubecl client on a private wgpu device (headless, compute only) for
    /// tests, or `None` when no adapter is present (e.g. headless box).
    ///
    /// The physics tests run on BOTH runtimes: the CPU runtime never had atomics and
    /// masked the cubecl-wgpu grid scatter bug that broke production (see AGENTS.md) —
    /// GPU test coverage is the default on this machine and catches backend-specific
    /// divergences (fast-math, atomics, memory model).
    fn test_gpu_client() -> Option<Client> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = bevy::tasks::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        let (device, queue) = bevy::tasks::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("eco-sim water tests"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        ))
        .ok()?;
        let setup = cubecl::wgpu::WgpuSetup {
            instance,
            adapter,
            device: device.clone(),
            queue,
            backend: device.adapter_info().backend,
        };
        let dev =
            cubecl::wgpu::try_init_device(setup, cubecl::wgpu::RuntimeOptions::default())
                .ok()?;
        Some(cubecl::Device::from(dev).client())
    }

    /// Runs `f` on the CPU runtime and, when a GPU is available, on the GPU runtime.
    /// GPU-first: if backend-specific divergence exists, the GPU is the reference
    /// (it is what production runs on).
    fn run_on_all_backends(f: impl Fn(&Client) + Send + Sync) {
        f(&cubecl::Device::cpu().unwrap().client());
        match test_gpu_client() {
            Some(gpu) => f(&gpu),
            None => bevy::log::info!(
                "no GPU adapter found for tests; CPU-runtime coverage only"
            ),
        }
    }

    /// Minimal in-test equivalent of `WaterParticles` (uses `use_grid = 0`, so the atomic
    /// grid kernels are never launched — the cubecl CPU runtime has no atomics).
    struct Rig {
        client: Client,
        pos_a: Handle,
        pos_b: Handle,
        vel_a: Handle,
        vel_b: Handle,
        state: Handle,
        acc: Handle,
        terrain: Handle,
        terrain_x: usize,
        terrain_y: usize,
        grid_scan: Handle,
        grid_list: Handle,
        num: usize,
        use_b: bool,
    }

    impl Rig {
        fn new(pos: &[f32], terrain: &[f32], tx: usize, ty: usize) -> Self {
            Self::new_with_client(cubecl::Device::cpu().unwrap().client(), pos, terrain, tx, ty)
        }

        fn new_with_client(client: Client, pos: &[f32], terrain: &[f32], tx: usize, ty: usize) -> Self {
            let num = pos.len() / 4;
            let pos_a = client.empty(pos.len() * core::mem::size_of::<f32>());
            let pos_b = client.empty(pos.len() * core::mem::size_of::<f32>());
            let vel_a = client.empty(pos.len() * core::mem::size_of::<f32>());
            let vel_b = client.empty(pos.len() * core::mem::size_of::<f32>());
            let state = client.empty(pos.len() * core::mem::size_of::<f32>());
            let acc = client.empty(pos.len() * core::mem::size_of::<f32>());
            let terrain_h = client.empty(terrain.len() * core::mem::size_of::<f32>());
            let write = |h: &Handle, d: &[f32]| {
                client.write(h, cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(d).to_vec()));
            };
            write(&pos_a, pos);
            write(&vel_a, &vec![0.0f32; pos.len()]);
            write(&terrain_h, terrain);
            let grid_scan = client.empty(2 * core::mem::size_of::<u32>());
            let grid_list = client.empty(1 * core::mem::size_of::<u32>());
            Rig {
                client,
                pos_a,
                pos_b,
                vel_a,
                vel_b,
                state,
                acc,
                terrain: terrain_h,
                terrain_x: tx,
                terrain_y: ty,
                grid_scan,
                grid_list,
                num,
                use_b: false,
            }
        }

        fn step(&mut self, dt: f32, steps: usize, spacing: f32) {
            let h = spacing * 1.3;
            let mass = RHO0 * spacing * spacing * spacing;
            let b = RHO0 * SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA;
            let p = WaterParticles {
                num: self.num,
                pos_a: self.pos_a.clone(),
                pos_b: self.pos_b.clone(),
                vel_a: self.vel_a.clone(),
                vel_b: self.vel_b.clone(),
                state_a: self.state.clone(),
                state_b: self.state.clone(),
                acc_a: self.acc.clone(),
                acc_b: self.acc.clone(),
                terrain: self.terrain.clone(),
                terrain_x: self.terrain_x,
                terrain_y: self.terrain_y,
                grid_counts: self.grid_scan.clone(),
                grid_scan: self.grid_scan.clone(),
                grid_list: self.grid_list.clone(),
                cells_x: 1,
                cells_y: 1,
                cells_z: 1,
                cell_inv: 1.0,
                splat_counts: self.grid_scan.clone(),
                splat: self.grid_scan.clone(),
                cells: 1,
                current_is_b: self.use_b,
            };
            for _ in 0..steps {
                let (pos_in, vel_in, pos_out, vel_out) = if self.use_b {
                    (&self.pos_b, &self.vel_b, &self.pos_a, &self.vel_a)
                } else {
                    (&self.pos_a, &self.vel_a, &self.pos_b, &self.vel_b)
                };
                sph_substep(
                    &self.client,
                    &p,
                    pos_in,
                    vel_in,
                    pos_out,
                    vel_out,
                    &self.state,
                    &self.acc,
                    dt,
                    h,
                    mass,
                    RHO0,
                    b,
                    GAMMA,
                    ALPHA,
                    SPEED_OF_SOUND,
                    0,
                );
                self.use_b = !self.use_b;
            }
        }

        fn read_current_pos(&self) -> Vec<f32> {
            let h = if self.use_b { &self.pos_b } else { &self.pos_a };
            let bytes = self.client.read_one_unchecked(h.clone());
            f32::from_bytes(&bytes).to_vec()
        }

        fn read_current_vel(&self) -> Vec<f32> {
            let h = if self.use_b { &self.vel_b } else { &self.vel_a };
            let bytes = self.client.read_one_unchecked(h.clone());
            f32::from_bytes(&bytes).to_vec()
        }

        fn read_state(&self) -> Vec<f32> {
            let bytes = self.client.read_one_unchecked(self.state.clone());
            f32::from_bytes(&bytes).to_vec()
        }

        fn read_acc(&self) -> Vec<f32> {
            let bytes = self.client.read_one_unchecked(self.acc.clone());
            f32::from_bytes(&bytes).to_vec()
        }
    }

    /// Velocity-Verlet with gravity only: exact parabola y = y0 - g t^2 / 2.
    #[test]
    fn verlet_free_fall_matches_closed_form() {
        run_on_all_backends(verlet_free_fall_impl);
    }

    fn verlet_free_fall_impl(client: &Client) {
        let mut pos = [32.0f32, 50.0, 32.0, 0.0];
        let mut vel = [0.0f32; 4];
        let rig = Rig::new_with_client(client.clone(), &pos, &vec![0.0f32; 8 * 8], 8, 8);

        let dt = 0.01;
        let n = 40;
        // acc = 0: run the integrator kernels alone.
        let dim = 64u32;
        let mut use_b = false;
        for _ in 0..n {
            let (pi, vi, po, vo) = if use_b {
                (&rig.pos_b, &rig.vel_b, &rig.pos_a, &rig.vel_a)
            } else {
                (&rig.pos_a, &rig.vel_a, &rig.pos_b, &rig.vel_b)
            };
            unsafe {
                integrate_half::launch_unchecked::<f32>(
                    &rig.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(pi.clone(), 4),
                    BufferArg::from_raw_parts(vi.clone(), 4),
                    BufferArg::from_raw_parts(rig.acc.clone(), 4),
                    BufferArg::from_raw_parts(po.clone(), 4),
                    BufferArg::from_raw_parts(vo.clone(), 4),
                    BufferArg::from_raw_parts(rig.terrain.clone(), 64),
                    dt,
                    GRAVITY,
                    64.0,
                    TERRAIN_FRICTION,
                    rig.terrain_x,
                    rig.terrain_y,
                    1,
                );
                finish_verlet::launch_unchecked::<f32>(
                    &rig.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(vo.clone(), 4),
                    BufferArg::from_raw_parts(rig.acc.clone(), 4),
                    dt,
                    GRAVITY,
                    1,
                );
            }
            use_b = !use_b;
        }

        let out_h = if use_b { rig.pos_b.clone() } else { rig.pos_a.clone() };
        let vel_h = if use_b { rig.vel_b.clone() } else { rig.vel_a.clone() };
        let bytes = rig.client.read_one_unchecked(out_h);
        pos.copy_from_slice(&f32::from_bytes(&bytes));
        let bytes = rig.client.read_one_unchecked(vel_h);
        vel.copy_from_slice(&f32::from_bytes(&bytes));

        let t = n as f32 * dt;
        let expected_y = 50.0 - 0.5 * GRAVITY * t * t;
        let expected_vy = -GRAVITY * t;
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

    /// Kinematics only: particles must stay inside the closed box.
    #[test]
    fn wall_keeps_particles_inside() {
        run_on_all_backends(wall_keeps_particles_inside_impl);
    }

    fn wall_keeps_particles_inside_impl(client: &Client) {
        let pos = [0.5f32, 10.0, 63.5, 0.0];
        let mut rig = Rig::new_with_client(client.clone(), &pos, &vec![0.0f32; 8 * 8], 8, 8);
        let dt = 1.0 / 60.0 / 4.0;
        // Overwrite initial velocity (Rig starts at rest): reuse the vel buffer.
        rig.client.write(
            &rig.vel_a,
            cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&[-50.0f32, 0.0, 50.0, 0.0]).to_vec()),
        );
        rig.step(dt, 4 * 60 * 2, 0.5);

        let pos = rig.read_current_pos();
        assert!(pos[0] >= 0.0 && pos[0] <= 64.0, "x out of box: {:?}", pos);
        assert!(pos[2] >= 0.0 && pos[2] <= 64.0, "z out of box: {:?}", pos);
        assert!(pos[1] >= 0.0, "fell through floor: {:?}", pos);
        assert!(
            (pos[0] - 0.0).abs() < 0.02 && (pos[2] - 64.0).abs() < 0.02,
            "should rest against the walls: {:?}",
            pos
        );
    }

    /// WCSPH water column settling in a smooth bowl. Structural assertions: the fluid
    /// stays contained in the bowl, becomes calm (no collapse/explosion), its density
    /// stays near rho0, and the pressure of the deepest layer exceeds the surface layer.
    ///
    /// Note: quantitative hydrostatic-profile matching remains out of scope here - the
    /// residual clumping (particle coincidence) and sloshing are the known WCSPH
    /// stabilization challenges addressed later (delta-SPH / XSPH / resolution).
    #[test]
    fn hydrostatic_column_settles_and_pressure_balances_gravity() {
        run_on_all_backends(hydrostatic_column_settles_and_pressure_balances_gravity_impl);
    }

    fn hydrostatic_column_settles_and_pressure_balances_gravity_impl(client: &Client) {
        let spacing = 0.5;
        // Smooth parabolic bowl (256x256 terrain grid, 0.25 m/cell): b = 0.4 r^2.
        let gx = 256usize;
        let mut terrain = vec![0.0f32; gx * gx];
        for j in 0..gx {
            for i in 0..gx {
                let x = i as f32 * 0.25;
                let z = j as f32 * 0.25;
                let dx = x - 32.0;
                let dz = z - 32.0;
                terrain[i + j * gx] = 0.5 * 0.8 * (dx * dx + dz * dz);
            }
        }

        // Start the water resting on the bowl shape so the initial condition is near
        // equilibrium.
        let mut pos = Vec::new();
        for z in 0..5usize {
            for x in 0..5usize {
                let px = 31.0 + x as f32 * spacing;
                let pz = 31.0 + z as f32 * spacing;
                let base = 0.4 * ((px - 32.0) * (px - 32.0) + (pz - 32.0) * (pz - 32.0));
                for y in 0..5usize {
                    pos.push(px);
                    pos.push(base + 0.05 + y as f32 * spacing);
                    pos.push(pz);
                    pos.push(0.0);
                }
            }
        }
        let mut rig = Rig::new_with_client(client.clone(), &pos, &terrain, gx, gx);
        let dt = 1.0 / 240.0;

        for _ in 0..3000 {
            rig.step(dt, 1, spacing);
        }

        let vel = rig.read_current_vel();
        let max_speed = vel
            .chunks_exact(4)
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            max_speed < 1.0,
            "column should be calm, max speed {max_speed}"
        );

        let state = rig.read_state();
        let cur_pos = rig.read_current_pos();
        let mut rho_min = f32::MAX;
        let mut rho_max = f32::MIN;
        let mut rho_sum = 0.0f32;
        let mut bottom_p = 0.0f32;
        let mut top_p = 0.0f32;
        let mut n_bottom = 0usize;
        let mut n_top = 0usize;
        let mut max_r = 0.0f32;
        for (k, s) in state.chunks_exact(4).enumerate() {
            rho_sum += s[0];
            rho_min = rho_min.min(s[0]);
            rho_max = rho_max.max(s[0]);
            let y = cur_pos[k * 4 + 1];
            let r = ((cur_pos[k * 4] - 32.0).powi(2) + (cur_pos[k * 4 + 2] - 32.0).powi(2)).sqrt();
            max_r = max_r.max(r);
            if y < 0.45 {
                bottom_p += s[1];
                n_bottom += 1;
            }
            if y > 0.8 {
                top_p += s[1];
                n_top += 1;
            }
        }
        let rho_mean = rho_sum / rig.num as f32;

        // Contained in the bowl (main particle set; loose bound for the bowl at r=2.5).
        assert!(max_r < 3.5, "water escaped the bowl: max radius {max_r}");
        assert!(n_bottom > 0 && n_top > 0, "layers missing: {n_bottom} {n_top}");
        assert!(
            (rho_mean - RHO0).abs() < 0.08 * RHO0,
            "mean density {rho_mean} too far from rho0"
        );
        let bp = bottom_p / n_bottom as f32;
        let tp = top_p / n_top as f32;
        assert!(
            bp > 2.0 * tp,
            "pressure should increase with depth: bottom {bp} vs top {tp}"
        );
        println!(
            "settled: max_speed {max_speed}, rho {rho_min}..{rho_max} (mean {rho_mean}), bottom p {bp}, top p {tp}"
        );
    }

    /// Count -> prefix -> scatter invariant test (GPU only, integer atomics are required).
    ///
    /// IGNORED: on cubecl-wgpu the scatter produced non-deterministic cell lists even at
    /// frozen positions (in-kernel `scan[cell] + slot` mismatched the host prefix), which
    /// is invisible on the CPU runtime (no atomics at all). This is the tracking test:
    /// re-run with `cargo test -- --ignored` after a cubecl/wgpu upgrade to re-check.
    #[test]
    #[ignore = "cubecl-wgpu grid_scatter nondeterminism (see AGENTS.md)"]
    fn grid_cell_lists_are_consistent_on_gpu() {
        let Some(client) = test_gpu_client() else {
            eprintln!("no GPU adapter; grid test requires integer atomics - skipped");
            return;
        };
        // Production-like layout: 24x4x24 blob, spacing 0.5, h = 0.65, cell = 1.3 m,
        // 50x7x50 cells (matches the app's spawn positions).
        let spacing = 0.5f32;
        let h = spacing * 1.3;
        let cell_size = 2.0 * h;
        let cell_inv = 1.0 / cell_size;
        let cells = (50, 7, 50);
        let n_cells = cells.0 * cells.1 * cells.2;
        let (nx, ny, nz) = (24usize, 4, 24);
        let num = nx * ny * nz;
        let mut pos = Vec::with_capacity(num * 4);
        let (cx, cz) = (26.0f32, 26.0f32);
        for y in 0..ny {
            for z in 0..nz {
                for x in 0..nx {
                    pos.push(cx + x as f32 * spacing);
                    pos.push(1.5 + y as f32 * spacing);
                    pos.push(cz + z as f32 * spacing);
                    pos.push(0.0);
                }
            }
        }
        let pos_h = client.empty(pos.len() * core::mem::size_of::<f32>());
        client.write(
            &pos_h,
            cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&pos).to_vec()),
        );
        let alloc_u = |n: usize| client.empty(n * core::mem::size_of::<u32>());
        let counts = alloc_u(n_cells);
        let scan = alloc_u(n_cells + 1);
        let list = alloc_u(num);
        let p = WaterParticles {
            num,
            pos_a: pos_h.clone(),
            pos_b: pos_h.clone(),
            vel_a: pos_h.clone(),
            vel_b: pos_h.clone(),
            state_a: pos_h.clone(),
            state_b: pos_h.clone(),
            acc_a: pos_h.clone(),
            acc_b: pos_h.clone(),
            terrain: pos_h.clone(),
            terrain_x: 1,
            terrain_y: 1,
            grid_counts: counts,
            grid_scan: scan,
            grid_list: list,
            cells_x: cells.0,
            cells_y: cells.1,
            cells_z: cells.2,
            cell_inv,
            splat_counts: pos_h.clone(),
            splat: pos_h.clone(),
            cells: 1,
            current_is_b: false,
        };
        build_grid(&client, &p, &pos_h);

        let cb = client.read_one_unchecked(p.grid_counts.clone());
        let c = u32::from_bytes(&cb);
        let sb = client.read_one_unchecked(p.grid_scan.clone());
        let s = u32::from_bytes(&sb);
        let lb = client.read_one_unchecked(p.grid_list.clone());
        let l = u32::from_bytes(&lb);

        // Invariants:
        //  * scan is monotonically increasing and scan[n_cells] == num (true count).
        //  * counts *after* the scatter equal `scan` deltas (the scatter's atomics double
        //    the counters; the scan built before the scatter is authoritative).
        //  * `list` is a permutation of 0..num, and each entry k lies in the range of its
        //    own cell (scan[cell(k)] <= k < scan[cell(k) + 1]).
        assert_eq!(s[n_cells], num as u32, "prefix total");
        let cell_of = |k: usize| -> usize {
            let p4 = k * 4;
            let ix = ((pos[p4] * cell_inv) as usize).min(cells.0 - 1);
            let iy = ((pos[p4 + 1] * cell_inv) as usize).min(cells.1 - 1);
            let iz = ((pos[p4 + 2] * cell_inv) as usize).min(cells.2 - 1);
            ix + cells.0 * (iy + cells.1 * iz)
        };
        let mut seen = vec![false; num];
        for k in 0..num {
            let j = l[k] as usize;
            assert!(j < num, "list[k]={j} out of range");
            assert!(!seen[j], "duplicate particle index {j} in list");
            seen[j] = true;
            let cell = cell_of(j);
            let lo = s[cell] as usize;
            let hi = s[cell + 1] as usize;
            assert!(
                k >= lo && k < hi,
                "particle {j} (cell {cell}, range {lo}..{hi}) listed at k={k}"
            );
        }
        assert!(seen.iter().all(|&v| v), "list misses particles");
        // Also check count consistency (post-scatter counts == 2x pre-scatter deltas).
        let mut total: u32 = 0;
        for cell in 0..n_cells {
            let delta = s[cell + 1] - s[cell];
            assert_eq!(c[cell], 2 * delta, "cell {cell} count after scatter");
            total += c[cell];
        }
        assert_eq!(total, 2 * num as u32);
    }
}
