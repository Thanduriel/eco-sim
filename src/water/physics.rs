//! Numerical core of the water simulation: weakly-compressible SPH hydrodynamics
//! (kernels, uniform-grid neighbor search, `run_water`). Rendering lives in
//! `render.rs`, the analytic test battery in `tests.rs`, the Bevy wiring in `mod.rs`.
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
//! Neighbor search: the *physics* kernels take a `NeighborSearch` switch. Production
//! runs `UniformGrid` (cell lists built with integer atomics: zero -> count -> prefix ->
//! scatter); `Dense` (all-pairs) remains the CPU-runtime test path (the cubecl CPU
//! runtime has no atomics). The scatter must atomically claim slots against a *zeroed*
//! counter array — the count pass leaves the counters at the per-cell totals, so the
//! scatter re-zeros them first (see `build_grid`). `grid_cell_lists_are_consistent_on_gpu`
//! pins the count/prefix/scatter invariant on the GPU.

use bevy::prelude::*;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::gpu;
use crate::parameters;
use crate::terrain;

/// Boundary model of the domain. Kernels receive the `u32` discriminant (cubecl kernel
/// arguments must be plain values, so the enum is the host-side spelling of those codes).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u32)]
#[allow(dead_code)] // FullPeriodic/PeriodicXZFloor are constructed by the analytic tests
pub enum BoundaryMode {
    /// Closed box: four walls + terrain, no wraparound (production).
    #[default]
    Closed = 0,
    /// Fully periodic: all three axes wrap, no walls/terrain.
    FullPeriodic = 1,
    /// x/z wrap + hard floor at y = 0 (laterally infinite slab).
    PeriodicXZFloor = 2,
}

/// Neighbor search strategy (kernels receive the `u32` discriminant).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u32)]
pub enum NeighborSearch {
    /// Dense all-pairs loop. Kept for the CPU test runtime, which has no atomics.
    #[default]
    Dense = 0,
    /// Uniform grid built with integer atomics (production).
    UniformGrid = 1,
}

/// Analytic half-space density correction at solid boundaries (1/0 in the old naming;
/// the kernel used to be called `floor_ghosts` for a ghost-particle scheme that was
/// since replaced — the parameter activates the current correction, not ghosts).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u32)]
pub enum BoundaryCorrection {
    #[default]
    Off = 0,
    On = 1,
}

/// Compute workgroup size used for every launch (matches the 1-D `CubeDim`).
pub(super) const LAUNCH_DIM: u32 = 64;

/// SPH fluid + integration parameters for one force pass / substep. The kernels were
/// getting ~25 positional arguments; this groups the physics constants (derived once per
/// tick from `WaterParameters`, or per test from the Rig's calibrated values). `dt` stays
/// a separate argument (it is a timestep, not a fluid property).
#[derive(Clone, Copy)]
pub struct SphParams {
    /// Smoothing length h (m).
    pub h: f32,
    /// h².
    pub h2: f32,
    /// Cubic-spline normalization 1/(π h³).
    pub sigma: f32,
    /// Particle mass (kg).
    pub mass: f32,
    /// Rest density ρ₀ (kg/m³).
    pub rho0: f32,
    /// EoS clamp: minimum density fraction (1.0 = no tension, < 1 allows bounded
    /// tension so a free surface can hold itself).
    pub rho_min_frac: f32,
    /// Tait EoS stiffness B = ρ₀ c₀² / γ (Pa).
    pub b: f32,
    /// Tait isentropic exponent γ.
    pub gamma: f32,
    /// Monaghan artificial viscosity coefficient α.
    pub alpha: f32,
    /// Reference sound speed c₀ (m/s).
    pub speed_of_sound: f32,
    /// Neighbor search strategy.
    pub neighbor: NeighborSearch,
    /// Domain box side length (m).
    pub domain: f32,
    /// External gravity (m/s², y-down in the integrator).
    pub gravity: f32,
    /// Tangential damping on terrain contact.
    pub friction: f32,
    /// Boundary model of the domain.
    pub boundary: BoundaryMode,
    /// Analytic half-space density correction at solid boundaries.
    pub correction: BoundaryCorrection,
    /// XSPH velocity-smoothing coefficient (0 = off).
    pub xsph: f32,
    /// delta-SPH density diffusion coefficient δ (0 = off).
    pub diffusion: f32,
}

impl SphParams {
    /// Production pipeline: derives the SPH constants from the F4 tunables. The neighbor
    /// search, boundary model and correction are the production choices (the analytic
    /// tests configure these explicitly).
    pub fn from_water_parameters(
        wp: &parameters::WaterParameters,
        spacing: f32,
        domain: f32,
    ) -> Self {
        let h = spacing * wp.smoothing_ratio;
        Self {
            h,
            h2: h * h,
            sigma: 1.0 / (core::f32::consts::PI * h * h * h),
            mass: wp.fluid_density * spacing * spacing * spacing,
            rho0: wp.fluid_density,
            rho_min_frac: 1.0,
            b: wp.fluid_density * wp.sound_speed * wp.sound_speed / wp.gamma,
            gamma: wp.gamma,
            alpha: wp.viscosity_alpha,
            speed_of_sound: wp.sound_speed,
            neighbor: NeighborSearch::UniformGrid,
            domain,
            gravity: 9.81,
            friction: wp.terrain_friction,
            boundary: BoundaryMode::Closed,
            correction: BoundaryCorrection::On,
            xsph: wp.xsph,
            diffusion: wp.diffusion,
        }
    }
}

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
    /// Previous pass densities (deterministic input for the delta-SPH diffusion term).
    pub rho_prev: Handle,
    /// Per-particle acceleration `[ax, ay, az, 0]` (pressure + viscosity only; gravity is
    /// applied by the integrator).
    pub acc_a: Handle,
    pub acc_b: Handle,
    /// XSPH velocity increment `[avx, avy, avz, 0]` written by `compute_force` each
    /// force pass (the sum is folded into the force's neighbor sweep at zero extra
    /// sweep cost) and applied by `integrate_half` (`v += xsph * delta`).
    pub xsph_delta: Handle,
    /// Per-particle cached pressure data `[p/rho², rho, 0, 0]` (written by every
    /// `compute_density`, read by `compute_force` so the force pass needs no `state`
    /// read and no per-pair `1/rho_j²` division). Bit-identical to the inline form.
    pub press_cache: Handle,
    /// Terrain height map (grid points at `i * h`, Vertex convention), uploaded once.
    pub terrain: Handle,
    pub terrain_x: usize,
    pub terrain_y: usize,
    /// Uniform neighbor grid: per-cell counts (`cells` u32), exclusive prefix (`cells+1`),
    /// and the compacted particle index list (`num` u32). The prefix is built by a
    /// hierarchical parallel scan: block totals (`nblocks` u32 = `cells/64`) -> block
    /// offsets (`nblocks+1` u32) -> inner scan. The offsets MUST be a separate buffer:
    /// sharing the scan buffer with the inner-scan writes races (inner threads rewrite
    /// the block offsets, corrupting the reads of later blocks). The old single-threaded
    /// prefix was ~6 ms/build.
    pub grid_counts: Handle,
    pub grid_scan: Handle,
    pub grid_list: Handle,
    pub grid_totals: Handle,
    pub grid_offsets: Handle,
    pub cells_x: usize,
    pub cells_y: usize,
    pub cells_z: usize,
    pub cell_inv: f32,
    /// Phase 3 surface: per-cell max particle height, encoded as `mm * 1000` in `u32`
    /// (`fetch_max`, integer atomics) ...
    pub surface_max: Handle,
    /// ... converted into `Rgba32Float` texels `[surface_height, depth, 0, 0]` where
    /// depth = surface - bilinear terrain height at the cell center.
    pub surface: Handle,
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


/// Copies the latest density into the `rho_prev` scratch (deterministic delta-SPH input).
#[cube(launch_unchecked)]
pub(super) fn copy_rho_prev<F: Float + LaunchArg>(state: &[F], rho_prev: &mut [F], num: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    rho_prev[p] = state[p];
}

/// Zero the per-cell counters.
#[cube(launch_unchecked)]
pub(super) fn grid_zero<I: Int + LaunchArg>(counts: &mut [I], cells: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i < cells {
        counts[i] = I::new(0);
    }
}

/// Per-particle cell counter (integer atomics). Cell size equals the SPH support
/// radius (2h), so a 3x3x3 cell window fully covers the kernel support.
#[cube(launch_unchecked)]
pub(super) fn grid_count<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    counts: &[Atomic<I>],
    cell_inv: F,
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

/// Block totals for the hierarchical prefix: thread `b` sums the 64 counts of block `b`
/// (≤64 sequential reads, `nblocks` threads in parallel) into `totals[b]`.
#[cube(launch_unchecked)]
pub(super) fn grid_block_totals<I: Int + LaunchArg>(
    counts: &[I],
    totals: &mut [I],
    cells: usize,
    treads: usize,
    nblocks: usize,
) {
    let b = ABSOLUTE_POS_X as usize;
    if b >= nblocks {
        terminate!();
    }
    let start = b * treads;
    let mut s = I::new(0);
    let mut k = 0usize;
    while k < treads {
        let idx = start + k;
        if idx < cells {
            s = s + counts[idx];
        }
        k = k + 1;
    }
    totals[b] = s;
}

/// Exclusive prefix over the block totals into `scan[0..=nblocks]` (one thread; ~274
/// iterations, µs). The previous single-threaded full-grid prefix (~17,500 iterations of
/// a serial dependency chain) was the real performance bottleneck (~6-9 ms per build).
#[cube(launch_unchecked)]
pub(super) fn grid_block_prefix<I: Int + LaunchArg>(
    totals: &[I],
    scan: &mut [I],
    nblocks: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i != 0 {
        terminate!();
    }
    scan[0] = I::new(0);
    for b in 0..nblocks {
        scan[b + 1] = scan[b] + totals[b];
    }
}

/// Final per-cell scan: `scan[c+1] = scan[block(c)] + Σ counts[block_start..=c]`, one
/// thread per cell, ≤64 reads per thread. Runs after [`grid_block_prefix`], which already
/// wrote `scan[0..=nblocks]` (the per-block exclusive offsets).
#[cube(launch_unchecked)]
pub(super) fn grid_inner_scan<I: Int + LaunchArg>(
    counts: &[I],
    offsets: &[I],
    scan: &mut [I],
    cells: usize,
    treads: usize,
) {
    let c = ABSOLUTE_POS_X as usize;
    if c >= cells {
        terminate!();
    }
    let b = c / treads;
    let base = b * treads;
    let mut s = offsets[b];
    let mut k = base;
    while k <= c {
        s = s + counts[k];
        k = k + 1;
    }
    scan[c + 1] = s;
    if c == 0 {
        scan[0] = I::new(0);
    }
}

/// Scatter particle indices into per-cell ranges (cell size 2h; see [`grid_count`]).
#[cube(launch_unchecked)]
pub(super) fn grid_scatter<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    scan: &[I],
    counts: &[Atomic<I>],
    list: &mut [I],
    cell_inv: F,
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
pub(super) fn compute_density<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    scan: &[I],
    list: &[I],
    state: &mut [F],
    rho_prev: &[F],
    // Per-particle cached pressure data [p/rho², rho, 0, 0], consumed by the force
    // pass so it needs no `state` read and no per-pair 1/rho_j² division (the values
    // are the same expressions the force pass used to compute inline).
    cache: &mut [F],
    terrain: &[F],
    tx: usize,
    ty: usize,
    h: F,
    h2: F,
    sigma: F,
    mass: F,
    diffusion: F,
    rho0: F,
    rho_min_frac: F,
    b: F,
    gamma: F,
    cell_inv: F,
    domain: F,
    cx: usize,
    cy: usize,
    cz: usize,
    num: usize,
    use_grid: u32,
    periodic: u32,
    floor_ghosts: u32,
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
    // delta-SPH density diffusion accumulation (uses the previous pass's densities).
    let mut diff = F::new(0.0_f32);
    let rho_prev_i = rho_prev[p];

    if use_grid == 0 {
        for j in 0..num {
            {
                let pj = j * 4;
                let mut dx = xi - pos[pj];
                let mut dy = yi - pos[pj + 1];
                let mut dz = zi - pos[pj + 2];
                if periodic == 1 {
                    let half = domain * F::new(0.5_f32);
                    if dx > half {
                        dx = dx - domain;
                    }
                    if dx < -half {
                        dx = dx + domain;
                    }
                    if dy > half {
                        dy = dy - domain;
                    }
                    if dy < -half {
                        dy = dy + domain;
                    }
                    if dz > half {
                        dz = dz - domain;
                    }
                    if dz < -half {
                        dz = dz + domain;
                    }
                }
                if periodic == 2 {
                    let half = domain * F::new(0.5_f32);
                    if dx > half {
                        dx = dx - domain;
                    }
                    if dx < -half {
                        dx = dx + domain;
                    }
                    if dz > half {
                        dz = dz - domain;
                    }
                    if dz < -half {
                        dz = dz + domain;
                    }
                }
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < support2 {
                    let q = (r2).sqrt() / h;
                    let q2 = q * q;
                    let mut w = F::new(0.0_f32);
                    let mut dw_dq = F::new(0.0_f32);
                    if q < F::new(1.0_f32) {
                        w = sigma * (F::new(1.0_f32) - F::new(1.5_f32) * q2 + F::new(0.75_f32) * q2 * q);
                        dw_dq = sigma * (F::new(-3.0_f32) * q + F::new(2.25_f32) * q2);
                    } else {
                        if q < F::new(2.0_f32) {
                            let t = F::new(2.0_f32) - q;
                            let t2 = t * t;
                            w = sigma * F::new(0.25_f32) * t2 * t;
                            dw_dq = F::new(-0.75_f32) * sigma * t2;
                        }
                    }
                    rho = rho + mass * w;
                    let rho_bar_prev = (rho_prev[pj] + rho_prev_i) * F::new(0.5_f32);
                    diff = diff
                        + (mass / rho_bar_prev)
                            * ((rho_prev[pj] - rho_prev_i) / rho_prev_i)
                            * dw_dq
                            / h;
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

        for cdz in 0..3i32 {
            for cdy in 0..3i32 {
                for cdx in 0..3i32 {
                    let nc = icx + (cdx - 1);
                    let nr = icy + (cdy - 1);
                    let nh = icz + (cdz - 1);
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

    if floor_ghosts == 1 {
        // Half-space boundary renormalization: the summation density near a solid
        // boundary is truncated (missing kernel support below the wall). c(q0) is the
        // analytic support fraction; dividing restores the continuum density WITHOUT
        // adding ghost masses (which double-count) and preserves the EoS compression
        // signal (unlike Shepard renormalization).
        let d = if tx > 1 {
            // Distance to the terrain SURFACE (local boundary model): for every terrain
            // cell within the support radius (2h; the radius scales with the grid
            // spacing, capped at 3 cells), the foot point is the bilinear surface point
            // at the particle's (x, z) clamped into that cell, and the distance is taken
            // AT THE CELL'S TANGENT PLANE (exact for the local bilinear patch at any
            // slope). The minimum over the neighborhood is the wall distance.
            //
            // The former (y - b) VERTICAL distance is exact only for a flat floor:
            // under/beside a steep face it collapses to ~0, so c(q0) -> 0.5 doubles the
            // density and the Tait EoS explodes the fluid (production max_surf 31 m).
            // The tangent-plane distance is the correct measure on cliffs, ridges and
            // any future heightfield feature; it also handles particles inside the
            // terrain (buried: returns the perpendicular burial depth, bounded).
            let th = domain / F::cast_from(tx as f32);
            let fx = xi / th;
            let fz = zi / th;
            let rad = usize::cast_from(
                (F::new(2.0_f32) * h / th).ceil().min(F::new(3.0_f32)),
            );
            let one = F::new(1.0_f32);
            let xm = usize::cast_from(clamp(fx, F::new(0.0_f32), F::cast_from((tx - 1) as f32)));
            let zm = usize::cast_from(clamp(fz, F::new(0.0_f32), F::cast_from((ty - 1) as f32)));
            let mut best = F::new(1e10_f32);
            for kz in 0..(rad + rad + 1usize) {
                for kx in 0..(rad + rad + 1usize) {
                    let xid = xm + kx;
                    let cxx = (if kx <= rad {
                        if xid >= rad {
                            xid - rad
                        } else {
                            0usize
                        }
                    } else {
                        xid - rad
                    })
                    .min(tx - 1usize);
                    let zid = zm + kz;
                    let czz = (if kz <= rad {
                        if zid >= rad {
                            zid - rad
                        } else {
                            0usize
                        }
                    } else {
                        zid - rad
                    })
                    .min(ty - 1usize);
                    let u = clamp(
                        fx - F::cast_from(cxx as f32),
                        F::new(0.0_f32),
                        F::new(1.0_f32),
                    );
                    let v = clamp(
                        fz - F::cast_from(czz as f32),
                        F::new(0.0_f32),
                        F::new(1.0_f32),
                    );
                    let xx1 = (cxx + 1usize).min(tx - 1usize);
                    let zz1 = (czz + 1usize).min(ty - 1usize);
                    let b00 = terrain[cxx + czz * tx];
                    let b10 = terrain[xx1 + czz * tx];
                    let b01 = terrain[cxx + zz1 * tx];
                    let b11 = terrain[xx1 + zz1 * tx];
                    let qy = b00 * (one - u) * (one - v)
                        + b10 * u * (one - v)
                        + b01 * (one - u) * v
                        + b11 * u * v;
                    let dx = xi - (F::cast_from(cxx as f32) + u) * th;
                    let dy = yi - qy;
                    let dz = zi - (F::cast_from(czz as f32) + v) * th;
                    let gx = ((b10 - b00) * (one - v) + (b11 - b01) * v) / th;
                    let gz = ((b01 - b00) * (one - u) + (b11 - b10) * u) / th;
                    let nrm = (gx * gx + gz * gz + one).sqrt();
                    if cxx == xm && czz == zm {
                        // Own cell: perpendicular distance to the local tangent plane
                        // (correct on steep faces: the vertical demo (y - b) is exact
                        // only for a floor, and a neighbor plane would be chosen in a
                        // CONCAVE wall interior where the extended rim plane cuts
                        // through the fluid at an artificially small distance and
                        // pumps the shore (the bowl-lake regression).
                        let s = (dy - gx * dx - gz * dz) / nrm;
                        let dk = if s < F::new(0.0_f32) { -s } else { s };
                        best = best.min(dk);
                    } else {
                        // Neighbor cells: distance to the actual foot point on the
                        // patch (no plane extension beyond its cell).
                        let dk = (dx * dx + dy * dy + dz * dz).sqrt();
                        best = best.min(dk);
                    }
                }
            }
            best.max(F::new(0.001_f32))
        } else {
            yi.max(F::new(0.001_f32))
        };
        // Closed-box world walls (periodic == 0) are boundaries too: the fluid pile
        // against them has its kernel support truncated exactly like at the terrain,
        // and without the renormalization it erupts (the production corner eruption).
        // The planes are exact flat walls, so the distance to each is exact.
        let d = if periodic == 0 {
            let dw = xi.min(domain - xi).min(zi).min(domain - zi);
            d.min(dw.max(F::new(0.001_f32)))
        } else {
            d
        };
        let q0 = (d / h).min(F::new(2.0_f32));
        // Closed-form support fraction (see the `halfspace_c` docs above).
        let mut below = F::new(0.0_f32);
        if q0 < F::new(1.0_f32) {
            let q2 = q0 * q0;
            let q3 = q2 * q0;
            let q5 = q3 * q2;
            let q6 = q5 * q0;
            let fa1 = F::new(0.158333_f32) - F::new(0.275_f32) * q0;
            let fa0 = F::new(-0.1666667_f32) * q3 + F::new(0.075_f32) * q5 - F::new(0.025_f32) * q6;
            let fb = F::new(0.0916667_f32) - F::new(0.05_f32) * q0;
            below = F::new(2.0_f32) * (fa1 - fa0 + fb);
        } else {
            if q0 < F::new(2.0_f32) {
                let t = F::new(2.0_f32) - q0;
                let t4 = t * t * t * t;
                let t5 = t4 * t;
                let t6 = t5 * t;
                below = F::new(2.0_f32) * (F::new(0.125_f32) * t * t4
                    - F::new(0.05_f32) * (F::new(4.0_f32) - q0) * t5
                    + t6 / F::new(24.0_f32));
            }
        }
        let corr = (F::new(1.0_f32) - below).max(F::new(0.5_f32));
        state[p] = (rho + diffusion * diff) / corr;
    } else {
        state[p] = rho + diffusion * diff;
    }
    // Clamp at rho0: no negative (tensile) pressure. This avoids the classical WCSPH
    // pairwise tensile instability, which would otherwise collapse a column into a
    // monolayer (fluid "sticking" to itself).
    let ratio = state[p].max(rho0 * rho_min_frac) / rho0;
    state[p + 1] = b * (ratio.powf(gamma) - F::new(1.0_f32));
    state[p + 2] = F::new(0.0_f32);
    state[p + 3] = F::new(0.0_f32);
    cache[p] = state[p + 1] * (F::new(1.0_f32) / (state[p] * state[p]));
    cache[p + 1] = state[p];
    cache[p + 2] = F::new(0.0_f32);
    cache[p + 3] = F::new(0.0_f32);
}

/// WCSPH accelerations: pressure gradient only (dense; viscosity added next).
#[cube(launch_unchecked)]
pub(super) fn compute_force<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    vel: &[F],
    // [p/rho², rho, 0, 0] per particle, written by `compute_density` right before.
    cache: &[F],
    scan: &[I],
    list: &[I],
    acc: &mut [F],
    // XSPH velocity increment (raw velocities; see the doc comment below). The force
    // pass already visits every in-support neighbor and reads vel[pj], so summing the
    // smoothing term here costs almost nothing — the old separate `xsph_pass` was a
    // third full sweep over the same 3x3x3 window (the largest single kernel at scale).
    delta: &mut [F],
    h: F,
    h2: F,
    sigma: F,
    mass: F,
    alpha: F,
    speed_of_sound: F,
    xsph: F,
    cell_inv: F,
    domain: F,
    cx: usize,
    cy: usize,
    cz: usize,
    num: usize,
    use_grid: u32,
    periodic: u32,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    let xi = pos[p];
    let yi = pos[p + 1];
    let zi = pos[p + 2];
    let k_i = cache[p];
    let rho_i = cache[p + 1];

    let support2 = F::new(4.0_f32) * h2;
    let mut ax = F::new(0.0_f32);
    let mut ay = F::new(0.0_f32);
    let mut az = F::new(0.0_f32);
    // XSPH: velocity-smoothed advection increment Σ m·w/ρ̄ · (v_j − v_i). Raw velocities
    // (the half-kicked v_j + a_j·dt/2 can't be read in-kernel — acc is being written, a
    // race; the raw form is Monaghan's standard XSPH). Momentum-conserving (antisymmetric
    // pairs), dissipative by design, tick-deterministic.
    let mut avx = F::new(0.0_f32);
    let mut avy = F::new(0.0_f32);
    let mut avz = F::new(0.0_f32);

    if use_grid == 0 {
        for j in 0..num {
            if j != i {
                let pj = j * 4;
                let mut dx = xi - pos[pj];
                let mut dy = yi - pos[pj + 1];
                let mut dz = zi - pos[pj + 2];
                if periodic == 1 {
                    let half = domain * F::new(0.5_f32);
                    if dx > half {
                        dx = dx - domain;
                    }
                    if dx < -half {
                        dx = dx + domain;
                    }
                    if dy > half {
                        dy = dy - domain;
                    }
                    if dy < -half {
                        dy = dy + domain;
                    }
                    if dz > half {
                        dz = dz - domain;
                    }
                    if dz < -half {
                        dz = dz + domain;
                    }
                }
                if periodic == 2 {
                    let half = domain * F::new(0.5_f32);
                    if dx > half {
                        dx = dx - domain;
                    }
                    if dx < -half {
                        dx = dx + domain;
                    }
                    if dz > half {
                        dz = dz - domain;
                    }
                    if dz < -half {
                        dz = dz + domain;
                    }
                }
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
                let k_j = cache[pj];
                let rho_j = cache[pj + 1];
                let mut coeff = mass * (k_i + k_j);

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
                if xsph > F::new(0.0_f32) {
                    let w = sigma * (F::new(1.0_f32) - F::new(1.5_f32) * q * q + F::new(0.75_f32) * q * q * q);
                    let f = mass * w / ((rho_i + rho_j) * F::new(0.5_f32));
                    avx = avx - f * vx;
                    avy = avy - f * vy;
                    avz = avz - f * vz;
                }
            }
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

        for cdz in 0..3i32 {
            for cdy in 0..3i32 {
                for cdx in 0..3i32 {
                    let nc = icx + (cdx - 1);
                    let nr = icy + (cdy - 1);
                    let nh = icz + (cdz - 1);
                    if nc >= 0 && nc < cx as i32 && nr >= 0 && nr < cy as i32
                        && nh >= 0 && nh < cz as i32
                    {
                        let cell = nc as usize + cx * (nr as usize + cy * nh as usize);
                        let r_start = usize::cast_from(scan[cell]);
                        let r_end = usize::cast_from(scan[cell + 1]);
                        for k in r_start..r_end {
                            let j = usize::cast_from(list[k]);
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
                                    let k_j = cache[pj];
                                    let rho_j = cache[pj + 1];
                                    let mut coeff = mass * (k_i + k_j);

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
                                    if xsph > F::new(0.0_f32) {
                                        let w = sigma * (F::new(1.0_f32) - F::new(1.5_f32) * q * q + F::new(0.75_f32) * q * q * q);
                                        let f = mass * w / ((rho_i + rho_j) * F::new(0.5_f32));
                                        avx = avx - f * vx;
                                        avy = avy - f * vy;
                                        avz = avz - f * vz;
                                    }
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
    delta[p] = avx;
    delta[p + 1] = avy;
    delta[p + 2] = avz;
    delta[p + 3] = F::new(0.0_f32);
}

/// Velocity-Verlet half step + boundary contact. Gravity is applied here (y-down is
/// negative). `vel_half = vel + (acc - g)*dt/2`, `pos_new = pos + vel_half*dt`, with
/// world walls and heightfield-terrain push-out + normal projection. `delta` is the
/// XSPH velocity increment from the merged force pass (applied only when `xsph > 0`).
#[cube(launch_unchecked)]
pub(super) fn integrate_half<F: Float + LaunchArg>(
    pos_in: &[F],
    vel_in: &[F],
    acc_in: &[F],
    delta: &[F],
    pos_out: &mut [F],
    vel_half: &mut [F],
    terrain: &[F],
    dt: F,
    gravity: F,
    domain_size: F,
    friction: F,
    xsph: F,
    h: F,
    tx: usize,
    ty: usize,
    num: usize,
    periodic: u32,
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

    // XSPH (Monaghan): advect with the velocity-smoothed v-bar. The neighbor sum is
    // folded into the force pass (raw velocities — see `compute_force`); here the
    // precomputed increment is applied, guarded so a stale/zero delta is inert when
    // xsph == 0.
    if xsph > F::new(0.0_f32) {
        vx = vx + xsph * delta[p];
        vy = vy + xsph * delta[p + 1];
        vz = vz + xsph * delta[p + 2];
    }

    let mut x = pos_in[p] + vx * dt;
    let mut y = pos_in[p + 1] + vy * dt;
    let mut z = pos_in[p + 2] + vz * dt;

    if periodic == 1 {
        // Wrap into [0, domain) on all axes; no walls, no terrain, no friction.
        x = x - domain_size * (x / domain_size).floor();
        y = y - domain_size * (y / domain_size).floor();
        z = z - domain_size * (z / domain_size).floor();
    } else {
    if periodic == 2 {
        // Periodic x/z with a hard floor. The contact stand-off is half a particle
        // spacing (s = h/1.3): a layer resting ON the wall plane (y=0) would double-count
        // with its own mirror ghost (the classic MPM resolution requires the equilibrium
        // stand-off instead of a hard plane).
        x = x - domain_size * (x / domain_size).floor();
        z = z - domain_size * (z / domain_size).floor();
        let floor_rest = F::new(0.5_f32) * (h / F::new(1.3_f32));
        if y < floor_rest {
            y = floor_rest;
            if vy < F::new(0.0_f32) {
                vy = F::new(0.0_f32);
            }
            let damp = (F::new(1.0_f32) - friction * dt).max(F::new(0.0_f32));
            vx = vx * damp;
            vz = vz * damp;
        }
    } else {
    // World walls (closed box). The clamp uses the SAME stand-off as the periodic
    // floor: a layer resting exactly ON the wall plane (z = domain_size) has its
    // renormalization distance collapse to d = 0 -> c(0) = 0.5 -> density x2 -> EoS
    // eruption of the pile (measured rho 1.5 rho0, p ~ 1 MPa at the corner).
    let floor_rest = F::new(0.5_f32) * (h / F::new(1.3_f32));
    if x < floor_rest {
        x = floor_rest;
        vx = F::new(0.0_f32);
    }
    if x > domain_size - floor_rest {
        x = domain_size - floor_rest;
        vx = F::new(0.0_f32);
    }
    if z < floor_rest {
        z = floor_rest;
        vz = F::new(0.0_f32);
    }
    if z > domain_size - floor_rest {
        z = domain_size - floor_rest;
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

        // SHALLOW penetration: push along the normal at up to 0.1 m/step (fast
        // absorption of slamming/compression; normal transient penetrations stay far
        // below that) and reflect the normal velocity.
        //
        // DEEP burial (pen >= 2 particle spacings -- e.g. water spawned or terrain
        // edited into the solid): RESCUE straight up to the local surface with zero
        // vertical velocity. Capped-normal pushes make buried particles swim through
        // the fluid at up to 24 m/s (0.1 m/step displacement cap) and a full normal
        // projection can deposit them INSIDE the pool (overlap shock -> 139 m/s); the
        // rescue reassembles them on the surface, where they fall/slide back into the
        // fluid. (A vertical-only push stays wrong ONLY for shallow penetration:
        // sliding particles would pop on top of steep faces -- hence the two regimes.)
        let deep = F::new(2.0_f32) * (h / F::new(1.3_f32));
        if pen > deep {
            y = b + F::new(0.5_f32) * (h / F::new(1.3_f32));
            vy = F::new(0.0_f32);
        } else {
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
        }
        let damp = (F::new(1.0_f32) - friction * dt).max(F::new(0.0_f32));
        vx = vx * damp;
        vy = vy * damp;
        vz = vz * damp;
    }
    }
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
pub(super) fn finish_verlet<F: Float + LaunchArg>(
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
pub(super) fn splat_clear<I: Int + LaunchArg>(splat: &mut [I], cells: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= cells * cells {
        terminate!();
    }
    splat[i] = I::new(0);
}

/// Scatter particles into cells: increment the per-cell counter with an integer atomic.
#[cube(launch_unchecked)]
pub(super) fn particle_to_splat<I: Int + LaunchArg, F: Float + LaunchArg>(
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
pub(super) fn counts_to_texels<I: Int + LaunchArg, F: Float + LaunchArg>(
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

/// Phase 3 surface kernels: per-cell maximum particle height on a 2D cell grid.
///
/// Height is encoded as `u32(f32::min(y, ...) * 1000)` (mm resolution) so the atomic
/// max can use integer atomics (float atomics are rejected by naga/wgpu-30).
#[cube(launch_unchecked)]
pub(super) fn surface_clear<I: Int + LaunchArg>(maxes: &mut [I], n: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i < n {
        maxes[i] = I::new(0);
    }
}

#[cube(launch_unchecked)]
pub(super) fn particle_to_surface<F: Float + LaunchArg, I: Int + LaunchArg>(
    pos: &[F],
    maxes: &[Atomic<I>],
    domain_size: F,
    cells: usize,
    num: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= num {
        terminate!();
    }
    let p = i * 4;
    let x = clamp(
        pos[p] / domain_size * F::cast_from(cells as f32),
        F::new(0.0_f32),
        F::cast_from((cells - 1) as f32),
    );
    let z = clamp(
        pos[p + 2] / domain_size * F::cast_from(cells as f32),
        F::new(0.0_f32),
        F::cast_from((cells - 1) as f32),
    );
    let cell = usize::cast_from(x) + usize::cast_from(z) * cells;
    let enc = I::cast_from(pos[p + 1] * F::new(1000.0_f32));
    maxes[cell].fetch_max(enc);
}

/// Decode the max heights and write `[surface, depth, 0, 0]` texels (depth vs the
/// bilinear terrain height at each cell center).
#[cube(launch_unchecked)]
pub(super) fn surface_to_texels<F: Float + LaunchArg, I: Int + LaunchArg>(
    maxes: &[I],
    terrain: &[F],
    texels: &mut [F],
    domain_size: F,
    cells: usize,
    tx: usize,
    ty: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= cells * cells {
        terminate!();
    }
    let p = i * 4;
    let cx = i % cells;
    let cz = i / cells;
    let surf = F::cast_from(maxes[i]) / F::new(1000.0_f32);
    let cell_uv = (F::cast_from(cx as f32) + F::new(0.5_f32)) / F::cast_from(cells as f32);
    let wx = cell_uv * domain_size;
    let wz = (F::cast_from(cz as f32) + F::new(0.5_f32)) / F::cast_from(cells as f32) * domain_size;

    let th = domain_size / F::cast_from(tx as f32);
    let fx = wx / th;
    let fz = wz / th;
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

    // Empty cells (no particle contributed: surf == 0) have NO water: depth must be 0,
    // not `-terrain` (which would paint a fake sheet over below-zero terrain valleys).
    // Depth is only meaningful where a particle actually touched the cell.
    let is_water = if surf > F::new(0.0_f32) {
        F::new(1.0_f32)
    } else {
        F::new(0.0_f32)
    };
    let depth = (surf - b).max(F::new(0.0_f32)) * is_water;
    texels[p] = surf;
    texels[p + 1] = depth;
    texels[p + 2] = F::new(0.0_f32);
    texels[p + 3] = F::new(0.0_f32);
}

// ---------------------------------------------------------------------------
// Setup & main loop
// ---------------------------------------------------------------------------

/// Builds the neighbor grid for the current positions (zero -> count -> prefix -> zero ->
/// scatter). The extra zero between the prefix and the scatter is the key: the scatter
/// atomically increments the counts to claim slots, and it must start from 0 (`grid_count`
/// already led them to the full counts; `grid_prefix` only reads them). Without the
/// re-zero every cell's slots start at count(cell) and the list lands one count late —
/// the historical "cubecl-wgpu scatter nondeterminism".
pub(super) fn build_grid(client: &Client, p: &WaterParticles, pos: &Handle) {
    let dim = LAUNCH_DIM;
    // Number of 64-thread workgroups needed for `n` parallel items.
    let wgs = |n: usize| n.div_ceil(dim as usize) as u32;
    let cells = p.cells_x * p.cells_y * p.cells_z;
    let cell_inv = p.cell_inv;
    let treads = 64usize; // cells per block; one thread sums one block's 64 counts
    let nblocks = cells.div_ceil(treads);
    unsafe {
        grid_zero::launch_unchecked::<u32>(
            client,
            CubeCount::Static(wgs(cells), 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            cells,
        );
        grid_count::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(wgs(p.num), 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            cell_inv,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
        );
        // Hierarchical parallel prefix: the old single-threaded `grid_prefix` ran a
        // 17,500-element serial dependency chain (~6-9 ms per build, 8 builds per tick
        // = ~52 ms/tick). Block totals (all parallel) -> tiny block prefix -> inner scan.
        grid_block_totals::launch_unchecked::<u32>(
            client,
            CubeCount::Static(wgs(nblocks), 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            BufferArg::from_raw_parts(p.grid_totals.clone(), nblocks),
            cells,
            treads,
            nblocks,
        );
        grid_block_prefix::launch_unchecked::<u32>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(p.grid_totals.clone(), nblocks),
            BufferArg::from_raw_parts(p.grid_offsets.clone(), nblocks + 1),
            nblocks,
        );
        grid_inner_scan::launch_unchecked::<u32>(
            client,
            CubeCount::Static(wgs(cells), 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            BufferArg::from_raw_parts(p.grid_offsets.clone(), nblocks + 1),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            cells,
            treads,
        );
        // The scatter claims each cell's slots by atomically incrementing `counts`, so
        // the counters must be zero at this point: `grid_count` left them holding the
        // per-cell totals and the scan passes only read them. Without the re-zero,
        // every cell's slots would start at count(cell), shifting each range's entries
        // one count into the next range and leaving the range heads untouched — which
        // historically read as "cubecl-wgpu atomics are non-deterministic" (the
        // invariant test failed on untouched zero entries) while the atomics were
        // always correct.
        grid_zero::launch_unchecked::<u32>(
            client,
            CubeCount::Static(wgs(cells), 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            cells,
        );
        grid_scatter::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(wgs(p.num), 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            BufferArg::from_raw_parts(p.grid_counts.clone(), cells),
            BufferArg::from_raw_parts(p.grid_list.clone(), p.num),
            cell_inv,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
        );
    }
}

/// Force pass: density + pressure + accelerations at the current positions.
#[allow(clippy::too_many_arguments)]

pub(super) fn sph_force_pass(
    client: &Client,
    p: &WaterParticles,
    pos: &Handle,
    vel: &Handle,
    state: &Handle,
    acc: &Handle,
    rho_prev: &Handle,
    params: SphParams,
) {
    // Destructure once so the kernel bodies keep reading the short, familiar names.
    let SphParams {
        h, h2, sigma, mass, rho0, rho_min_frac, b, gamma, alpha, speed_of_sound, xsph,
        neighbor, domain, boundary, correction, diffusion, ..
    } = params;
    let dim = LAUNCH_DIM;
    let cells = p.cells_x * p.cells_y * p.cells_z;
    let cell_inv = p.cell_inv;
    unsafe {
        compute_density::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            BufferArg::from_raw_parts(p.grid_list.clone(), p.num),
            BufferArg::from_raw_parts(state.clone(), p.num * 4),
            BufferArg::from_raw_parts(rho_prev.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.press_cache.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.terrain.clone(), p.terrain_x * p.terrain_y),
            p.terrain_x,
            p.terrain_y,
            h,
            h2,
            sigma,
            mass,
            diffusion,
            rho0,
            rho_min_frac,
            b,
            gamma,
            cell_inv,
            domain,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
            neighbor as u32,
            boundary as u32,
            correction as u32,
        );
        compute_force::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(vel.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.press_cache.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.grid_scan.clone(), cells + 1),
            BufferArg::from_raw_parts(p.grid_list.clone(), p.num),
            BufferArg::from_raw_parts(acc.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.xsph_delta.clone(), p.num * 4),
            h,
            h2,
            sigma,
            mass,
            alpha,
            speed_of_sound,
            xsph,
            cell_inv,
            domain,
            p.cells_x,
            p.cells_y,
            p.cells_z,
            p.num,
            neighbor as u32,
            boundary as u32,
        );
    }
}

/// One velocity-Verlet substep with two force evaluations (each preceded by a grid
/// rebuild for its positions).
#[allow(clippy::too_many_arguments)]
pub(super) fn sph_substep(
    client: &Client,
    p: &WaterParticles,
    pos_in: &Handle,
    vel_in: &Handle,
    pos_out: &Handle,
    vel_out: &Handle,
    state_scratch: &Handle,
    acc_scratch: &Handle,
    rho_prev: &Handle,
    dt: f32,
    params: SphParams,
) {
    // Destructure once so the kernel bodies keep reading the short, familiar names.
    // (The EoS coefficients and the diffusion term are consumed by the force passes via
    // `params`, not here — they are only part of the integration inputs.)
    let SphParams {
        h, neighbor, domain, gravity, friction, boundary, xsph, ..
    } = params;
    let dim = LAUNCH_DIM;

    // Force at the current positions.
    if neighbor == NeighborSearch::UniformGrid {
        build_grid(client, p, pos_in);
    }
    sph_force_pass(
        client,
        p,
        pos_in,
        vel_in,
        state_scratch,
        acc_scratch,
        rho_prev,
        params,
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
            BufferArg::from_raw_parts(p.xsph_delta.clone(), p.num * 4),
            BufferArg::from_raw_parts(pos_out.clone(), p.num * 4),
            BufferArg::from_raw_parts(vel_out.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.terrain.clone(), p.terrain_x * p.terrain_y),
            dt,
            gravity,
            domain,
            friction,
            xsph,
            h,
            p.terrain_x,
            p.terrain_y,
            p.num,
            boundary as u32,
        );
    }

    // Force at the new positions.
    if neighbor == NeighborSearch::UniformGrid {
        build_grid(client, p, pos_out);
    }
    sph_force_pass(
        client,
        p,
        pos_out,
        vel_out,
        state_scratch,
        acc_scratch,
        rho_prev,
        params,
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
            gravity,
            p.num,
        );
        // Record the latest densities as the deterministic delta-SPH input for the next pass.
        copy_rho_prev::launch_unchecked::<f32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(state_scratch.clone(), p.num * 4),
            BufferArg::from_raw_parts(rho_prev.clone(), p.num * 4),
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
    // Float the blob above the LOCAL terrain: a fixed base_y can be below the ridge
    // surface, where the half-space density correction sees d ~ 0 (c -> 0.5) and doubles
    // the density -> Tait EoS explosion at startup (particles to 30+ m, then tunneling
    // through the bed). Empirically the blob must rest above the footprint.
    let mut base_y = 1.5;
    {
        let half = sx.max(sz) * 0.5;
        let mut local_max = f32::MIN;
        for iz in 0..=16 {
            for ix in 0..=16 {
                let p = Vec2::new(
                    cx - half + 2.0 * half * ix as f32 / 16.0,
                    cz - half + 2.0 * half * iz as f32 / 16.0,
                );
                local_max = local_max.max(terrain.height_map.get_bilinear(p));
            }
        }
        base_y = local_max.max(base_y) + 0.5;
    }

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
    let rho_prev = client.empty((num * 4) * core::mem::size_of::<f32>());
    let acc_a = client.empty((num * 4) * core::mem::size_of::<f32>());
    let acc_b = client.empty((num * 4) * core::mem::size_of::<f32>());
    let xsph_delta = client.empty((num * 4) * core::mem::size_of::<f32>());
    let press_cache = client.empty((num * 4) * core::mem::size_of::<f32>());
    client.write(&pos_a, bytes(&pos));
    client.write(&vel_a, bytes(&vel));
    client.write(
        &rho_prev,
        bytes(&vec![params.fluid_density; num * 4]),
    );

    let terrain_size = terrain.height_map.size;
    let terrain_bytes = terrain.height_map.buffer.len() * core::mem::size_of::<f32>();
    let terrain_handle = client.empty(terrain_bytes);
    client.write(&terrain_handle, bytes(&terrain.height_map.buffer));

    // Neighbor grid: cell size = SPH support (2h).
    let cell_size = 2.0 * s * params.smoothing_ratio;
    let cells_x = (crate::domain::SIZE_F32.x / cell_size).ceil() as usize;
    let cells_y = (crate::domain::SIZE_F32.y / cell_size).ceil() as usize;
    let cells_z = (crate::domain::SIZE_F32.y / cell_size).ceil() as usize;
    let cells = cells_x * cells_y * cells_z;
    let grid_counts = client.empty(cells * core::mem::size_of::<u32>());
    let grid_scan = client.empty((cells + 1) * core::mem::size_of::<u32>());
    let grid_list = client.empty(num * core::mem::size_of::<u32>());
    let treads = 64usize;
    let nblocks = cells.div_ceil(treads);
    let grid_totals = client.empty(nblocks * core::mem::size_of::<u32>());
    let grid_offsets = client.empty((nblocks + 1) * core::mem::size_of::<u32>());

    let splat_cells = 256usize;
    let splat_counts = client.empty(splat_cells * splat_cells * core::mem::size_of::<u32>());
    let splat = client.empty(splat_cells * splat_cells * 4 * core::mem::size_of::<f32>());
    let surface_max = client.empty(splat_cells * splat_cells * core::mem::size_of::<u32>());
    let surface =
        client.empty(splat_cells * splat_cells * 4 * core::mem::size_of::<f32>());

    commands.insert_resource(WaterParticles {
        num,
        pos_a,
        pos_b,
        vel_a,
        vel_b,
        state_a,
        state_b,
        rho_prev,
        acc_a,
        acc_b,
        xsph_delta,
        press_cache,
        terrain: terrain_handle,
        terrain_x: terrain_size.x,
        terrain_y: terrain_size.y,
        grid_counts,
        grid_scan,
        grid_list,
        grid_totals,
        grid_offsets,
        cells_x,
        cells_y,
        cells_z,
        cell_inv: 1.0 / cell_size,
        splat_counts,
        splat,
        surface_max,
        surface,
        cells: splat_cells,
        current_is_b: false,
    });
}

/// Phase 2 main loop: `substeps` fixed velocity-Verlet steps per physics tick, then refresh
/// the debug splat and publish the buffer for rendering.
/// Raw GPU buffer and its byte offset backing a cubecl handle, so the render world can
/// copy from it directly (`WgpuServer` keeps the cubecl buffer alive via the handle).
/// Panics if the handle is gone — the sim structures must keep it referenced.
fn gpu_buffer_of(client: &Client, handle: &Handle) -> (cubecl::wgpu::wgpu::Buffer, u64) {
    let managed = client
        .get_resource::<cubecl::wgpu::WgpuServer<cubecl::wgpu::AutoCompiler>>(handle.clone())
        .expect("cubecl buffer should still be alive");
    let resource = managed.resource();
    (resource.buffer.clone(), resource.offset)
}

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


    let sub = wp.substeps.max(1);
    let dt =
        1.0 / parameters::PHYSICS_TICKS_PER_SEC as f32 / sub as f32;
    let s = wp.particle_spacing;
    let sph = SphParams::from_water_parameters(wp, s, crate::domain::SIZE_F32.x);
    let dim = LAUNCH_DIM;
    let cells_sq = p.cells * p.cells;

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
            &p.rho_prev,
            dt,
            sph,
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
            CubeCount::Static(cells_sq.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.splat_counts.clone(), cells_sq),
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
            CubeCount::Static(cells_sq.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.splat_counts.clone(), cells_sq),
            BufferArg::from_raw_parts(p.splat.clone(), cells_sq * 4),
            wp.debug_count_scale,
            p.cells,
        );
    }

    let (water_buf, water_off) = gpu_buffer_of(client, &p.splat);
    field_vis_gpu.water.buffer = Some(water_buf);
    field_vis_gpu.water.offset = water_off;
    field_vis_gpu.water.width = p.cells as u32;
    field_vis_gpu.water.height = p.cells as u32;
    if field_vis_gpu.image != terrain_assets.field_vis_image {
        field_vis_gpu.image = terrain_assets.field_vis_image.clone();
    }

    // Phase 4: publish the current particle positions (the sprite layer reads them via
    // `gpu::copy_particles_to_shader_buffer`; the ping-pong buffers hold the same data).
    let (particle_buf, particle_off) = gpu_buffer_of(client, cur_pos);
    field_vis_gpu.particles.buffer = Some(particle_buf);
    field_vis_gpu.particles.offset = particle_off;
    field_vis_gpu.particles.count = p.num as u32;

    // Phase 3 surface: clear the per-cell max, scatter particle heights (integer atomic
    // max; the `cubecl::Atomic` only supports integer ops), decode to texels with the
    // depth vs the terrain, and publish for the renderer (copied every frame by
    // `gpu::copy_field_vis_to_texture`, independent of the F1/F3/F5 field selection).
    unsafe {
        surface_clear::launch_unchecked::<u32>(
            client,
            CubeCount::Static(cells_sq.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.surface_max.clone(), cells_sq),
            cells_sq,
        );
        particle_to_surface::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static(p.num.div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(cur_pos.clone(), p.num * 4),
            BufferArg::from_raw_parts(p.surface_max.clone(), p.cells * p.cells),
            crate::domain::SIZE_F32.x,
            p.cells,
            p.num,
        );
        surface_to_texels::launch_unchecked::<f32, u32>(
            client,
            CubeCount::Static((p.cells * p.cells).div_ceil(dim as usize) as u32, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(p.surface_max.clone(), p.cells * p.cells),
            BufferArg::from_raw_parts(p.terrain.clone(), p.terrain_x * p.terrain_y),
            BufferArg::from_raw_parts(p.surface.clone(), p.cells * p.cells * 4),
            crate::domain::SIZE_F32.x,
            p.cells,
            p.terrain_x,
            p.terrain_y,
        );
    }
    let (surface_buf, surface_off) = gpu_buffer_of(client, &p.surface);
    field_vis_gpu.surface.buffer = Some(surface_buf);
    field_vis_gpu.surface.offset = surface_off;
    field_vis_gpu.surface.width = p.cells as u32;
    field_vis_gpu.surface.height = p.cells as u32;
    field_vis_gpu.surface_image = terrain_assets.water_surface_image.clone();
}
