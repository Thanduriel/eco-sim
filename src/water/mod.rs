//! Water simulation and rendering, split by concern:
//! - [`physics`]: the numerical core — SPH kernels, uniform-grid neighbor search,
//!   `setup_water`/`run_water` (everything that was `src/water.rs`).
//! - [`render`]: the GPU-side surface plane and particle sprites (previously
//!   `src/water_surface.rs` + `src/particles.rs`).
//! - [`tests`]: the analytic + GPU test battery (CPU and wgpu backends).
//!
//! [`WaterPlugin`] wires all of it into Bevy.

pub mod physics;
pub mod render;

#[cfg(test)]
mod tests;

use bevy::prelude::*;

use crate::gpu;

/// Registers the water simulation + rendering: the cubecl sim loop (`FixedUpdate`), the
/// surface/particle materials and their uniform-refresh systems (`Update`), and the
/// render-world copy systems (surface texture + particle storage buffer + field-vis
/// overlay) through [`gpu::GpuRenderCopyPlugin`].
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            gpu::GpuRenderCopyPlugin,
            MaterialPlugin::<render::WaterSurfaceMaterial>::default(),
            MaterialPlugin::<render::ParticleMaterial>::default(),
        ))
        .add_systems(
            Startup,
            (
                physics::setup_water,
                render::setup_water_surface,
                render::setup_particles,
            )
                // The sim needs the terrain entity and the cubecl client; the surface
                // plane and the particle layer need the assets the two setups create.
                .chain()
                .after(gpu::setup_gpu_context)
                .after(crate::terrain::setup_terrain),
        )
        .add_systems(
            Update,
            (
                render::update_water_surface_uniform,
                render::update_particle_uniform,
            ),
        )
        .add_systems(FixedUpdate, physics::run_water);
    }
}
