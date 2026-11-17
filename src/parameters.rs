use bevy::prelude::*;
use bevy_egui::egui::Align2;
use bevy_egui::{EguiContexts, egui};
use egui_probe::{EguiProbe, Probe};

use crate::grass;

#[derive(EguiProbe)]
pub struct GameSpeedParameters {
    pub auto_slow_fps: f32,
    pub min_speed: f32,
    pub max_speed: f32,
}

pub const PHYSICS_TICKS_PER_SEC: f32 = 60.0;

impl Default for GameSpeedParameters {
    fn default() -> Self {
        GameSpeedParameters {
            auto_slow_fps: 15.0,
            min_speed: 0.125,
            max_speed: 64.0,
        }
    }
}

#[derive(EguiProbe)]
pub struct SunParameters {
    pub day_duration: f32,
    pub is_moving: bool,
}

impl Default for SunParameters {
    fn default() -> Self {
        SunParameters {
            day_duration: 120.0,
            is_moving: false,
        }
    }
}

#[derive(EguiProbe, Clone, Copy)]
pub enum BoundaryCondition {
    Dirichlet(f32),
    Closed,
}

#[derive(EguiProbe)]
pub struct DiffusionParameters {
    pub diffusivity: f32,
    pub bc: BoundaryCondition,
    pub is_active: bool,
}

#[derive(EguiProbe)]
pub struct GroundParameters {
    pub soil_water: DiffusionParameters,
}

impl Default for GroundParameters {
    fn default() -> Self {
        GroundParameters {
            soil_water: DiffusionParameters {
                diffusivity: 0.01,
                bc: BoundaryCondition::Dirichlet(0.5),
                is_active: true,
            },
        }
    }
}

#[derive(EguiProbe)]
pub struct WaterParameters {
    pub is_active: bool,
    /// Inner integration steps per physics tick (constant, keeps the fixed timestep).
    pub substeps: u32,
    /// Initial particle spacing (smaller = more particles).
    pub particle_spacing: f32,
    /// Initial water block extents (x, y, z) placed at the world center.
    pub blob_size: [f32; 3],
    /// Particle count per splat cell that maps to full color in the F5 overlay.
    ///
    /// The field-vis LUT (Incandescent in `color_map.rs`) is inverted: 0 = white,
    /// 1 = deep red. Seeded particles pack up to ~4 per 0.25 m splat cell, so the
    /// scale must be that magnitude for the blob to use the colored part of the LUT.
    pub debug_count_scale: f32,
    /// Tangential damping on terrain contact (smooth the dambreak settling).
    pub terrain_friction: f32,
    /// SPH: rest density ρ0 (kg/m³).
    pub fluid_density: f32,
    /// SPH: reference sound speed c0 (m/s); the Tait exponent gamma is implied by
    /// `p = B((rho/rho0)^gamma - 1)` with `B = rho0 c0^2 / gamma`.
    /// Calibrated with the analytic battery: c0 = 40 (c0 = 20 amplifies the drop-slam
    /// of the initial blob into an eruption on rugged terrain).
    pub sound_speed: f32,
    /// SPH: velocity smoothing used inside the half-step integrator (stabilizes the
    /// slosh; 0 = plain WCSPH). Test-validated value 0.2.
    pub xsph: f32,
    /// SPH: delta-SPH density diffusion coefficient (0 = off). Test-validated 0.0.
    pub diffusion: f32,
    /// SPH: Tait exponent gamma.
    pub gamma: f32,
    /// SPH: Monaghan artificial viscosity coefficient alpha.
    pub viscosity_alpha: f32,
    /// SPH: density clamp (`rho >= stiffness_clamp * rho0`) to limit negative pressure.
    pub stiffness_clamp: f32,
    /// SPH: smoothing length as a multiple of the particle spacing (`h = ratio * spacing`).
    pub smoothing_ratio: f32,
    /// Grid extent in y used for the neighbor cells (particles above this are invalid).
    pub max_particle_height: f32,
    /// Phase 3 rendering: water depth (m) at which the surface color saturates to "deep".
    pub surface_depth_scale: f32,
    /// Phase 3 rendering: water depth (m) at which the surface alpha saturates (shoreline
    /// transparency).
    pub surface_alpha_scale: f32,
}

impl Default for WaterParameters {
    fn default() -> Self {
        WaterParameters {
            is_active: true,
            substeps: 4,
            particle_spacing: 0.5,
            blob_size: [16.0, 2.0, 16.0],
            debug_count_scale: 4.0,
            terrain_friction: 3.0,
            fluid_density: 1000.0,
            sound_speed: 40.0,
            gamma: 7.0,
            viscosity_alpha: 0.3,
            xsph: 0.2,
            diffusion: 0.0,
            stiffness_clamp: 0.1,
            smoothing_ratio: 1.3,
            max_particle_height: 8.0,
            surface_depth_scale: 4.0,
            surface_alpha_scale: 1.0,
        }
    }
}

#[derive(Resource, EguiProbe, Default)]
pub struct GeneralParameters {
    pub game_speed: GameSpeedParameters,
    pub sun: SunParameters,
    pub grass: grass::GrassParameters,
    pub ground: GroundParameters,
    pub water: WaterParameters,
}

#[derive(Default)]
pub struct ParameterUiConfig {
    is_visible: bool,
}

pub fn parameter_ui_system(
    mut contexts: EguiContexts,
    mut ui_config: Local<ParameterUiConfig>,
    key_input: Res<ButtonInput<KeyCode>>,
    mut general_params: ResMut<GeneralParameters>,
) -> Result {
    if key_input.just_pressed(KeyCode::F4) {
        ui_config.is_visible = !ui_config.is_visible;
    }

    if ui_config.is_visible {
        egui::Window::new("Parameters")
            .default_open(true)
            .auto_sized()
            //.max_size([300.0, 200.0])
            .anchor(Align2::RIGHT_TOP, egui::vec2(5.0, 5.0))
            .vscroll(true)
            .show(contexts.ctx_mut()?, |ui| {
                Probe::new(&mut *general_params).show(ui);
            });
    }

    Ok(())
}
