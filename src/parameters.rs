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
    pub debug_count_scale: f32,
    /// Tangential damping on terrain contact (smooth the dambreak settling).
    pub terrain_friction: f32,
}

impl Default for WaterParameters {
    fn default() -> Self {
        WaterParameters {
            is_active: true,
            substeps: 4,
            particle_spacing: 0.5,
            blob_size: [12.0, 2.0, 12.0],
            debug_count_scale: 32.0,
            terrain_friction: 3.0,
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
