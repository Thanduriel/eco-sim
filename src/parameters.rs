use bevy::prelude::*;
use bevy_egui::egui::Align2;
use bevy_egui::{EguiContexts, egui};

use egui_probe::{EguiProbe, Probe};

use crate::grass;

#[derive(EguiProbe)]
pub struct SunParameters {
    pub day_duration: f32
}

impl Default for SunParameters {
    fn default() -> Self {
        SunParameters {day_duration : 120.0 }
    }
}

#[derive(Resource, EguiProbe, Default)]
pub struct GeneralParameters {
    pub sun : SunParameters,
    pub grass: grass::GrassParameters,
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
            .max_size([300.0, 200.0])
            .anchor(Align2::RIGHT_TOP, egui::vec2(5.0, 5.0))
            .vscroll(true)
            .show(contexts.ctx_mut()?, |ui| {
                Probe::new(&mut *general_params).show(ui);
            });
    }

    Ok(())
}
