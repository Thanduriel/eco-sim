use bevy::input::{keyboard::KeyboardInput, mouse::MouseButtonInput, mouse::MouseWheel};
use bevy::prelude::*;
use bevy_egui::egui::Align2;
use bevy_egui::{EguiContexts, egui};

use egui_probe::{Probe, EguiProbe};

use crate::grass;

#[derive(Resource, EguiProbe, Default)]
pub struct GeneralParameters {
   pub grass : grass::GrassParameters
}

#[derive(Default)]
pub struct LastMessages {
    keyboard_input: Option<KeyboardInput>,
    mouse_button_input: Option<MouseButtonInput>,
    mouse_wheel: Option<MouseWheel>,
}

pub fn parameter_ui_system(
    mut contexts: EguiContexts,
    mut general_params: ResMut<GeneralParameters>,
    mut last_messages: Local<LastMessages>,
    mut keyboard_input_reader: MessageReader<KeyboardInput>,
    mut mouse_button_input_reader: MessageReader<MouseButtonInput>,
    mut mouse_wheel_reader: MessageReader<MouseWheel>,
) -> Result {
    if let Some(message) = keyboard_input_reader.read().last() {
        last_messages.keyboard_input = Some(message.clone());
    }
    if let Some(message) = mouse_button_input_reader.read().last() {
        last_messages.mouse_button_input = Some(*message);
    }
    if let Some(message) = mouse_wheel_reader.read().last() {
        last_messages.mouse_wheel = Some(*message);
    }

    egui::Window::new("Parameters")
        .max_size([300.0, 200.0])
        .anchor(Align2::RIGHT_TOP, egui::vec2(5.0, 5.0))
        .vscroll(true)
        .show(contexts.ctx_mut()?, |ui| {
            Probe::new(&mut *general_params).show(ui);
        });

    Ok(())
}
