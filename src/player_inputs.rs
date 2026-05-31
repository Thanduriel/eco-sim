use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;
use std::time::Duration;

use crate::grass;
use crate::organism;
use crate::parameters;
use crate::terrain::*;

#[derive(Default, PartialEq, Copy, Clone)]
enum FieldType {
    #[default]
    None,
    VegDensity,
    SoilMoisture,
}

#[derive(Resource, Default)]
pub struct FieldVisState {
    field_type: FieldType,
}

pub fn vis_fields_system(
    mut terrain_query: Query<(&mut MeshMaterial3d<StandardMaterial>, &Mesh3d), With<Terrain>>,
    surface_query: Query<&Surface>,
    key_input: Res<ButtonInput<KeyCode>>,
    mut field_vis_state: ResMut<FieldVisState>,
    terrain_assets: Res<TerrainAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // check user inputs
    let prev_field_vis_type = field_vis_state.field_type;
    if key_input.just_pressed(KeyCode::F1) {
        field_vis_state.field_type = FieldType::None;
    } else if key_input.just_pressed(KeyCode::F2) {
        field_vis_state.field_type = FieldType::VegDensity;
    } else if key_input.just_pressed(KeyCode::F3) {
        field_vis_state.field_type = FieldType::SoilMoisture;
    }

    let (mut mat3d, mesh3d) = terrain_query.single_mut().unwrap();

    // update material
    if prev_field_vis_type != field_vis_state.field_type {
        match field_vis_state.field_type {
            FieldType::None => {
                mat3d.0 = terrain_assets.ground_material.clone();
                if let Some(mesh) = meshes.get_mut(&mesh3d.0) {
                    reset_terrain_color(mesh);
                }
            }
            FieldType::VegDensity | FieldType::SoilMoisture => {
                mat3d.0 = terrain_assets.field_vis_material.clone();
            }
        }
    }

    if field_vis_state.field_type == FieldType::None {
        return;
    }

    // update vertex colors to visualize field
    //let now = std::time::Instant::now();
    let surface = surface_query.single().unwrap();
    if let Some(mesh) = meshes.get_mut(&mesh3d.0) {
        match field_vis_state.field_type {
            FieldType::None => panic!(),
            FieldType::VegDensity => {
                set_terrain_color(mesh, &surface.veg_density, Some((0.0, 1.0)));
            }
            FieldType::SoilMoisture => {
                set_terrain_color(mesh, &surface.soil_moisture, Some((0.0, 1.0)));
            }
        };
    }

    //println!("{}", now.elapsed().as_secs_f64());
}

#[derive(Default, PartialEq, Copy, Clone)]
pub enum PickingMode {
    #[default]
    Seed,
    Water,
}

pub fn picking_system(
    mut commands: Commands,
    grass_assets: Res<grass::GrassAssets>,
    mut ray_cast: MeshRayCast,
    terrain_query: Query<(), With<Terrain>>,
    mut surface_query: Query<&mut Surface>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    window_query: Query<&Window>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    key_input: Res<ButtonInput<KeyCode>>,
    mut rng: Single<&mut WyRand, With<GlobalRng>>,
    mut picking_mode: Local<PickingMode>,
) {
    if key_input.just_released(KeyCode::Digit1) {
        *picking_mode = PickingMode::Seed;
    } else if key_input.just_released(KeyCode::Digit2) {
        *picking_mode = PickingMode::Water;
    }

    if !mouse_button_input.just_released(MouseButton::Right) {
        return;
    }

    let Ok(window) = window_query.single() else {
        return;
    };

    let Some(cursor_position) = window.cursor_position() else {
        return;
    };

    let Ok((camera, camera_global_transform)) = camera_query.single() else {
        return;
    };

    let Ok(ray) = camera.viewport_to_world(camera_global_transform, cursor_position) else {
        return;
    };

    let filter = |entity| terrain_query.contains(entity);
    let early_exit_test = |_entity| true;

    // Ignore the visibility of entities. This allows ray casting hidden entities.
    let visibility = RayCastVisibility::Any;

    let settings = MeshRayCastSettings::default()
        .with_filter(&filter)
        .with_early_exit_test(&early_exit_test)
        .with_visibility(visibility);

    // Cast the ray with the settings, returning a list of intersections.
    let hits = ray_cast.cast_ray(ray, &settings);

    for (_, hit) in hits {
        match *picking_mode {
            PickingMode::Seed => {
                commands.spawn(organism::OrganismBundle {
                    mesh: Mesh3d(grass_assets.mesh.clone()),
                    no_shadow: bevy::light::NotShadowCaster::default(),
                    material: MeshMaterial3d(grass_assets.material.clone()),
                    transform: Transform::from_translation(hit.point - vec3(0.0, 0.1, 0.0))
                        .with_scale(Vec3::ZERO)
                        .with_rotation(Quat::from_axis_angle(
                            Vec3::new(0.0, 1.0, 0.0),
                            rng.random::<f32>() * 2.0 * PI,
                        )),
                    organism: organism::Organism::default(),
                });
            }
            PickingMode::Water => {
                let mut surface = surface_query.single_mut().unwrap();
                surface.soil_moisture.add_kernel(hit.point.xz(), 2.0, 1.0);
            }
        };
    }
}

pub fn general_actions_system(
    key_input: Res<ButtonInput<KeyCode>>,
    mut time: ResMut<Time<Virtual>>,
    real_time: Res<Time<Real>>,
    params: Res<parameters::GeneralParameters>,
    mut frames_since_throttle: Local<i32>,
) {
    let relative_speed = time.relative_speed();

    // Reduce speed if the update can't keep up.
    // Wait a few steps in between for the framerate to stabilize.
    *frames_since_throttle += 1;
    if relative_speed > 1.0 && *frames_since_throttle > 8 {
        if real_time.delta() >= Duration::from_secs_f32(1.0 / params.game_speed.auto_slow_fps) {
            time.set_relative_speed(relative_speed * 0.5);
            *frames_since_throttle = 0;
            bevy::log::info!("Reducing game speed because the update can't keep up.");
        }
    }

    time.set_max_delta(Duration::from_secs_f32(
        params.game_speed.max_speed / parameters::PHYSICS_TICKS_PER_SEC,
    ));

    // user controls
    if relative_speed < params.game_speed.max_speed && key_input.just_pressed(KeyCode::ArrowUp) {
        time.set_relative_speed(relative_speed * 2.0);
    } else if relative_speed > params.game_speed.min_speed
        && key_input.just_pressed(KeyCode::ArrowDown)
    {
        time.set_relative_speed(relative_speed * 0.5);
    }
}
