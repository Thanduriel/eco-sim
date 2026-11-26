use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;
use std::time::Duration;

use crate::gpu;
use crate::grass;
use crate::organism;
use crate::parameters;
use crate::terrain::*;
use crate::water;

/// Handles tfield-visualization toggle. It only records the requested field
/// type; `terrain::field_vis_refresh_system` is responsible for applying it.
pub fn vis_fields_system(
    key_input: Res<ButtonInput<KeyCode>>,
    mut field_vis_state: ResMut<FieldVisState>,
) {
    if key_input.just_pressed(KeyCode::F1) {
        field_vis_state.field_type = FieldType::None;
    } else if key_input.just_pressed(KeyCode::F2) {
        field_vis_state.field_type = FieldType::VegDensity;
    } else if key_input.just_pressed(KeyCode::F3) {
        field_vis_state.field_type = FieldType::SoilMoisture;
    } else if key_input.just_pressed(KeyCode::F5) {
        field_vis_state.field_type = FieldType::WaterHeight;
    }
}

/// Which mouse tool is active (`1` / `2` / `3`). A resource (not a `Local`) because the
/// one-shot system and the stream system both read it.
#[derive(Resource, Default, PartialEq, Copy, Clone)]
pub enum PickingMode {
    #[default]
    Seed, // spawn a new organism (right-click)
    Water, // increase surface moisture (right-click)
    Stream, // pour water while the right mouse button is held
}

/// First terrain point under the cursor (nearest ray hit, any visibility).
fn terrain_point_under_cursor(
    ray_cast: &mut MeshRayCast,
    terrain_query: &Query<(), With<Terrain>>,
    camera_query: &Query<(&Camera, &GlobalTransform)>,
    window_query: &Query<&Window>,
) -> Option<Vec3> {
    let window = window_query.single().ok()?;
    let cursor_position = window.cursor_position()?;
    let (camera, camera_global_transform) = camera_query.single().ok()?;
    let ray = camera
        .viewport_to_world(camera_global_transform, cursor_position)
        .ok()?;

    let filter = |entity| terrain_query.contains(entity);
    let early_exit_test = |_entity| true;

    // Ignore the visibility of entities. This allows ray casting hidden entities.
    let visibility = RayCastVisibility::Any;

    let settings = MeshRayCastSettings::default()
        .with_filter(&filter)
        .with_early_exit_test(&early_exit_test)
        .with_visibility(visibility);

    // Cast the ray with the settings, returning a list of intersections (sorted by
    // distance: the nearest is the visible surface).
    let hits = ray_cast.cast_ray(ray, &settings);
    hits.into_iter().map(|(_, hit)| hit.point).next()
}

/// Stream brush: a continuous pour while the right mouse button is held, at a constant
/// height offset above the terrain under the cursor — the same button the other pick
/// tools use, so the camera's left-hold look is untouched.
pub fn water_stream_system(
    mut ray_cast: MeshRayCast,
    terrain_query: Query<(), With<Terrain>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    window_query: Query<&Window>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut rng: Single<&mut WyRand, With<GlobalRng>>,
    picking_mode: Res<PickingMode>,
    params: Res<parameters::GeneralParameters>,
    gpu_context: Option<Res<gpu::GpuContext>>,
    particles: Option<ResMut<water::physics::WaterParticles>>,
    mut spawn_state: Local<(f32, bool)>, // (fractional spawn budget, capacity-warned)
    time: Res<Time>,
) {
    if *picking_mode != PickingMode::Stream {
        return;
    }
    if !mouse_button_input.pressed(MouseButton::Right) {
        // Re-arm the capacity warning on the next press.
        spawn_state.1 = false;
        return;
    }
    if !params.water.is_active {
        // The sim is frozen (and stops publishing the particle count); don't queue
        // spawn data it won't integrate.
        return;
    }
    let (Some(gpu_context), Some(mut particles)) = (gpu_context, particles) else {
        return;
    };
    if particles.num >= particles.capacity {
        if !spawn_state.1 {
            bevy::log::warn!(
                "Water particle capacity reached ({}): the stream brush is clamped.",
                particles.capacity
            );
            spawn_state.1 = true;
        }
        return;
    }
    let wp = &params.water;
    spawn_state.0 += wp.spawn_flow * time.delta_secs();
    let n = spawn_state
        .0
        .floor()
        .min((particles.capacity - particles.num) as f32) as usize;
    spawn_state.0 -= n as f32;
    if n == 0 {
        return;
    }
    let Some(point) =
        terrain_point_under_cursor(&mut ray_cast, &terrain_query, &camera_query, &window_query)
    else {
        return;
    };
    let mut positions = Vec::with_capacity(n);
    for _ in 0..n {
        let jitter = Vec3::new(
            rng.random::<f32>() - 0.5,
            0.0,
            rng.random::<f32>() - 0.5,
        ) * (2.0 * wp.spawn_radius);
        positions.push(point + Vec3::Y * wp.spawn_height_offset + jitter);
    }
    water::physics::append_particles(
        &gpu_context.client,
        &mut particles,
        &positions,
        wp.fluid_density,
    );
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
    mut picking_mode: ResMut<PickingMode>,
) {
    if key_input.just_released(KeyCode::Digit1) {
        *picking_mode = PickingMode::Seed;
    } else if key_input.just_released(KeyCode::Digit2) {
        *picking_mode = PickingMode::Water;
    } else if key_input.just_released(KeyCode::Digit3) {
        *picking_mode = PickingMode::Stream;
    }

    // The stream is its own system (`water_stream_system`); only the one-shot tools live
    // here.
    if !mouse_button_input.just_released(MouseButton::Right) {
        return;
    }

    let Some(point) =
        terrain_point_under_cursor(&mut ray_cast, &terrain_query, &camera_query, &window_query)
    else {
        return;
    };

    match *picking_mode {
        PickingMode::Seed => {
            commands.spawn(organism::OrganismBundle {
                mesh: Mesh3d(grass_assets.mesh.clone()),
                no_shadow: bevy::light::NotShadowCaster::default(),
                material: MeshMaterial3d(grass_assets.material.clone()),
                transform: Transform::from_translation(point - vec3(0.0, 0.1, 0.0))
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
            surface.soil_moisture.add_kernel(point.xz(), 2.0, 1.0);
        }
        PickingMode::Stream => {}
    };
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
