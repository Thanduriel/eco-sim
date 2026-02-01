use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;

use crate::grass;
use crate::organism;
use crate::terrain::*;

#[derive(Default, PartialEq)]
enum FieldType {
    #[default]
    None,
    Reset,
    VegDensity,
}

#[derive(Resource, Default)]
pub struct FieldVisState {
    field_type: FieldType,
}

pub fn vis_fields_system(
    mut meshes: ResMut<Assets<Mesh>>,
    terrain_query: Query<(&Mesh3d, &Surface), With<Terrain>>,
    key_input: Res<ButtonInput<KeyCode>>,
    mut field_vis_state: ResMut<FieldVisState>,
) {
    // switch debug coloring
    if key_input.pressed(KeyCode::F1) {
        field_vis_state.field_type = FieldType::Reset;
    }
    if key_input.pressed(KeyCode::F2) {
        field_vis_state.field_type = FieldType::VegDensity;
    }

    if field_vis_state.field_type == FieldType::None {
        return;
    }

    //let now = std::time::Instant::now();
    let (mesh3d, surface) = terrain_query.single().unwrap();
    if let Some(mut mesh) = meshes.get_mut(&mesh3d.0) {
        match field_vis_state.field_type {
            FieldType::None => panic!(),
            FieldType::Reset => {
                reset_terrain_color(&mut mesh);
                field_vis_state.field_type = FieldType::None
            }
            FieldType::VegDensity => {
                set_terrain_color(&mut mesh, &surface.veg_density, Some((0.0, 1.0)))
            }
        };
        //    println!("{}", now.elapsed().as_secs_f64());
    }
}

pub fn picking_system(
    mut commands: Commands,
    grass_assets: Res<grass::GrassAssets>,
    mut ray_cast: MeshRayCast,
    terrain_query: Query<(), With<Terrain>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    window_query: Query<&Window>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut rng: Single<&mut WyRand, With<GlobalRng>>,
) {
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
        commands.spawn((
            Mesh3d(grass_assets.mesh.clone()),
            bevy::light::NotShadowCaster::default(),
            MeshMaterial3d(grass_assets.material.clone()),
            Transform::from_translation(hit.point - vec3(0.0, 0.1, 0.0))
                .with_scale(Vec3::ZERO)
                .with_rotation(Quat::from_axis_angle(
                    Vec3::new(0.0, 1.0, 0.0),
                    rng.random::<f32>() * 2.0 * PI,
                )),
            organism::Organism::default(),
        ));
    }
}

pub fn general_actions_system(
    key_input: Res<ButtonInput<KeyCode>>,
    mut time: ResMut<Time<Virtual>>,
) {
    let relative_speed = time.relative_speed();
    if key_input.just_pressed(KeyCode::ArrowUp) {
        time.set_relative_speed(relative_speed * 2.0);
    } else if key_input.just_pressed(KeyCode::ArrowDown) {
        time.set_relative_speed(relative_speed * 0.5);
    }
}
