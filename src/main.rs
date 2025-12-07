use bevy::light::CascadeShadowConfigBuilder;
use bevy::math::f32;
use bevy::prelude::*;
use bevy::render::render_resource::Face;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;

use crate::camera_controller::*;
use crate::grass::{GrassAssets, create_grass_material, create_grass_mesh};
use crate::terrain::*;

mod camera_controller;
mod domain;
mod grass;
mod organism;
mod terrain;

fn main() {
    App::new()
        .insert_resource(grass::GrassAssets::default())
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .add_plugins(DefaultPlugins)
        .add_plugins(CameraControllerPlugin)
        .add_plugins(EntropyPlugin::<WyRand>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, input_system)
        .add_systems(FixedUpdate, organism::update_organisms)
        .add_systems(FixedUpdate, organism::propagate_organisms)
        .run();
}
/*
fn setup_world(world: &mut World){
    world.insert_resource(grass::GrassAssets {
        mesh: Handle::,
        material: materials.add(Color::linear_rgb(0.0, 1.0, 0.0)),
    });
}*/

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut grass_assets: ResMut<grass::GrassAssets>,
) {
    grass_assets.mesh = meshes.add(create_grass_mesh(2)); //meshes.add(Cuboid::new(0.05, 0.5, 0.05));
    grass_assets.material = materials.add(create_grass_material());
    // circular base
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    // terrain
    let terrain_material = StandardMaterial {
        alpha_mode: AlphaMode::Opaque,
        double_sided: false,
        perceptual_roughness: 1.0,
        reflectance: 0.4,
        cull_mode: Some(Face::Back),
        flip_normal_map_y: true,
        ..default()
    };

    let terrain = Terrain::new(3);
    commands.spawn((
        //   Mesh3d(meshes.add(generate_terrain_mesh(domain::BOUNDS.min, domain::BOUNDS.size(), 512))),
        Mesh3d(meshes.add(generate_terrain_mesh(&terrain.height_map))),
        MeshMaterial3d(materials.add(terrain_material)),
        Transform::from_xyz(domain::HALF_SIZE.x as f32, 0.0, domain::HALF_SIZE.y as f32),
        terrain,
        Surface {
            veg_density: domain::Field::new(3),
        },
    ));

    // point light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // sun
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: false,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::from_rotation_x(-PI / 4.),
            ..default()
        },
        CascadeShadowConfigBuilder { ..default() }.build(),
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        CameraController::default(),
    ));
}

fn input_system(
    mut commands: Commands,
    grass_assets: Res<GrassAssets>,
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
