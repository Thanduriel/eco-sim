use bevy::camera;
use bevy::light;
use bevy::math::f32;
use bevy::pbr;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::render_resource::Face;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use std::f32::consts::PI;

use bevy_egui::{
    EguiPlugin, EguiPrimaryContextPass,
    input::{egui_wants_any_keyboard_input, egui_wants_any_pointer_input},
};

use crate::camera_controller::*;
use crate::grass::{GrassAssets, create_grass_material, create_grass_mesh};
use crate::terrain::*;

mod camera_controller;
mod color_map;
mod domain;
mod grass;
mod hud;
mod organism;
mod parameters;
mod player_inputs;
mod terrain;

fn main() {
    App::new()
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(CameraControllerPlugin)
        .add_plugins(EntropyPlugin::<WyRand>::default())
        .insert_resource(grass::GrassAssets::default())
        .add_plugins(MaterialPlugin::<grass::GrassMaterial>::default())
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .insert_resource(player_inputs::FieldVisState::default())
        .insert_resource(parameters::GeneralParameters::default())
        .add_systems(EguiPrimaryContextPass, parameters::parameter_ui_system)
        //      .add_plugins(ScreenSpaceAmbientOcclusionPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, day_night_cycle)
        .add_systems(
            Update,
            player_inputs::picking_system
                .run_if(not(egui_wants_any_keyboard_input).and(not(egui_wants_any_pointer_input))),
        )
        .add_systems(Update, player_inputs::vis_fields_system)
        .add_systems(Update, hud::hud_system)
        .add_systems(Update, player_inputs::general_actions_system)
        .add_systems(FixedUpdate, organism::update_organisms_system)
        .add_systems(FixedUpdate, organism::propagate_organisms_system)
        .run();
}
/*
fn setup_world(world: &mut World){
    world.insert_resource(grass::GrassAssets {
        mesh: Handle::,
        material: materials.add(Color::linear_rgb(0.0, 1.0, 0.0)),
    });
}*/

/// set scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ext_materials: ResMut<Assets<grass::GrassMaterial>>,
    mut grass_assets: ResMut<grass::GrassAssets>,
    mut scattering_mediums: ResMut<Assets<pbr::ScatteringMedium>>,
) {
    grass_assets.mesh = meshes.add(create_grass_mesh(4, 0.15));
    grass_assets.material = ext_materials.add(create_grass_material());
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
        //   alpha_mode: AlphaMode::Opaque,
        double_sided: false,
        perceptual_roughness: 1.0,
        reflectance: 0.4,
        cull_mode: Some(Face::Back),
        //    flip_normal_map_y: true,
        ..default()
    };

    let terrain = Terrain::new(3);
    commands.spawn((
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
            illuminance: light_consts::lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        light::VolumetricLight,
        Transform {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::from_rotation_x(-PI / 4.),
            ..default()
        },
        light::CascadeShadowConfigBuilder { ..default() }.build(),
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        CameraController::default(),
        Msaa::Off,
        pbr::ScreenSpaceAmbientOcclusion {
            quality_level: pbr::ScreenSpaceAmbientOcclusionQualityLevel::High,
            constant_object_thickness: 4.0,
        },
        // Earthlike atmosphere
        pbr::Atmosphere::earthlike(scattering_mediums.add(pbr::ScatteringMedium::default())),
        // Can be adjusted to change the scene scale and rendering quality
        pbr::AtmosphereSettings::default(),
        // The directional light illuminance used in this scene
        // (the one recommended for use with this feature) is
        // quite bright, so raising the exposure compensation helps
        // bring the scene to a nicer brightness range.
        camera::Exposure { ev100: 13.0 },
        // Tonemapper chosen just because it looked good with the scene, any
        // tonemapper would be fine :)
        // Tonemapping::AcesFitted,
        // Bloom gives the sun a much more natural look.
        Bloom::NATURAL,
        // Enables the atmosphere to drive reflections and ambient lighting (IBL) for this view
        light::AtmosphereEnvironmentMapLight::default(),
        light::VolumetricFog {
            ambient_intensity: 0.0,
            ..default()
        },
    ));

    // spawn the fog volume
    commands.spawn((
        light::FogVolume::default(),
        Transform::from_scale(Vec3::new(10.0, 1.0, 10.0)).with_translation(Vec3::Y * 0.5),
    ));

    // game speed indicator
    commands.spawn((
        Text::new("game speed: 0.0"),
        TextLayout::new_with_justify(Justify::Right),
        Node {
            position_type: PositionType::Absolute,
            top: px(5),
            left: px(5),
            ..default()
        },
    ));
}



fn day_night_cycle(
    mut suns: Query<&mut Transform, With<DirectionalLight>>,
    time: Res<Time>,
    params: Res<parameters::GeneralParameters>,
) {
    suns.iter_mut()
        .for_each(|mut tf| tf.rotate_x(-time.delta_secs() * 2.0 * PI / params.sun.day_duration));
}
