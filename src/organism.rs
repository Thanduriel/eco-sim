use crate::domain;
use crate::parameters;
use crate::{Surface, Terrain};
use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;

#[derive(Component, Default)]
pub struct Organism {
    age: f32, // [s]
    surface_area: f32,
}

const MAX_SIZE: f32 = 1.0;

pub fn update_organisms_system(
    time: Res<Time>,
    mut commands: Commands,
    mut surface_query: Query<&mut Surface>,
    mut organism_query: Query<(Entity, &mut Transform, &mut Organism)>,
    general_params: Res<parameters::GeneralParameters>,
) {
    let mut surface = surface_query.single_mut().unwrap();

    for (id, mut transform, mut organism) in organism_query.iter_mut() {
        let dt = time.delta_secs();
        // still growing
        if organism.age < MAX_SIZE {
            let delta = dt.min(1.0 - organism.age);
            transform.scale = Vec3::ONE * (organism.age + delta);

            // add surface area usage
            let center = transform.translation.xz();
            let delta_area = delta * general_params.grass.surface_area;
            surface
                .veg_density
                .add_kernel(center, organism.surface_area, -1.0);
            organism.surface_area += delta_area;
            surface
                .veg_density
                .add_kernel(center, organism.surface_area, 1.0);
        }

        organism.age += time.delta_secs();

        // death
        if organism.age > general_params.grass.max_age {
            let p = transform.translation.xz();
            surface
                .veg_density
                .add_kernel(p, organism.surface_area, -1.0);
            // delete entity
            commands.entity(id).despawn();
        }
    }
}

const SPAWN_PROP: f32 = 0.01;
const MIN_PROPAGATION_AGE: f32 = 2.0;

pub fn propagate_organisms_system(
    //    time: Res<Time>,
    mut commands: Commands,
    organism_query: Query<(&Transform, &Organism)>,
    terrain_query: Query<&Terrain>,
    surface_query: Query<&Surface>,
    mut rng: Single<&mut WyRand, With<GlobalRng>>,
    grass_assets: Res<crate::GrassAssets>,
    general_params: Res<parameters::GeneralParameters>,
) {
    let terrain = terrain_query.single().unwrap();
    let surface = surface_query.single().unwrap();

    for (transform, organism) in organism_query.iter() {
        if organism.age < MIN_PROPAGATION_AGE {
            continue;
        }
        if rng.random::<f32>() >= SPAWN_PROP {
            continue;
        }

        let area = Circle::new(general_params.grass.spawn_radius);
        let p = area.sample_interior(&mut rng) + transform.translation.xz();
        if !domain::BOUNDS.contains(p) {
            continue;
        }
        if surface.veg_density.get_bilinear(p) > 0.5 {
            continue;
        }

        /*    let axis_circle = Circle::new(grass::ORIENTATION_MAX_RADIUS);
        let tip = axis_circle.sample_interior(&mut rng);
        let axis = Vec3::new(tip.x, 1.0, tip.y).normalize();*/

        commands.spawn((
            Mesh3d(grass_assets.mesh.clone()),
            bevy::light::NotShadowCaster::default(),
            MeshMaterial3d(grass_assets.material.clone()),
            Transform::from_translation(Vec3::new(
                p.x,
                terrain.height_map.get_bilinear(p) - general_params.grass.below_surface_depth,
                p.y,
            ))
            .with_scale(Vec3::ZERO)
            .with_rotation(Quat::from_euler(
                EulerRot::XYZEx,
                (rng.random::<f32>() - 0.5) * PI * general_params.grass.orientation_max_angle,
                rng.random::<f32>() * 2.0 * PI,
                0.0,
            )),
            //    .with_rotation(Quat::from_axis_angle(axis, rng.random::<f32>() * 2.0 * PI)),
            Organism::default(),
        ));
    }
}
