use crate::domain;
use crate::grass;
use crate::{Surface, Terrain};
use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;

#[derive(Component, Default)]
pub struct Organism {
    age: f32, // [s]
}

const MAX_SIZE: f32 = 1.0;

pub fn update_organisms(
    time: Res<Time>,
    mut commands: Commands,
    mut surface_query: Query<&mut Surface>,
    mut organism_query: Query<(Entity, &mut Transform, &mut Organism)>,
) {
    let mut surface = surface_query.single_mut().unwrap();

    for (id, mut transform, mut organism) in organism_query.iter_mut() {
        organism.age += time.delta_secs();
        transform.scale = Vec3::ONE * organism.age.min(MAX_SIZE);

        // death
        if organism.age > grass::MAX_AGE {
            let p = transform.translation.xz();
            surface.veg_density.add_kernel(p, grass::SURFACE_AREA, -1.0);
            // delete entity
            commands.entity(id).despawn();
        }
    }
}

const SPAWN_PROP: f32 = 0.01;
const MIN_PROPAGATION_AGE: f32 = 2.0;

pub fn propagate_organisms(
    //    time: Res<Time>,
    mut commands: Commands,
    organism_query: Query<(&Transform, &Organism)>,
    terrain_query: Query<&Terrain>,
    mut surface_query: Query<&mut Surface>,
    mut rng: Single<&mut WyRand, With<GlobalRng>>,
    grass_assets: Res<crate::GrassAssets>,
) {
    let terrain = terrain_query.single().unwrap();
    let mut surface = surface_query.single_mut().unwrap();

    for (transform, organism) in organism_query.iter() {
        if organism.age < MIN_PROPAGATION_AGE {
            continue;
        }
        if rng.random::<f32>() >= SPAWN_PROP {
            continue;
        }

        let area = Circle::new(grass::SPAWN_RADIUS);
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
                terrain.height_map.get_bilinear(p) - grass::BELOW_SURFACE_DEPTH,
                p.y,
            ))
            .with_scale(Vec3::ZERO)
            .with_rotation(Quat::from_euler(
                EulerRot::XYZEx,
                (rng.random::<f32>() - 0.5) * PI * grass::ORIENTATION_MAX_ANGLE,
                rng.random::<f32>() * 2.0 * PI,
                0.0
            )),
            //    .with_rotation(Quat::from_axis_angle(axis, rng.random::<f32>() * 2.0 * PI)),
            Organism::default(),
        ));

        // add surface space usage
        surface.veg_density.add_kernel(p, grass::SURFACE_AREA, 1.0);
    }
}
