use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::*;
use rand::prelude::*;
use std::f32::consts::PI;
use crate::{Terrain, Surface};

#[derive(Component, Default)]
pub struct Organism {
    age: f32, // [s]
}

const MAX_SIZE: f32 = 1.0;
const MAX_AGE: f32 = 60.0;

pub fn update_organisms(
    time: Res<Time>,
    mut commands: Commands,
    mut organism_query: Query<(Entity, &mut Transform, &mut Organism)>,
) {
    for (id, mut transform, mut organism) in organism_query.iter_mut() {
        organism.age += time.delta_secs();
        transform.scale = Vec3::ONE * organism.age.min(MAX_SIZE);

        if organism.age > MAX_AGE {
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
        
        let area = Circle::new(1.0);
        let p = area.sample_interior(&mut rng) + transform.translation.xz();
        if surface.veg_density.get_nearest(p) > 0.5 {
            continue;
        }

        commands.spawn((
            Mesh3d(grass_assets.mesh.clone()),
            MeshMaterial3d(grass_assets.material.clone()),
            Transform::from_translation(
                Vec3::new(p.x, terrain.height_map.get_nearest(p) - 0.1, p.y),
            )
            .with_scale(Vec3::ZERO)
            .with_rotation(Quat::from_axis_angle(
                Vec3::new(0.0, 1.0, 0.0),
                rng.random::<f32>() * 2.0 * PI,
            )),
            Organism::default(),
        ));
        println!("{}", surface.veg_density.get_nearest(p));
        surface.veg_density.add_kernel(p, 0.125, 1.0);
        println!("{}", surface.veg_density.get_nearest(p));
    }
}
