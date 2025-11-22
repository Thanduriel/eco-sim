use bevy::prelude::*;

#[derive(Component, Default)]
pub struct Organism {
    age: f32,
}

pub fn update_organisms(
    time: Res<Time>,
    mut commands: Commands,
    mut organism_query: Query<(Entity, &mut Transform, &mut Organism)>,
) {
    for (id, mut transform, mut organism) in organism_query.iter_mut() {
        organism.age += time.delta_secs();
        transform.scale = Vec3::ONE * organism.age;

		if organism.age > 10.0 {
			commands.entity(id).despawn();
		}
    }
}
