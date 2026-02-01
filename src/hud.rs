use bevy::prelude::*;

pub fn hud_system(mut game_speed_query: Query<&mut Text>, time: Res<Time<Virtual>>) {
    let mut game_speed_text = game_speed_query.single_mut().unwrap();
    **game_speed_text = format!("game speed: {}x", time.relative_speed());
}
