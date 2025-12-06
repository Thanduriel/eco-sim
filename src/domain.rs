use bevy::prelude::*;
use std::ops::{Index, IndexMut};

const SIZE_POW: UVec2 = UVec2 { x: 6, y: 6 };
pub const SIZE: UVec2 = UVec2 {
    x: 1 << (SIZE_POW.x),
    y: 1 << (SIZE_POW.y),
};
pub const HALF_SIZE: UVec2 = UVec2 {
    x: SIZE.x >> 1,
    y: SIZE.y >> 1,
};
/*
pub const BOUNDS: Rect = Rect {
    min: Vec2 {
        x: -(HALF_SIZE.x as f32),
        y: -(HALF_SIZE.y as f32),
    },
    max: Vec2 {
        x: HALF_SIZE.x as f32,
        y: HALF_SIZE.y as f32,
    },
};*/

pub struct Field<T> {
    buffer: Vec<T>,
//    subdivisions: i32,
    pub idx_scale: f32,
    pub size: UVec2,
}

impl<T: Default + Copy> Field<T> {
    pub fn new(subdivisions: i32) -> Self {
        let size = UVec2::new(
            1 << (SIZE_POW.x as i32 + subdivisions) as u32,
            1 << (SIZE_POW.y as i32 + subdivisions) as u32,
        );

        Field {
            buffer: vec![T::default(); size.x as usize * size.y as usize],
     //       subdivisions: subdivisions,
            idx_scale: 2.0_f32.powf(subdivisions as f32),
            size: size,
        }
    }

    pub fn get_nearest(&self, pos : Vec2) -> T {
        let x = ((pos.x * self.idx_scale).round() as usize).clamp(0, self.size.x as usize - 1);
        let y = ((pos.y * self.idx_scale).round() as usize).clamp(0, self.size.y as usize - 1);
        self.buffer[x + y * self.size.x as usize]
    }
}

impl<T> Index<(u32, u32)> for Field<T> {
    type Output = T;

    fn index(&self, index: (u32, u32)) -> &T {
        let (x,y) = index;
        &self.buffer[x as usize + y as usize * self.size.x as usize]
    }
}

impl<T> IndexMut<(u32, u32)> for Field<T> {
    fn index_mut(&mut self, index: (u32, u32)) -> &mut T {
        let (x,y) = index;
        &mut self.buffer[x as usize + y as usize * self.size.x as usize]
    }
}