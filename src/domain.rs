use bevy::math::{FloatPow, USizeVec2, usizevec2};
use bevy::prelude::*;
use num_traits::{Bounded, NumAssign};
use std::ops::{Add, Index, IndexMut, Mul};

const SIZE_POW: USizeVec2 = USizeVec2 { x: 6, y: 6 };
pub const SIZE: USizeVec2 = USizeVec2 {
    x: 1 << (SIZE_POW.x),
    y: 1 << (SIZE_POW.y),
};
pub const HALF_SIZE: USizeVec2 = USizeVec2 {
    x: SIZE.x >> 1,
    y: SIZE.y >> 1,
};

pub const BOUNDS: Rect = Rect {
    min: Vec2 { x: 0.0, y: 0.0 },
    max: Vec2 {
        x: SIZE.x as f32,
        y: SIZE.y as f32,
    },
};

pub struct Field<T> {
    buffer: Vec<T>,
    //    subdivisions: i32,
    pub idx_scale: f32,
    pub size: USizeVec2,
}
/*
trait FieldType:
    Default + Copy + NumAssign {
}*/

impl<T: Default + Copy + NumAssign> Field<T> {
    pub fn new(subdivisions: i32) -> Self {
        let size = USizeVec2::new(
            1 << (SIZE_POW.x as i32 + subdivisions) as usize,
            1 << (SIZE_POW.y as i32 + subdivisions) as usize,
        );

        Field {
            buffer: vec![T::default(); size.x * size.y],
            //       subdivisions: subdivisions,
            idx_scale: 2.0_f32.powf(subdivisions as f32),
            size: size,
        }
    }

    fn clamp_index(&self, idx: USizeVec2) -> USizeVec2 {
        idx.clamp(USizeVec2::ZERO, self.size - USizeVec2::ONE)
    }

    fn flat_index(&self, idx: USizeVec2) -> usize {
        idx.x + idx.y * self.size.x
    }

    /*fn flat_index(&self, idx : [usize; 2]) -> usize {
        idx[0] + idx[1] * self.size.x
    }*/

    #[allow(dead_code)]
    pub fn get_nearest(&self, pos: Vec2) -> T {
        let idx = self.clamp_index((pos * self.idx_scale).round().as_usizevec2());
        self.buffer[self.flat_index(idx)]
    }
}

impl<T: Default + Copy + NumAssign + Mul<f32, Output = T> + Add<T, Output = T>> Field<T> {
    pub fn get_bilinear(&self, pos: Vec2) -> T {
        let pos_scaled = pos * self.idx_scale;
        let t = pos_scaled.fract();
        let lower_idx = self.clamp_index(pos_scaled.floor().as_usizevec2());
        let upper_idx = self.clamp_index(pos_scaled.ceil().as_usizevec2());
        let v00 = self.buffer[self.flat_index(lower_idx)];
        let v10 = self.buffer[self.flat_index(usizevec2(upper_idx.x, lower_idx.y))];
        let v01 = self.buffer[self.flat_index(usizevec2(lower_idx.x, upper_idx.y))];
        let v11 = self.buffer[self.flat_index(upper_idx)];

        // interpolate along x
        let v0 = v00 * (1.0 - t.x) + v10 * t.x;
        let v1 = v01 * (1.0 - t.x) + v11 * t.x;

        // interpolate along y
        return v0 * (1.0 - t.y) + v1 * t.y;
    }
}

impl<T: Copy + Bounded + std::cmp::PartialOrd> Field<T> {
    pub fn compute_min_max(&self) -> (T, T) {
        let min = self.buffer.iter().fold(T::max_value(), |a, &b| {
            match PartialOrd::partial_cmp(&a, &b) {
                None => a,
                Some(std::cmp::Ordering::Less) => a,
                Some(_) => b,
            }
        });
        let max = self.buffer.iter().fold(T::min_value(), |a, &b| {
            match PartialOrd::partial_cmp(&a, &b) {
                None => a,
                Some(std::cmp::Ordering::Less) => b,
                Some(_) => a,
            }
        });
        (min, max)
    }
}

impl<T: Default + Copy + NumAssign + Mul<f32, Output = T>> Field<T> {
    pub fn add_kernel(&mut self, pos: Vec2, radius: f32, value: T) {
        let pos_local = pos * Vec2::splat(self.idx_scale);
        let r_local = radius * self.idx_scale;
        let min_pos_local = pos_local - Vec2::splat(r_local);
        let max_pos_local = pos_local + Vec2::splat(r_local);
        let min_idx = self.clamp_index((min_pos_local).floor().as_usizevec2());
        let max_idx = self.clamp_index((max_pos_local).ceil().as_usizevec2());

        let r_sq = r_local.squared();
        for iy in min_idx.y..max_idx.y {
            for ix in min_idx.x..max_idx.x {
                let d_sq = pos_local.distance_squared(vec2(ix as f32, iy as f32));
                if d_sq < r_sq {
                    let flat_idx = self.flat_index(USizeVec2::new(ix, iy));
                    self.buffer[flat_idx] += value * (1.0 - d_sq / r_sq);
                }
            }
        }
    }
}

impl<T: Default + Copy + NumAssign> Index<[usize; 2]> for Field<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &T {
        &self.buffer[self.flat_index(index.into())]
    }
}

impl<T: Default + Copy + NumAssign> IndexMut<[usize; 2]> for Field<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut T {
        let flat_idx = self.flat_index(index.into());
        &mut self.buffer[flat_idx]
    }
}
