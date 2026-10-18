//! The aim lab's rules, with no scene and no input in them.
//!
//! The spawn box, the five targets, and the hit rule. Everything here is a pure
//! function of its arguments, so the rules can be tested without an ECS, a GPU or a
//! window.

use glam::Vec3;

/// The number of targets alive at once.
pub const TARGET_COUNT: usize = 5;

/// Seconds a target stays up before it counts as a miss.
pub const TARGET_LIFETIME: f32 = 6.0;

/// Seconds between spawns, while a target slot is free.
pub const SPAWN_INTERVAL: f32 = 1.0;

/// The pick radius, and the size `Sphere.glb` is drawn at.
pub const TARGET_RADIUS: f32 = 1.0;

/// Points a hit is worth.
pub const HIT_POINTS: u32 = 100;

/// The spawn box: x in `[-8, 8]`, y in `[-1, 5]`, z 10.
pub const SPAWN_X: (f32, f32) = (-8.0, 8.0);
pub const SPAWN_Y: (f32, f32) = (-1.0, 5.0);
pub const SPAWN_Z: f32 = 10.0;

/// The orbit radius: the camera sits this far from its focal point, the origin.
pub const CAMERA_DISTANCE: f32 = 5.0;

/// The camera's yaw, in degrees, so its local -Z turns onto +Z toward the wall.
pub const CAMERA_YAW_DEGREES: f32 = 180.0;

/// The focal point less `CAMERA_DISTANCE` along the view direction: `-5` on Z.
///
/// A camera looking down its own -Z at a focal point placed at the origin sits at
/// `focal - forward * distance`, so facing the wall at `z = 10` puts it at
/// `z = -5`. The wall is then `CAMERA_DISTANCE + SPAWN_Z = 15` away, and a target
/// on it reads `15..18` from the eye instead of the old 5.
pub const CAMERA_POSITION: Vec3 = Vec3::new(0.0, 0.0, -CAMERA_DISTANCE);

/// Ground, light and camera: everything `build` adds besides the targets.
pub const STAGE_ENTITIES: usize = 3;

/// Field of view in degrees, then near and far planes.
pub const CAMERA_FOV: f32 = 45.0;
pub const CAMERA_NEAR: f32 = 0.1;
pub const CAMERA_FAR: f32 = 1000.0;

/// Hits, misses, shots and points, for the plugin and for a status bar.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct Score {
    pub hits: u32,
    pub misses: u32,
    pub shots: u32,
    pub points: u32,
}

impl Score {
    pub fn hit(&mut self) {
        self.hits += 1;
        self.points += HIT_POINTS;
    }

    /// A target that ran out its lifetime unhit.
    pub fn miss(&mut self) {
        self.misses += 1;
    }

    pub fn shot(&mut self) {
        self.shots += 1;
    }

    /// Hits over shots, as a percentage. `0.0` before the first shot.
    pub fn accuracy(&self) -> f32 {
        if self.shots == 0 {
            0.0
        } else {
            self.hits as f32 / self.shots as f32 * 100.0
        }
    }
}

/// A 32-bit xorshift. A non-deterministic seed is exactly what a headless run cannot
/// have; a seed of its own makes a fixed step count reproduce.
pub struct Rng(u32);

impl Rng {
    pub const DEFAULT_SEED: u32 = 0x2545_F491;

    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { Self::DEFAULT_SEED } else { seed })
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// `[0.0, 1.0)`, from the top 24 bits so every value is exactly
    /// representable.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }
}

/// A position on the target wall. Two draws, x then y.
pub fn spawn_position(rng: &mut Rng) -> Vec3 {
    let x = SPAWN_X.0 + (SPAWN_X.1 - SPAWN_X.0) * rng.next_f32();
    let y = SPAWN_Y.0 + (SPAWN_Y.1 - SPAWN_Y.0) * rng.next_f32();
    Vec3::new(x, y, SPAWN_Z)
}

/// Distance along `direction` to a sphere, or `None`.
///
/// `direction` need not be normalised, and a ray starting inside the sphere returns
/// the point it leaves rather than a hit behind the eye.
pub fn ray_sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let to_origin = origin - center;
    let a = direction.length_squared();
    if a <= f32::EPSILON {
        return None;
    }
    let b = to_origin.dot(direction);
    let c = to_origin.length_squared() - radius * radius;
    let discriminant = b * b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let near = (-b - root) / a;
    if near >= 0.0 {
        return Some(near);
    }
    let far = (-b + root) / a;
    if far >= 0.0 {
        Some(far)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rng_repeats_for_a_seed() {
        let mut first = Rng::new(7);
        let mut second = Rng::new(7);
        for _ in 0..8 {
            assert_eq!(first.next_u32(), second.next_u32());
        }
        assert_ne!(Rng::new(1).next_u32(), Rng::new(2).next_u32());
    }

    #[test]
    fn a_spawn_position_stays_inside_the_cpp_box() {
        let mut rng = Rng::default();
        for _ in 0..64 {
            let position = spawn_position(&mut rng);
            assert!((SPAWN_X.0..=SPAWN_X.1).contains(&position.x), "{position}");
            assert!((SPAWN_Y.0..=SPAWN_Y.1).contains(&position.y), "{position}");
            assert_eq!(position.z, SPAWN_Z);
        }
    }

    #[test]
    fn next_f32_stays_in_the_unit_interval() {
        let mut rng = Rng::default();
        for _ in 0..256 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value), "{value}");
        }
    }
}
