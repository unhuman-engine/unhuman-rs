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

/// Rounds in a magazine.
pub const MAX_AMMO: u32 = 7;

/// Seconds a reload takes.
///
/// The C++ reads this from the reload clip's duration; there is no clip on this
/// path, so the C++'s own documented default is the value.
pub const RELOAD_DURATION: f32 = 3.5;

/// Seconds a recoil kick takes to decay to zero, matching the 0.5 s fire
/// animation. No clip drives it here, so the duration is explicit.
pub const RECOIL_DURATION: f32 = 0.5;

/// The per-shot kick in degrees. The reference kick is pitch-only; the yaw
/// channel stays in the state so a caller can add horizontal kick later.
pub const RECOIL_PITCH_DEGREES: f32 = 6.0;
pub const RECOIL_YAW_DEGREES: f32 = 0.0;

/// The most either recoil axis may accumulate under rapid fire.
pub const RECOIL_MAX_DEGREES: f32 = 12.0;

/// The gun's scale and its offset in camera-local metres: right, up, back.
pub const GUN_SCALE: f32 = 0.05;
pub const GUN_OFFSET: Vec3 = Vec3::new(0.46, 0.05, -0.65);

/// The gun model's own facing correction, in degrees.
pub const GUN_ROTATION_DEGREES: Vec3 = Vec3::new(-1.0, -178.0, -1.0);

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

/// Ground, light, camera and the gun seam: everything `build` adds besides the
/// targets. The gun is an entity even when its mesh is absent, so the count is
/// the same whether or not the pistol asset is present.
pub const STAGE_ENTITIES: usize = 4;

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

/// Magazine state: rounds left and the reload clock.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ammo {
    pub rounds: u32,
    pub reloading: bool,
    /// Seconds left on the reload; zero when not reloading.
    pub timer: f32,
}

impl Default for Ammo {
    fn default() -> Self {
        Self {
            rounds: MAX_AMMO,
            reloading: false,
            timer: 0.0,
        }
    }
}

impl Ammo {
    /// Whether a shot can be taken now.
    pub fn can_fire(&self) -> bool {
        self.rounds > 0 && !self.reloading
    }

    /// Spends a round, returning whether one was spent.
    pub fn fire(&mut self) -> bool {
        if self.rounds == 0 {
            return false;
        }
        self.rounds -= 1;
        true
    }

    /// Starts a reload if the magazine is not full and none is running.
    pub fn begin_reload(&mut self) -> bool {
        if self.reloading || self.rounds >= MAX_AMMO {
            return false;
        }
        self.reloading = true;
        self.timer = RELOAD_DURATION;
        true
    }

    /// Advances the reload clock; a finished reload refills the magazine.
    pub fn tick(&mut self, dt: f32) {
        if !self.reloading {
            return;
        }
        self.timer -= dt;
        if self.timer <= 0.0 {
            self.reloading = false;
            self.timer = 0.0;
            self.rounds = MAX_AMMO;
        }
    }

    /// Fraction of the reload done, `0.0..=1.0`.
    pub fn reload_progress(&self) -> f32 {
        if !self.reloading {
            return 0.0;
        }
        (1.0 - self.timer / RELOAD_DURATION).clamp(0.0, 1.0)
    }
}

/// A pitch/yaw camera kick in degrees, decaying to zero over
/// [`RECOIL_DURATION`].
///
/// Kept apart from the mouse aim so a kick composes with the look instead of
/// overwriting it: the camera rotates by the base aim plus this offset.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Recoil {
    /// Positive pitch looks up; positive yaw follows the scene's yaw convention.
    pub pitch: f32,
    pub yaw: f32,
    remaining: f32,
}

impl Recoil {
    /// Adds a kick and restarts the decay clock. Each axis is clamped to
    /// [`RECOIL_MAX_DEGREES`], so sustained fire cannot exceed it.
    pub fn kick(&mut self, pitch_degrees: f32, yaw_degrees: f32) {
        self.pitch = (self.pitch + pitch_degrees).clamp(-RECOIL_MAX_DEGREES, RECOIL_MAX_DEGREES);
        self.yaw = (self.yaw + yaw_degrees).clamp(-RECOIL_MAX_DEGREES, RECOIL_MAX_DEGREES);
        self.remaining = RECOIL_DURATION;
    }

    /// Advances the decay. The per-step ratio telescopes, so any step
    /// partition reaches zero at the same time.
    pub fn tick(&mut self, dt: f32) {
        if self.remaining <= 0.0 {
            return;
        }
        let before = self.remaining;
        self.remaining = (self.remaining - dt).max(0.0);
        if self.remaining <= 0.0 {
            self.pitch = 0.0;
            self.yaw = 0.0;
        } else {
            let scale = self.remaining / before;
            self.pitch *= scale;
            self.yaw *= scale;
        }
    }

    /// Whether a kick is still decaying.
    pub fn is_active(&self) -> bool {
        self.remaining > 0.0
    }
}

/// Crosshair geometry in points, drawn at the centre of the viewport.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Crosshair {
    pub half_length: f32,
    pub gap: f32,
    pub thickness: f32,
    pub dot_radius: f32,
    /// RGBA, matching the aim-lab green and its outline.
    pub color: [u8; 4],
    pub outline: [u8; 4],
}

impl Default for Crosshair {
    fn default() -> Self {
        Self {
            half_length: 12.0,
            gap: 4.0,
            thickness: 2.0,
            dot_radius: 2.0,
            color: [0, 255, 70, 230],
            outline: [0, 0, 0, 180],
        }
    }
}

/// What a HUD draws. The demo hands this out rather than reaching into a
/// renderer, so the HUD is the sandbox's or the engine's to build.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Report {
    pub score: Score,
    pub ammo: Ammo,
}

impl Report {
    /// Hits over shots, as a percentage; `0.0` before the first shot.
    pub fn accuracy(&self) -> f32 {
        self.score.accuracy()
    }

    /// A screen-centre marker has no per-frame state, so the crosshair is a
    /// constant. A HUD places it at half the viewport and draws these arms.
    pub const CROSSHAIR: Crosshair = Crosshair {
        half_length: 12.0,
        gap: 4.0,
        thickness: 2.0,
        dot_radius: 2.0,
        color: [0, 255, 70, 230],
        outline: [0, 0, 0, 180],
    };

    /// The controls this demo honours, for a HUD to list.
    ///
    /// Cursor release/lock needs the window layer, and orbit and zoom belong to
    /// the editor camera, so neither belongs in this demo's list.
    pub const CONTROLS: [&'static str; 3] = ["Mouse: Aim", "Left Click: Shoot", "R: Reload"];
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
    fn an_empty_magazine_refuses_a_shot() {
        let mut ammo = Ammo {
            rounds: 0,
            ..Ammo::default()
        };
        assert!(!ammo.can_fire());
        assert!(!ammo.fire());
        assert_eq!(ammo.rounds, 0);
    }

    #[test]
    fn a_reload_refills_after_the_timer() {
        let mut ammo = Ammo::default();
        ammo.fire();
        ammo.fire();
        assert_eq!(ammo.rounds, MAX_AMMO - 2);

        assert!(ammo.begin_reload());
        assert!(ammo.reloading);
        assert!(!ammo.can_fire(), "no firing while reloading");

        ammo.tick(RELOAD_DURATION - 0.1);
        assert!(ammo.reloading, "still reloading before the timer");
        assert_eq!(ammo.rounds, MAX_AMMO - 2);

        ammo.tick(0.2);
        assert!(!ammo.reloading);
        assert_eq!(ammo.rounds, MAX_AMMO);
        assert!(ammo.can_fire());
    }

    #[test]
    fn a_reload_at_full_does_nothing() {
        let mut ammo = Ammo::default();
        assert!(!ammo.begin_reload());
        assert!(!ammo.reloading);
    }

    #[test]
    fn a_second_reload_does_not_restart_the_first() {
        let mut ammo = Ammo::default();
        ammo.fire();
        assert!(ammo.begin_reload());
        ammo.tick(1.0);

        let timer = ammo.timer;
        assert!(!ammo.begin_reload());
        assert_eq!(ammo.timer, timer, "the first reload is not restarted");
    }

    #[test]
    fn reload_progress_runs_from_zero_to_one() {
        let mut ammo = Ammo::default();
        ammo.fire();
        ammo.begin_reload();
        assert_eq!(ammo.reload_progress(), 0.0);

        ammo.tick(RELOAD_DURATION / 2.0);
        assert!((ammo.reload_progress() - 0.5).abs() < 1e-4);

        ammo.tick(RELOAD_DURATION);
        assert_eq!(ammo.reload_progress(), 0.0, "zero once the reload is done");
    }

    #[test]
    fn report_accuracy_is_hits_over_shots() {
        let mut report = Report::default();
        report.score.shot();
        report.score.shot();
        report.score.hit();

        assert_eq!(report.accuracy(), 50.0);
        assert_eq!(report.ammo.rounds, MAX_AMMO);
        assert_eq!(Report::CONTROLS.len(), 3);
    }

    #[test]
    fn next_f32_stays_in_the_unit_interval() {
        let mut rng = Rng::default();
        for _ in 0..256 {
            let value = rng.next_f32();
            assert!((0.0..1.0).contains(&value), "{value}");
        }
    }

    #[test]
    fn recoil_decays_to_zero_over_its_duration() {
        let mut recoil = Recoil::default();
        recoil.kick(RECOIL_PITCH_DEGREES, 1.0);
        assert_eq!(recoil.pitch, RECOIL_PITCH_DEGREES);
        assert_eq!(recoil.yaw, 1.0);
        assert!(recoil.is_active());

        recoil.tick(RECOIL_DURATION / 2.0);
        assert!((recoil.pitch - RECOIL_PITCH_DEGREES / 2.0).abs() < 1e-4);
        assert!((recoil.yaw - 0.5).abs() < 1e-4);

        recoil.tick(RECOIL_DURATION / 2.0);
        assert_eq!(recoil.pitch, 0.0);
        assert_eq!(recoil.yaw, 0.0);
        assert!(!recoil.is_active());
    }

    #[test]
    fn rapid_kicks_are_capped() {
        let mut recoil = Recoil::default();
        for _ in 0..8 {
            recoil.kick(RECOIL_PITCH_DEGREES, RECOIL_YAW_DEGREES);
        }
        assert_eq!(recoil.pitch, RECOIL_MAX_DEGREES);
        assert_eq!(recoil.yaw, RECOIL_YAW_DEGREES);

        recoil.kick(-RECOIL_MAX_DEGREES * 2.0, 0.0);
        assert_eq!(recoil.pitch, -RECOIL_MAX_DEGREES);
    }

    #[test]
    fn a_fresh_kick_restarts_the_decay() {
        let mut recoil = Recoil::default();
        recoil.kick(RECOIL_PITCH_DEGREES, 0.0);
        recoil.tick(RECOIL_DURATION * 0.75);
        assert!(recoil.is_active());

        recoil.kick(0.0, 0.0);
        recoil.tick(RECOIL_DURATION * 0.5);
        assert!(recoil.is_active(), "the timer restarted");

        recoil.tick(RECOIL_DURATION * 0.5);
        assert_eq!(recoil.pitch, 0.0);
    }
}
