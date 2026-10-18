//! The aim lab demo, as a plugin.
//!
//! The plugin owns the demo's state and reads the engine: the cursor and the buttons
//! come from `Engine::input`, which the window layer fills in, and the entities live
//! in `Engine::scene`.
//!
//! Five targets, a spawn box, a hit rule and the look sensitivity, plus a target
//! lifetime, a spawn cadence and a point total.
//!
//! The demo's assets are the repository's: `assets/models/Sphere.glb` for the
//! targets, `assets/models/Box.gltf` for the ground and `assets/audio/gunshot.wav`
//! for the shot. The 12 MB `pistol_animations_blender.glb` is deliberately not
//! copied, because no code path here names it.

mod rules;
mod scene;

use adam::{Engine, FrameCtx, Plugin, Result};
use glam::Vec3;
use meats::{Entity, Scene};
use pings::MouseButton;

pub use rules::{
    ray_sphere, spawn_position, Rng, Score, CAMERA_DISTANCE, CAMERA_FAR, CAMERA_FOV, CAMERA_NEAR,
    CAMERA_POSITION, CAMERA_YAW_DEGREES, HIT_POINTS, SPAWN_INTERVAL, SPAWN_X, SPAWN_Y, SPAWN_Z,
    STAGE_ENTITIES, TARGET_COUNT, TARGET_LIFETIME, TARGET_RADIUS,
};

/// The target and ground meshes, as virtual paths against the workspace asset root.
const TARGET_MESH: &str = "assets/models/Sphere.glb";
const GROUND_MESH: &str = "assets/models/Box.gltf";

/// The sound played on every shot.
const SHOT_SOUND: &str = "assets/audio/gunshot.wav";

/// The `physics_test.unhuman` floor: scale 20x1x20, top face at y = -1.5, which
/// is the bottom of the lowest target.
const GROUND_TRANSLATION: Vec3 = Vec3::new(0.0, -2.0, SPAWN_Z);
const GROUND_SCALE: Vec3 = Vec3::new(20.0, 1.0, 20.0);

/// 0.0024 radians per pixel, and the 200-pixel jump ignored because a re-captured
/// cursor warps.
const MOUSE_SENSITIVITY: f32 = 0.0024;
const MAX_MOUSE_DELTA: f32 = 200.0;
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 * 0.995;

struct Target {
    entity: Entity,
    life: f32,
}

/// The aim lab.
pub struct AimLabPlugin {
    score: Score,
    targets: Vec<Target>,
    spawned: u32,
    spawn_timer: f32,
    rng: Rng,
    yaw: f32,
    pitch: f32,
    mouse_was_down: bool,
    built: bool,
}

impl AimLabPlugin {
    pub fn new() -> Self {
        Self {
            score: Score::default(),
            targets: Vec::new(),
            spawned: 0,
            spawn_timer: SPAWN_INTERVAL,
            rng: Rng::default(),
            yaw: CAMERA_YAW_DEGREES.to_radians(),
            pitch: 0.0,
            mouse_was_down: false,
            built: false,
        }
    }

    /// Hits, misses, shots and points, for a status bar.
    pub fn score(&self) -> Score {
        self.score
    }

    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    pub fn target_entities(&self) -> Vec<Entity> {
        self.targets.iter().map(|target| target.entity).collect()
    }

    fn spawn_target(&mut self, eng: &mut Engine, life: f32) {
        let name = format!("Target_{}", self.spawned);
        self.spawned += 1;
        let position = spawn_position(&mut self.rng);
        let entity = scene::spawn(&mut eng.scene, &name, position, Vec3::splat(TARGET_RADIUS));
        scene::add_mesh(&mut eng.scene, entity, TARGET_MESH);
        self.targets.push(Target { entity, life });
    }

    /// The nearest target the crosshair is over, picked by a camera ray.
    fn pick(&self, scene: &Scene) -> Option<Entity> {
        let (origin, direction) = scene::camera_pose(scene)?;
        let mut nearest: Option<(f32, Entity)> = None;
        for target in &self.targets {
            let center = scene::translation(scene, target.entity);
            if let Some(distance) = ray_sphere(origin, direction, center, TARGET_RADIUS) {
                if nearest.is_none_or(|(best, _)| distance < best) {
                    nearest = Some((distance, target.entity));
                }
            }
        }
        nearest.map(|(_, entity)| entity)
    }

    fn expire_targets(&mut self, eng: &mut Engine, dt: f32) {
        let mut expired: Vec<Entity> = Vec::new();
        for target in &mut self.targets {
            target.life -= dt;
            if target.life <= 0.0 {
                expired.push(target.entity);
            }
        }
        for entity in expired {
            eng.scene.destroy(entity);
            self.targets.retain(|target| target.entity != entity);
            self.score.miss();
        }
    }

    /// Mouse look: yaw and pitch in radians.
    fn look(&mut self, eng: &mut Engine) {
        let (dx, dy) = eng.input.cursor_delta();
        if dx == 0.0 && dy == 0.0 {
            return;
        }
        if (dx * dx + dy * dy).sqrt() > MAX_MOUSE_DELTA {
            return;
        }
        // Yaw is about the camera's own Y axis, so `-= dx` turns right from any
        // starting yaw, including the wall-facing 180 the camera now starts at.
        self.yaw -= dx * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch + dy * MOUSE_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        scene::aim(
            &mut eng.scene,
            self.yaw.to_degrees(),
            self.pitch.to_degrees(),
        );
    }

    /// One shot per press. The button is held rather than pressed, because a frame can
    /// run several substeps and "pressed" stays true for all of them.
    fn shoot(&mut self, eng: &mut Engine) {
        let down = eng.input.is_mouse_held(MouseButton::Left);
        let fired = down && !self.mouse_was_down;
        self.mouse_was_down = down;
        if !fired {
            return;
        }

        self.score.shot();
        self.play_shot(eng);
        let Some(entity) = self.pick(&eng.scene) else {
            return;
        };

        self.score.hit();
        let position = spawn_position(&mut self.rng);
        scene::set_translation(&mut eng.scene, entity, position);
        if let Some(target) = self.targets.iter_mut().find(|t| t.entity == entity) {
            target.life = TARGET_LIFETIME;
        }
    }

    /// Play the shot as a one-shot, through the backend's `play` rather than a
    /// `playing` flag that `wire` would keep restarting; the clip and its gain come
    /// from the camera's sound component, which `resolve_assets` filled from the path.
    fn play_shot(&self, eng: &mut Engine) {
        let Some(camera) = eng.scene.primary_camera() else {
            return;
        };
        let Some((clip, gain)) = scene::audio(&eng.scene, camera) else {
            return;
        };
        if let Some(audio) = eng.audio.as_mut() {
            audio.play(&eng.assets, clip, gain);
        }
    }
}

impl Default for AimLabPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for AimLabPlugin {
    fn name(&self) -> &'static str {
        "aimlab"
    }

    fn build(&mut self, eng: &mut Engine) -> Result<()> {
        let ground = scene::spawn(&mut eng.scene, "Ground", GROUND_TRANSLATION, GROUND_SCALE);
        scene::add_mesh(&mut eng.scene, ground, GROUND_MESH);

        let light = scene::spawn(&mut eng.scene, "DirectionalLight", Vec3::ZERO, Vec3::ONE);
        scene::add_directional_light(&mut eng.scene, light);

        let camera = scene::spawn(&mut eng.scene, "Camera", CAMERA_POSITION, Vec3::ONE);
        scene::add_primary_camera(&mut eng.scene, camera, CAMERA_FOV, CAMERA_NEAR, CAMERA_FAR);
        // `look` only writes the rotation once the mouse moves, so the initial
        // yaw that turns -Z onto the wall has to be set here.
        scene::aim(&mut eng.scene, CAMERA_YAW_DEGREES, 0.0);

        // There is no gun entity, so the clip hangs on the camera: the ear and the
        // muzzle are the same point for a one-shot the listener takes.
        scene::add_audio_source(&mut eng.scene, camera, SHOT_SOUND);

        // Staggered by the cadence so the five do not expire in one tick.
        for index in 0..TARGET_COUNT {
            self.spawn_target(eng, TARGET_LIFETIME + index as f32 * SPAWN_INTERVAL);
        }
        // The first slot frees when target 0 expires, so the first spawn attempt
        // belongs on that same tick. Counting from `TARGET_LIFETIME` rather than
        // `SPAWN_INTERVAL` keeps the two countdowns on one `dt` sequence instead
        // of two that drift apart, which is what made the cadence non-deterministic.
        self.spawn_timer = TARGET_LIFETIME;
        self.built = true;

        // A `MeshRenderer` with a path and no handle is invisible to
        // `Scene::meshes`, so the paths are resolved here, once, against the
        // registry the bin built. A missing model is a warning.
        let Engine { scene, assets, .. } = eng;
        for error in scene.resolve_assets(assets) {
            log::warn!("aimlab: {error}");
        }

        log::info!(
            "aimlab: {} entities, {} targets",
            eng.scene.entities().len(),
            self.targets.len()
        );
        Ok(())
    }

    fn on_update(&mut self, eng: &mut Engine, ctx: &mut FrameCtx) {
        if !self.built {
            return;
        }
        let dt = ctx.step_seconds();

        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 {
            self.spawn_timer += SPAWN_INTERVAL;
            if self.targets.len() < TARGET_COUNT {
                self.spawn_target(eng, TARGET_LIFETIME);
            }
        }

        self.expire_targets(eng, dt);
        self.look(eng);
        self.shoot(eng);
    }
}
