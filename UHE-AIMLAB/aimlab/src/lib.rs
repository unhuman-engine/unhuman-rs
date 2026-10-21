//! The aim lab demo, as a plugin.
//!
//! The plugin owns the demo's state and reads the engine: the cursor and the buttons
//! come from `Engine::input`, which the window layer fills in, and the entities live
//! in `Engine::scene`.
//!
//! Five targets, a spawn box, a hit rule and the look sensitivity, plus a target
//! lifetime, a spawn cadence, a point total, and a seven-round magazine with an
//! `R` reload and an auto-reload at zero.
//!
//! The demo's assets are the repository's: `assets/models/Sphere.glb` for the
//! targets, `assets/models/Box.gltf` for the ground and `assets/audio/gunshot.wav`
//! for the shot.
//!
//! The animated pistol is a camera child. `bifrost` attaches its skin and
//! skeleton when the entity carries an `Animator`, so the idle clip plays over
//! the same transform seam the other meshes use. A checkout without the asset
//! leaves the transform only and changes nothing else.
//!
//! Displayed state goes through [`AimLabPlugin::report`]; the demo owns no HUD.

mod rules;
mod scene;

use adam::{Engine, FrameCtx, Plugin, Result};
use glam::Vec3;
use meats::{Entity, MeshRenderer, Scene};
use pings::{KeyCode, MouseButton};

pub use rules::{
    ray_sphere, spawn_position, Ammo, Crosshair, Recoil, Report, Rng, Score, CAMERA_DISTANCE,
    CAMERA_FAR, CAMERA_FOV, CAMERA_NEAR, CAMERA_POSITION, CAMERA_YAW_DEGREES, GUN_OFFSET,
    GUN_ROTATION_DEGREES, GUN_SCALE, HIT_POINTS, MAX_AMMO, RECOIL_DURATION, RECOIL_MAX_DEGREES,
    RECOIL_PITCH_DEGREES, RECOIL_YAW_DEGREES, RELOAD_DURATION, SPAWN_INTERVAL, SPAWN_X, SPAWN_Y,
    SPAWN_Z, STAGE_ENTITIES, TARGET_COUNT, TARGET_LIFETIME, TARGET_RADIUS,
};

/// The target and ground meshes, as virtual paths against the workspace asset root.
const TARGET_MESH: &str = "assets/models/Sphere.glb";
const GROUND_MESH: &str = "assets/models/Box.gltf";

/// The animated pistol, attached to the gun seam when the checkout has it.
const GUN_MESH: &str = "assets/models/pistol_animations_blender.glb";

/// The pistol rig's clips. It has no plain `Reload`, so the full reload is
/// `Reload_Complete`; `Reload_Fast` is the shorter variant.
pub const GUN_IDLE_CLIP: &str = "Idle";
pub const GUN_FIRE_CLIP: &str = "Fire";
pub const GUN_RELOAD_CLIP: &str = "Reload_Complete";

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
    ammo: Ammo,
    targets: Vec<Target>,
    spawned: u32,
    spawn_timer: f32,
    rng: Rng,
    yaw: f32,
    pitch: f32,
    recoil: Recoil,
    mouse_was_down: bool,
    r_was_down: bool,
    gun: Option<Entity>,
    /// Seconds left on the one-shot fire clip.
    fire_timer: f32,
    /// The fire clip's own duration; `RECOIL_DURATION` until the asset supplies it.
    fire_duration: f32,
    built: bool,
}

impl AimLabPlugin {
    pub fn new() -> Self {
        Self {
            score: Score::default(),
            ammo: Ammo::default(),
            targets: Vec::new(),
            spawned: 0,
            spawn_timer: SPAWN_INTERVAL,
            rng: Rng::default(),
            yaw: CAMERA_YAW_DEGREES.to_radians(),
            pitch: 0.0,
            recoil: Recoil::default(),
            mouse_was_down: false,
            r_was_down: false,
            gun: None,
            fire_timer: 0.0,
            fire_duration: RECOIL_DURATION,
            built: false,
        }
    }

    /// Hits, misses, shots and points, for a status bar.
    pub fn score(&self) -> Score {
        self.score
    }

    /// Rounds left and the reload clock.
    pub fn ammo(&self) -> Ammo {
        self.ammo
    }

    /// Everything a HUD draws: score, ammo and the controls list. The display
    /// seam; the demo never draws.
    pub fn report(&self) -> Report {
        Report {
            score: self.score,
            ammo: self.ammo,
        }
    }

    /// The gun entity, present even when its mesh is not.
    pub fn gun(&self) -> Option<Entity> {
        self.gun
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
    fn look(&mut self, eng: &Engine) {
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
    }

    /// Writes base aim plus the recoil offset once per frame, so a kick never
    /// overwrites the mouse look and a still mouse still lets the kick decay.
    fn aim(&self, eng: &mut Engine) {
        let pitch = (self.pitch + self.recoil.pitch.to_radians()).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        scene::aim(
            &mut eng.scene,
            self.yaw.to_degrees() + self.recoil.yaw,
            pitch.to_degrees(),
        );
    }

    /// One shot per press, spent from the magazine. The button is held rather
    /// than pressed, because a frame can run several substeps and "pressed"
    /// stays true for all of them.
    fn shoot(&mut self, eng: &mut Engine) {
        let down = eng.input.is_mouse_held(MouseButton::Left);
        let fired = down && !self.mouse_was_down;
        self.mouse_was_down = down;
        // An empty or reloading magazine eats the click, exactly as the C++
        // consumes the press before it checks the ammo.
        if !fired || !self.ammo.can_fire() {
            return;
        }

        self.ammo.fire();
        self.score.shot();
        self.recoil.kick(RECOIL_PITCH_DEGREES, RECOIL_YAW_DEGREES);
        // The model resolves on a worker after `build`, so the clip's duration
        // is not known then. Read it on the shot, when the asset is normally
        // ready, and fall back while it still is not.
        if let Some(gun) = self.gun {
            self.fire_duration = fire_clip_duration(&eng.scene, &eng.assets, gun);
        }
        self.fire_timer = self.fire_duration;
        self.play_shot(eng);
        if let Some(entity) = self.pick(&eng.scene) {
            self.score.hit();
            let position = spawn_position(&mut self.rng);
            scene::set_translation(&mut eng.scene, entity, position);
            if let Some(target) = self.targets.iter_mut().find(|t| t.entity == entity) {
                target.life = TARGET_LIFETIME;
            }
        }

        // Auto-reload on the shot that empties the magazine.
        if self.ammo.rounds == 0 {
            self.ammo.begin_reload();
        }
    }

    /// `R` starts a reload, one edge per press.
    fn reload(&mut self, eng: &mut Engine) {
        let down = eng.input.is_key_held(KeyCode::R);
        let pressed = down && !self.r_was_down;
        self.r_was_down = down;
        if pressed {
            self.ammo.begin_reload();
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

    /// Drives the gun's clip from the ammo state: reload wins over the one-shot
    /// fire, and idle is the resting state.
    fn animate_gun(&mut self, eng: &mut Engine, dt: f32) {
        let Some(gun) = self.gun else {
            return;
        };
        if self.fire_timer > 0.0 {
            self.fire_timer = (self.fire_timer - dt).max(0.0);
        }
        let (clip, looping) = if self.ammo.reloading {
            (GUN_RELOAD_CLIP, true)
        } else if self.fire_timer > 0.0 {
            (GUN_FIRE_CLIP, false)
        } else {
            (GUN_IDLE_CLIP, true)
        };
        scene::play_clip(&mut eng.scene, gun, clip, looping);
    }
}

/// The Fire clip's duration from the loaded pistol, so the state does not
/// duplicate it. `RECOIL_DURATION` is the same 0.5 s when the asset is absent.
fn fire_clip_duration(scene: &Scene, assets: &vault::Assets, gun: Entity) -> f32 {
    scene
        .get(gun)
        .and_then(|entry| entry.get::<MeshRenderer>())
        .and_then(|renderer| renderer.mesh)
        .and_then(|mesh| assets.meshes(mesh))
        .and_then(|meshes| {
            meshes
                .iter()
                .flat_map(|mesh| mesh.animations.iter())
                .find(|clip| clip.name == GUN_FIRE_CLIP)
        })
        .map(|clip| clip.duration)
        .filter(|duration| *duration > 0.0)
        .unwrap_or(RECOIL_DURATION)
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

        // The gun is a camera child, so its world transform follows the look
        // without a per-frame update of its own. The mesh is attached only when
        // the pistol is present; the animator stays either way, so ammo, reload
        // and a test can read the clip state without the asset.
        let gun = scene::spawn(&mut eng.scene, "Gun", GUN_OFFSET, Vec3::splat(GUN_SCALE));
        scene::set_rotation(&mut eng.scene, gun, GUN_ROTATION_DEGREES);
        eng.scene.set_parent(gun, Some(camera));
        scene::add_animator(&mut eng.scene, gun);
        if eng.assets.vfs().resolve(GUN_MESH).is_ok() {
            scene::add_mesh(&mut eng.scene, gun, GUN_MESH);
        } else {
            log::info!("aimlab: {GUN_MESH} absent; the gun is a transform-only seam");
        }
        self.gun = Some(gun);

        // There is no gun entity with a sound, so the clip hangs on the camera:
        // the ear and the muzzle are the same point for a one-shot the listener
        // takes.
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
        let Engine {
            scene,
            assets,
            jobs,
            ..
        } = eng;
        for error in scene.resolve_assets_async(jobs, assets) {
            log::warn!("aimlab: {error}");
        }
        self.fire_duration = fire_clip_duration(scene, assets, gun);

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

        self.ammo.tick(dt);
        self.recoil.tick(dt);

        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 {
            self.spawn_timer += SPAWN_INTERVAL;
            if self.targets.len() < TARGET_COUNT {
                self.spawn_target(eng, TARGET_LIFETIME);
            }
        }

        self.expire_targets(eng, dt);
        self.look(eng);
        self.reload(eng);
        self.shoot(eng);
        // After the shot, so the pick uses the aim the crosshair was on and the
        // new kick lands on top of it.
        self.aim(eng);
        self.animate_gun(eng, dt);
    }
}
