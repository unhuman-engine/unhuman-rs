//! The aim lab's rules against a real `meats` scene and no GPU.
//!
//! Every test here drives the plugin's own hooks with a fixed step, which is what
//! a headless run does; nothing touches a device, a window or the filesystem.

use std::time::Duration;

use adam::{Engine, FrameCtx, Plugin};
use aimlab::{
    ray_sphere, AimLabPlugin, Score, CAMERA_DISTANCE, HIT_POINTS, SPAWN_X, SPAWN_Y, SPAWN_Z,
    STAGE_ENTITIES, TARGET_COUNT, TARGET_RADIUS,
};
use glam::Vec3;
use meats::{Entity, Transform};
use pings::{Event, MouseButton};

const STEP: f32 = 1.0 / 60.0;
const STEPS_PER_SECOND: usize = 60;

fn context() -> FrameCtx {
    FrameCtx {
        frame: 0,
        delta: Duration::from_secs_f32(STEP),
        step: Duration::from_secs_f32(STEP),
        scale: 1.0,
        quit: false,
    }
}

fn built() -> (Engine, AimLabPlugin) {
    let mut engine = Engine::new();
    let mut plugin = AimLabPlugin::new();
    plugin.build(&mut engine).unwrap();
    (engine, plugin)
}

fn run(steps: usize) -> (Engine, AimLabPlugin) {
    let (mut engine, mut plugin) = built();
    let mut ctx = context();
    for _ in 0..steps {
        plugin.on_update(&mut engine, &mut ctx);
    }
    (engine, plugin)
}

fn translation(engine: &Engine, entity: Entity) -> Vec3 {
    engine.scene.world_transform(entity).w_axis.truncate()
}

fn place(engine: &mut Engine, entity: Entity, position: Vec3) {
    let mut entry = engine.scene.get_mut(entity).expect("entity exists");
    entry
        .get_mut::<Transform>()
        .expect("every entity has a transform")
        .translation = position;
}

fn click(engine: &mut Engine) {
    engine.input.begin_frame();
    engine
        .input
        .apply(&Event::MouseButtonPressed(MouseButton::Left));
}

#[test]
fn build_populates_a_scene() {
    let (engine, plugin) = built();

    assert_eq!(engine.scene.entities().len(), STAGE_ENTITIES + TARGET_COUNT);
    assert_eq!(plugin.target_count(), TARGET_COUNT);
    assert!(engine.scene.primary_camera().is_some());

    for entity in plugin.target_entities() {
        let position = translation(&engine, entity);
        assert_eq!(position.z, SPAWN_Z);
        assert!((SPAWN_X.0..=SPAWN_X.1).contains(&position.x), "{position}");
        assert!((SPAWN_Y.0..=SPAWN_Y.1).contains(&position.y), "{position}");
    }
}

#[test]
fn a_hit_adds_a_hit_and_the_points() {
    let (mut engine, mut plugin) = built();
    let target = plugin.target_entities()[0];
    place(&mut engine, target, Vec3::new(0.0, 0.0, SPAWN_Z));

    click(&mut engine);
    plugin.on_update(&mut engine, &mut context());

    assert_eq!(plugin.score().shots, 1);
    assert_eq!(plugin.score().hits, 1);
    assert_eq!(plugin.score().points, HIT_POINTS);
    assert_eq!(plugin.score().misses, 0);
    assert_eq!(plugin.target_count(), TARGET_COUNT);
}

#[test]
fn a_shot_that_hits_nothing_still_counts_as_a_shot() {
    let (mut engine, mut plugin) = built();
    let camera = engine.scene.primary_camera().unwrap();
    // Well clear of the spawn box, so the centre ray meets no target.
    place(&mut engine, camera, Vec3::new(100.0, 0.0, SPAWN_Z + 5.0));

    click(&mut engine);
    plugin.on_update(&mut engine, &mut context());

    assert_eq!(plugin.score().shots, 1);
    assert_eq!(plugin.score().hits, 0);
    assert_eq!(plugin.score().points, 0);
}

#[test]
fn holding_the_button_fires_once() {
    let (mut engine, mut plugin) = built();
    let target = plugin.target_entities()[0];
    place(&mut engine, target, Vec3::new(0.0, 0.0, SPAWN_Z));

    click(&mut engine);
    let mut ctx = context();
    for _ in 0..3 {
        plugin.on_update(&mut engine, &mut ctx);
    }

    assert_eq!(plugin.score().shots, 1, "a held button is one shot");
    assert_eq!(plugin.score().hits, 1);
}

#[test]
fn an_expired_target_counts_as_a_miss() {
    let (engine, plugin) = run(6 * STEPS_PER_SECOND + STEPS_PER_SECOND / 2);

    assert_eq!(plugin.score().misses, 1);
    assert_eq!(plugin.score().hits, 0);
    assert_eq!(plugin.score().points, 0);
    assert_eq!(plugin.target_count(), TARGET_COUNT - 1);
    assert_eq!(
        engine.scene.entities().len(),
        STAGE_ENTITIES + TARGET_COUNT - 1
    );
}

#[test]
fn the_spawn_cadence_is_deterministic_for_a_fixed_step_count() {
    let (engine_a, plugin_a) = run(9 * STEPS_PER_SECOND + STEPS_PER_SECOND / 2);
    let (engine_b, plugin_b) = run(9 * STEPS_PER_SECOND + STEPS_PER_SECOND / 2);

    // One target per second from 6s, replaced one per second from 7s.
    assert_eq!(plugin_a.score().misses, 4);
    assert_eq!(plugin_a.target_count(), TARGET_COUNT - 1);
    assert_eq!(
        engine_a.scene.entities().len(),
        STAGE_ENTITIES + TARGET_COUNT - 1
    );

    assert_eq!(plugin_a.score(), plugin_b.score());
    let positions_a: Vec<Vec3> = plugin_a
        .target_entities()
        .iter()
        .map(|entity| translation(&engine_a, *entity))
        .collect();
    let positions_b: Vec<Vec3> = plugin_b
        .target_entities()
        .iter()
        .map(|entity| translation(&engine_b, *entity))
        .collect();
    assert_eq!(positions_a, positions_b);
    assert!(!positions_a.is_empty());
}

#[test]
fn the_camera_carries_the_shot_sound() {
    let (engine, _) = built();
    let camera = engine.scene.primary_camera().expect("primary camera");
    let entry = engine.scene.get(camera).expect("the camera entity");
    let source = entry
        .get::<meats::AudioSource>()
        .expect("a sound component");

    assert_eq!(source.clip_path, "assets/audio/gunshot.wav");
    assert!(
        !source.playing,
        "a one-shot goes through the backend, not a held `playing` flag"
    );
}

#[test]
fn a_target_directly_ahead_is_picked() {
    let (mut engine, mut plugin) = built();
    // Clear the centre ray of the other four so the hit is unambiguous.
    for entity in plugin.target_entities() {
        place(&mut engine, entity, Vec3::new(8.0, 5.0, SPAWN_Z));
    }
    let target = plugin.target_entities()[0];
    place(&mut engine, target, Vec3::new(0.0, 0.0, SPAWN_Z));

    click(&mut engine);
    plugin.on_update(&mut engine, &mut context());

    assert_eq!(plugin.score().hits, 1);
}

#[test]
fn a_target_off_to_the_side_is_not_picked() {
    let (mut engine, mut plugin) = built();
    // `sqrt(8^2 + 5^2) ~= 9.4` off the axis is well outside a radius-1 sphere.
    for entity in plugin.target_entities() {
        place(&mut engine, entity, Vec3::new(8.0, 5.0, SPAWN_Z));
    }

    click(&mut engine);
    plugin.on_update(&mut engine, &mut context());

    assert_eq!(plugin.score().shots, 1);
    assert_eq!(plugin.score().hits, 0);
}

#[test]
fn the_camera_faces_the_wall_from_five_units_behind_the_origin() {
    let (engine, _) = built();
    let camera = engine.scene.primary_camera().expect("primary camera");
    let transform = engine.scene.world_transform(camera);
    let position = transform.w_axis.truncate();
    let forward = -transform.z_axis.truncate().normalize_or_zero();

    assert!(
        (position - Vec3::new(0.0, 0.0, -CAMERA_DISTANCE)).length() < 1e-4,
        "camera at {position}"
    );
    assert!(
        forward.dot(Vec3::Z) > 0.999,
        "forward {forward} misses the wall"
    );
}

#[test]
fn targets_sit_at_the_cpp_distance_not_five() {
    let (engine, plugin) = built();
    let camera = engine.scene.primary_camera().expect("primary camera");
    let camera_position = translation(&engine, camera);
    // The wall centre is exactly the orbit radius plus the wall's own Z.
    let wall_centre = Vec3::new(0.0, 0.0, SPAWN_Z);
    assert!(((wall_centre - camera_position).length() - (CAMERA_DISTANCE + SPAWN_Z)).abs() < 1e-4);
    for entity in plugin.target_entities() {
        let distance = (translation(&engine, entity) - camera_position).length();
        assert!((14.0..=18.0).contains(&distance), "{distance}");
        assert!((distance - 5.0).abs() > 1.0, "the old 5-unit framing");
    }
}

#[test]
fn a_hit_is_worth_its_points_and_a_miss_is_not() {
    let mut score = Score::default();
    score.hit();
    assert_eq!(score.hits, 1);
    assert_eq!(score.points, HIT_POINTS);

    score.miss();
    assert_eq!(score.misses, 1);
    assert_eq!(score.points, HIT_POINTS, "a miss scores nothing");

    score.shot();
    score.shot();
    assert_eq!(score.shots, 2);
    assert_eq!(score.accuracy(), 50.0);
}

#[test]
fn a_ray_at_the_drawn_silhouette_edge_hits() {
    let forward = Vec3::Z;
    let center = Vec3::new(0.0, 0.0, 15.0);

    assert_eq!(
        ray_sphere(Vec3::ZERO, forward, center, TARGET_RADIUS),
        Some(14.0)
    );
    // Tangent: a ray at exactly the drawn radius still grazes the sphere.
    let edge = Vec3::new(TARGET_RADIUS, 0.0, 0.0);
    assert!(ray_sphere(edge, forward, center, TARGET_RADIUS).is_some());
    // A hair past the silhouette clears it.
    let outside = Vec3::new(TARGET_RADIUS + 0.01, 0.0, 0.0);
    assert!(ray_sphere(outside, forward, center, TARGET_RADIUS).is_none());

    assert!(ray_sphere(
        Vec3::ZERO,
        forward,
        Vec3::new(0.0, 0.0, -5.0),
        TARGET_RADIUS
    )
    .is_none());
    assert!(ray_sphere(Vec3::ZERO, Vec3::ZERO, center, TARGET_RADIUS).is_none());
}
