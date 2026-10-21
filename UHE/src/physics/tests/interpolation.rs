//! Render-time interpolation: the scene a frame sees is between the last two
//! fixed steps, and that blend must not feed back into the simulation.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::{Physics3D, GRAVITY};
use glam::Vec3;
use meats::{RigidBodyKind, Scene};

fn falling(name: &str) -> (Scene, meats::Entity, Physics3D) {
    let mut scene = Scene::new();
    let entity = add_body(
        &mut scene,
        name,
        Vec3::new(0.0, 10.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );
    (scene, entity, Physics3D::new())
}

#[test]
fn a_frame_between_steps_sees_a_blend_of_the_last_two() {
    let (mut scene, entity, mut physics) = falling("box");

    physics.step(&mut scene, DT);
    let first = translation(&scene, entity);
    physics.step(&mut scene, DT);
    let second = translation(&scene, entity);
    assert!(second.y < first.y, "two steps did not advance the body");

    physics.interpolate(&mut scene, 0.0);
    let start = translation(&scene, entity);
    assert!(
        (start - first).length() < 1.0e-5,
        "alpha 0 showed {start:?}, not the previous step {first:?}"
    );

    physics.interpolate(&mut scene, 1.0);
    let end = translation(&scene, entity);
    assert!(
        (end - second).length() < 1.0e-5,
        "alpha 1 showed {end:?}, not the current step {second:?}"
    );

    physics.interpolate(&mut scene, 0.5);
    let middle = translation(&scene, entity);
    let expected = first.lerp(second, 0.5);
    assert!(
        (middle - expected).length() < 1.0e-5,
        "alpha 0.5 showed {middle:?}, not {expected:?}"
    );
}

#[test]
fn the_interpolated_pose_is_not_pushed_back_as_an_authored_change() {
    let (mut scene, entity, mut physics) = falling("box");

    physics.step(&mut scene, DT);
    let first = translation(&scene, entity);
    physics.step(&mut scene, DT);
    let second = translation(&scene, entity);

    // Rewind the scene to the previous step, as a frame between steps does.
    physics.interpolate(&mut scene, 0.0);
    assert!((translation(&scene, entity) - first).length() < 1.0e-5);

    // The next step must continue from Rapier's pose, not from the rewound
    // scene. A feedback bug shows as a body that stalls or falls slower.
    physics.step(&mut scene, DT);
    let third = translation(&scene, entity);
    // Constant acceleration: each step's displacement grows by `g * dt^2`.
    let expected = second + (second - first) + GRAVITY * DT * DT;
    assert!(
        (third - expected).length() < 1.0e-3,
        "step after a rewind landed at {third:?}, expected about {expected:?}"
    );
}

#[test]
fn an_authored_teleport_still_wins_over_the_blend() {
    let (mut scene, entity, mut physics) = falling("box");

    physics.step(&mut scene, DT);
    physics.interpolate(&mut scene, 0.0);
    scene
        .get_mut(entity)
        .expect("entity")
        .get_mut::<meats::Transform>()
        .expect("transform")
        .translation = Vec3::new(10.0, 4.0, 0.0);

    physics.step(&mut scene, DT);
    let moved = translation(&scene, entity);
    assert!(
        (moved.x - 10.0).abs() < 1.0e-4 && moved.y < 4.0,
        "a teleport was overwritten by the blend: {moved:?}"
    );
}
