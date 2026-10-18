//! The timestep boundary: the engine's accumulator owns the rate and the clamp,
//! and the backend must take what it is given without running time backwards.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::Physics3D;
use glam::Vec3;
use meats::{RigidBodyKind, Scene};

#[test]
fn a_negative_delta_does_not_run_time_backwards() {
    let mut scene = Scene::new();
    let entity = add_body(
        &mut scene,
        "box",
        Vec3::new(0.0, 10.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );

    let mut physics = Physics3D::new();
    // The engine clamps its accumulator at zero; a backend that fed Rapier a
    // negative dt would integrate a step backwards and lift the body.
    physics.step(&mut scene, -1.0);

    let after = translation(&scene, entity);
    assert!(
        (after - Vec3::new(0.0, 10.0, 0.0)).length() < 1.0e-6,
        "a negative delta moved the body to {after:?}"
    );
}

#[test]
fn a_zero_delta_changes_nothing_but_a_fixed_step_does() {
    let mut scene = Scene::new();
    let entity = add_body(
        &mut scene,
        "box",
        Vec3::new(0.0, 10.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );

    let mut physics = Physics3D::new();
    physics.step(&mut scene, 0.0);
    let after_zero = translation(&scene, entity);
    assert!(
        (after_zero - Vec3::new(0.0, 10.0, 0.0)).length() < 1.0e-6,
        "a zero delta moved the body to {after_zero:?}"
    );

    physics.step(&mut scene, DT);
    assert!(
        translation(&scene, entity).y < after_zero.y,
        "one fixed step did not advance the body"
    );
}
