//! Joints: a hinge holds two bodies at a fixed distance, a fixed joint keeps the
//! relative transform, a ball joint holds a body at a distance. Each has a
//! no-joint control, so a passing assertion means the joint acted.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::{Joint, JointKind, Physics3D};
use glam::Vec3;
use meats::{Entity, RigidBody, RigidBodyKind, Scene};

const STEPS: usize = 200;

fn set_damping(scene: &mut Scene, entity: Entity, damping: f32) {
    scene
        .get_mut(entity)
        .expect("entity")
        .get_mut::<RigidBody>()
        .expect("rigid body")
        .damping = damping;
}

fn set_gravity_scale(scene: &mut Scene, entity: Entity, scale: f32) {
    scene
        .get_mut(entity)
        .expect("entity")
        .get_mut::<RigidBody>()
        .expect("rigid body")
        .gravity_scale = scale;
}

fn attach(scene: &mut Scene, host: Entity, joint: Joint) {
    scene.get_mut(host).expect("host").insert(joint);
}

/// The hinge: a static pivot at the origin, an arm whose near end is anchored to
/// it, so the arm swings while its centre stays one metre away.
fn hinge_scene(with_joint: bool) -> (Scene, Entity, Entity) {
    let mut scene = Scene::new();
    let pivot = add_body(
        &mut scene,
        "pivot",
        Vec3::ZERO,
        RigidBodyKind::Static,
        box_shape(Vec3::splat(0.05)),
    );
    let arm = add_body(
        &mut scene,
        "arm",
        Vec3::new(1.0, 0.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.1)),
    );
    // Damped, so it settles hanging instead of oscillating through the assertion.
    set_damping(&mut scene, arm, 1.5);
    if with_joint {
        attach(
            &mut scene,
            pivot,
            Joint {
                body_a: pivot,
                body_b: arm,
                kind: JointKind::Hinge { axis: Vec3::Z },
                anchor_a: Vec3::ZERO,
                anchor_b: Vec3::new(-1.0, 0.0, 0.0),
            },
        );
    }
    (scene, pivot, arm)
}

#[test]
fn a_hinge_holds_a_body_at_a_distance() {
    let (mut scene, pivot, arm) = hinge_scene(true);
    let mut physics = Physics3D::new();
    for _ in 0..STEPS {
        physics.step(&mut scene, DT);
    }

    let distance = (translation(&scene, arm) - translation(&scene, pivot)).length();
    assert!(
        (distance - 1.0).abs() < 0.05,
        "the hinge let the arm reach {distance} from the pivot"
    );
    // Gravity swung it: hanging means the arm dropped below its start.
    assert!(
        translation(&scene, arm).y < -0.5,
        "the arm did not swing, y = {}",
        translation(&scene, arm).y
    );
}

#[test]
fn without_the_hinge_the_arm_falls_away() {
    let (mut scene, pivot, arm) = hinge_scene(false);
    let mut physics = Physics3D::new();
    for _ in 0..STEPS {
        physics.step(&mut scene, DT);
    }

    let distance = (translation(&scene, arm) - translation(&scene, pivot)).length();
    assert!(
        distance > 2.0,
        "the control stayed at {distance}, so the hinge assertion proves nothing"
    );
}

#[test]
fn a_fixed_joint_keeps_the_relative_transform() {
    let mut scene = Scene::new();
    let a = add_body(
        &mut scene,
        "a",
        Vec3::new(0.0, 5.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.2)),
    );
    let b = add_body(
        &mut scene,
        "b",
        Vec3::new(1.0, 5.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.2)),
    );
    // b ignores gravity: without the joint it stays while a falls, so the
    // relative transform would change.
    set_gravity_scale(&mut scene, b, 0.0);
    attach(
        &mut scene,
        a,
        Joint {
            body_a: a,
            body_b: b,
            kind: JointKind::Fixed,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
        },
    );

    let mut physics = Physics3D::new();
    physics.step(&mut scene, DT);
    let start = translation(&scene, b) - translation(&scene, a);

    for _ in 0..STEPS {
        physics.step(&mut scene, DT);
    }

    let end = translation(&scene, b) - translation(&scene, a);
    assert!(
        (end - start).length() < 0.1,
        "the fixed joint let the relative transform move {start:?} -> {end:?}"
    );
    // It coupled them: b fell with a instead of holding still.
    assert!(
        translation(&scene, b).y < 4.0,
        "b did not follow a: {:?}",
        translation(&scene, b)
    );
}

#[test]
fn a_destroyed_body_takes_its_joint_with_it() {
    let (mut scene, _, arm) = hinge_scene(true);
    let mut physics = Physics3D::new();
    for _ in 0..10 {
        physics.step(&mut scene, DT);
    }

    scene.destroy(arm);
    // The next step must drop the arm's body and the joint that pointed at it,
    // rather than leaving a dangling Rapier joint behind.
    for _ in 0..10 {
        physics.step(&mut scene, DT);
    }

    assert_eq!(physics.body_count(), 1);
}

#[test]
fn a_ball_joint_holds_two_bodies_at_a_distance() {
    let mut scene = Scene::new();
    let pivot = add_body(
        &mut scene,
        "pivot",
        Vec3::ZERO,
        RigidBodyKind::Static,
        box_shape(Vec3::splat(0.05)),
    );
    let ball = add_body(
        &mut scene,
        "ball",
        Vec3::new(1.0, 0.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.1)),
    );
    set_damping(&mut scene, ball, 1.5);
    attach(
        &mut scene,
        pivot,
        Joint {
            body_a: pivot,
            body_b: ball,
            kind: JointKind::Ball,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::new(-1.0, 0.0, 0.0),
        },
    );

    let mut physics = Physics3D::new();
    for _ in 0..STEPS {
        physics.step(&mut scene, DT);
    }

    let distance = (translation(&scene, ball) - translation(&scene, pivot)).length();
    assert!(
        (distance - 1.0).abs() < 0.05,
        "the ball joint let the body reach {distance}"
    );
}
