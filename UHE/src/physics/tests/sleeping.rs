//! Sleeping bodies: the pose at the sleep transition must reach the scene.
//!
//! The C++ sync-back skipped every inactive body, so the step a body fell asleep
//! on was dropped and a scene saved afterwards held the previous frame's pose.
//! [`Physics3D::pose_of`] is the simulation's own pose, so the assertion here is
//! the exact one the bug broke.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::Physics3D;
use glam::Vec3;
use meats::{RigidBodyKind, Scene};

#[test]
fn a_body_that_falls_asleep_leaves_the_scene_holding_its_sleep_pose() {
    let (mut scene, entity) = {
        let mut scene = Scene::new();
        add_body(
            &mut scene,
            "ground",
            Vec3::new(0.0, -1.0, 0.0),
            RigidBodyKind::Static,
            box_shape(Vec3::new(50.0, 1.0, 50.0)),
        );
        let entity = add_body(
            &mut scene,
            "box",
            Vec3::new(0.0, 2.0, 0.0),
            RigidBodyKind::Dynamic,
            box_shape(Vec3::splat(0.5)),
        );
        (scene, entity)
    };

    let mut physics = Physics3D::new();
    for _ in 0..400 {
        physics.step(&mut scene, DT);
    }
    assert!(physics.is_sleeping(entity), "the box never slept");

    // Bit-exact: the failing case differs by the last step's motion, which is
    // below the sleep threshold and so invisible to a tolerance.
    let (simulated, _) = physics.pose_of(entity).expect("the box has a body");
    assert_eq!(
        translation(&scene, entity),
        simulated,
        "the scene holds a stale pose from before the body slept"
    );
}

#[test]
fn a_sleeping_body_that_is_edited_is_pushed_and_woken() {
    let (mut scene, entity) = {
        let mut scene = Scene::new();
        add_body(
            &mut scene,
            "ground",
            Vec3::new(0.0, -1.0, 0.0),
            RigidBodyKind::Static,
            box_shape(Vec3::new(50.0, 1.0, 50.0)),
        );
        let entity = add_body(
            &mut scene,
            "box",
            Vec3::new(0.0, 2.0, 0.0),
            RigidBodyKind::Dynamic,
            box_shape(Vec3::splat(0.5)),
        );
        (scene, entity)
    };

    let mut physics = Physics3D::new();
    for _ in 0..400 {
        physics.step(&mut scene, DT);
    }
    assert!(physics.is_sleeping(entity));

    // An editor drag while the body is asleep. The next step must start from the
    // edit, which means the body was woken, not silently skipped.
    let lifted = Vec3::new(0.0, 3.0, 0.0);
    scene
        .get_mut(entity)
        .expect("box")
        .get_mut::<meats::Transform>()
        .expect("transform")
        .translation = lifted;
    physics.step(&mut scene, DT);

    assert!(
        !physics.is_sleeping(entity),
        "the edit did not wake the body"
    );
    let after = translation(&scene, entity);
    assert!(after.y > lifted.y - 0.05, "the edit was ignored: {after:?}");
}
