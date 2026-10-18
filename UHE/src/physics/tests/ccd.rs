//! Per-body CCD: a fast body against a thin wall.
//!
//! Rapier already runs CCD for a fast dynamic body against fixed colliders, so
//! the wall here is kinematic: only `Ccd::enabled` widens the sweep from
//! fixed-only to every body. Without the toggle the body tunnels, which is the
//! control.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::{Ccd, Physics3D};
use glam::Vec3;
use meats::{Entity, RigidBody, RigidBodyKind, Scene};

/// A bullet fired at a thin kinematic wall. `ccd` selects the toggle.
fn bullet_scene(ccd: bool) -> (Scene, Entity) {
    let mut scene = Scene::new();
    add_body(
        &mut scene,
        "wall",
        Vec3::ZERO,
        RigidBodyKind::Kinematic,
        box_shape(Vec3::new(0.05, 5.0, 5.0)),
    );
    let bullet = add_body(
        &mut scene,
        "bullet",
        Vec3::new(-3.0, 0.0, 0.0),
        RigidBodyKind::Dynamic,
        femur::ColliderShape::Sphere { radius: 0.05 },
    );

    {
        let mut built = scene.get_mut(bullet).expect("bullet");
        let body = built.get_mut::<RigidBody>().expect("rigid body");
        // 300 m/s is 5 m per step, far past the 0.1 m wall; no gravity so the
        // path stays on the axis.
        body.velocity = Vec3::new(300.0, 0.0, 0.0);
        body.gravity_scale = 0.0;
    }
    if ccd {
        scene.get_mut(bullet).expect("bullet").insert(Ccd {
            enabled: true,
            prediction: 0.0,
        });
    }

    (scene, bullet)
}

#[test]
fn ccd_stops_a_fast_body_tunnelling_through_a_thin_wall() {
    let (mut scene, bullet) = bullet_scene(true);
    let mut physics = Physics3D::new();
    for _ in 0..8 {
        physics.step(&mut scene, DT);
    }

    let x = translation(&scene, bullet).x;
    assert!(x < 0.0, "the bullet tunnelled to x = {x} despite CCD");
}

#[test]
fn without_ccd_the_same_body_tunnels() {
    let (mut scene, bullet) = bullet_scene(false);
    let mut physics = Physics3D::new();
    for _ in 0..8 {
        physics.step(&mut scene, DT);
    }

    let x = translation(&scene, bullet).x;
    assert!(
        x > 0.0,
        "the bullet stopped at x = {x} without CCD, so the wall was not thin enough to prove the toggle"
    );
}
