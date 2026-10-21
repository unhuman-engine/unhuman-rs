//! `AllowedDOFs`: a file that constrains a body must constrain the simulation.
//!
//! The C++ serialized and deserialized the component and never handed it to the
//! solver, so a plane-locked body fell anyway. Each case has a control with no
//! component, so a pass means the constraint acted and not that the scene was
//! already still.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::{AllowedDOFs, Physics3D};
use glam::Vec3;
use meats::{RigidBodyKind, Scene};

/// Falls one step, against gravity. `dofs` is the constraint under test.
fn falling(dofs: Option<AllowedDOFs>) -> (Scene, meats::Entity) {
    let mut scene = Scene::new();
    let entity = add_body(
        &mut scene,
        "box",
        Vec3::new(0.0, 5.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );
    if let Some(dofs) = dofs {
        scene.get_mut(entity).expect("box").insert(dofs);
    }
    (scene, entity)
}

fn run(scene: &mut Scene, steps: usize) {
    let mut physics = Physics3D::new();
    for _ in 0..steps {
        physics.step(scene, DT);
    }
}

#[test]
fn locking_the_y_translation_keeps_a_body_on_its_plane() {
    // Translation x and z free, y locked; all rotations free. The body may slide
    // in the x/z plane but gravity must not pull it down it.
    let (mut scene, entity) = falling(Some(AllowedDOFs::new(
        [true, false, true],
        [true, true, true],
    )));
    run(&mut scene, 120);

    let y = translation(&scene, entity).y;
    assert!((y - 5.0).abs() < 1.0e-4, "a y-locked body fell to y = {y}");
}

#[test]
fn without_the_component_the_same_body_falls() {
    let (mut scene, entity) = falling(None);
    run(&mut scene, 120);

    let y = translation(&scene, entity).y;
    assert!(y < 4.0, "the control did not fall: y = {y}");
}

#[test]
fn a_plane_lock_also_holds_the_tip_rotations() {
    // The usual plane constraint: one translation axis locked and the two
    // rotations that would tip the body off the plane. An off-centre start gives
    // the rotation something to do.
    let (mut scene, entity) = falling(Some(AllowedDOFs::new(
        [true, false, true],
        [false, true, false],
    )));
    scene
        .get_mut(entity)
        .expect("box")
        .get_mut::<meats::Transform>()
        .expect("transform")
        .rotation = Vec3::new(20.0, 0.0, 10.0);
    run(&mut scene, 120);

    let after = rotation(&scene, entity);
    assert!(
        (after - Vec3::new(20.0, 0.0, 10.0)).length() < 1.0e-3,
        "a locked rotation moved: {after:?}"
    );
}

#[test]
fn the_default_component_allows_every_axis() {
    let (mut scene, entity) = falling(Some(AllowedDOFs::ALL));
    run(&mut scene, 120);

    let y = translation(&scene, entity).y;
    assert!(y < 4.0, "AllowedDOFs::ALL still locked a body: y = {y}");
}
