//! The collider shape in the file is the shape in the simulation.
//!
//! The C++ declared a `RuntimeShape` for the box collider and never wrote it, so
//! the shape the file described and the shape the solver used could disagree.
//! Here the resting height is the shape's own size, so a change to the file's
//! half-extent or the transform's scale must move it.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::Physics3D;
use glam::Vec3;
use meats::{Collider, Entity, RigidBodyKind, Scene, Transform};

fn resting_scene() -> (Scene, Entity) {
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
}

fn settle(scene: &mut Scene, physics: &mut Physics3D) {
    for _ in 0..300 {
        physics.step(scene, DT);
    }
}

#[test]
fn changing_the_files_half_extent_moves_the_resting_body() {
    let (mut scene, entity) = resting_scene();
    let mut physics = Physics3D::new();
    settle(&mut scene, &mut physics);
    assert!(
        (translation(&scene, entity).y - 0.5).abs() < 0.05,
        "a 0.5 half-extent did not rest at 0.5"
    );

    // The file's shape changes; the simulation's must follow it, not keep the
    // shape it was first built with.
    scene
        .get_mut(entity)
        .expect("box")
        .get_mut::<Collider>()
        .expect("collider")
        .shape = femur::ColliderShape::Box {
        half_extent: Vec3::splat(1.0),
    };
    settle(&mut scene, &mut physics);

    let y = translation(&scene, entity).y;
    assert!(
        (y - 1.0).abs() < 0.05,
        "the file says a 1.0 half-extent, the body rests at {y}"
    );
}

#[test]
fn the_transforms_scale_is_applied_to_the_simulated_shape() {
    let (mut scene, entity) = resting_scene();
    let mut physics = Physics3D::new();
    settle(&mut scene, &mut physics);

    // Scaling y by two makes the box's half-extent 1.0 along y, so it rests a
    // metre up instead of half.
    scene
        .get_mut(entity)
        .expect("box")
        .get_mut::<Transform>()
        .expect("transform")
        .scale = Vec3::new(1.0, 2.0, 1.0);
    settle(&mut scene, &mut physics);

    let y = translation(&scene, entity).y;
    assert!(
        (y - 1.0).abs() < 0.05,
        "the scale was not applied to the collider: rests at {y}"
    );
}
