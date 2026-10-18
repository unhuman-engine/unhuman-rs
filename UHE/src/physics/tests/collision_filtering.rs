//! Collision filtering: a group that excludes an otherwise certain contact, and
//! a sensor that passes through it. Both have a control that keeps the contact,
//! so the assertion is about the filter and not the scene.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::{CollisionFilter, CollisionGroups, Physics3D};
use glam::Vec3;
use meats::{Entity, RigidBodyKind, Scene};

const GROUP_0: u32 = 1 << 0;
const GROUP_1: u32 = 1 << 1;

/// A box resting on a static ground; it contacts the ground at y = 0.5.
fn resting_scene() -> (Scene, Entity, Entity) {
    let mut scene = Scene::new();
    let ground = add_body(
        &mut scene,
        "ground",
        Vec3::new(0.0, -1.0, 0.0),
        RigidBodyKind::Static,
        box_shape(Vec3::new(50.0, 1.0, 50.0)),
    );
    let box_entity = add_body(
        &mut scene,
        "box",
        Vec3::new(0.0, 0.5, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );
    (scene, ground, box_entity)
}

#[test]
fn a_collision_group_prevents_an_otherwise_certain_contact() {
    let (mut scene, ground, box_entity) = resting_scene();
    // Each is in its own group and filters only for itself, so neither sees the
    // other: the box falls through a contact it would otherwise rest on.
    scene
        .get_mut(ground)
        .expect("ground")
        .insert(CollisionFilter {
            groups: CollisionGroups::new(GROUP_1, GROUP_1),
            sensor: false,
        });
    scene
        .get_mut(box_entity)
        .expect("box")
        .insert(CollisionFilter {
            groups: CollisionGroups::new(GROUP_0, GROUP_0),
            sensor: false,
        });

    let mut physics = Physics3D::new();
    for _ in 0..120 {
        physics.step(&mut scene, DT);
    }

    let y = translation(&scene, box_entity).y;
    assert!(
        y < -1.0,
        "the box rested on a group it cannot collide with: y = {y}"
    );
}

#[test]
fn the_default_groups_keep_the_contact() {
    let (mut scene, _, box_entity) = resting_scene();
    // No filter component at all: the default is every group.
    let mut physics = Physics3D::new();
    for _ in 0..300 {
        physics.step(&mut scene, DT);
    }

    let y = translation(&scene, box_entity).y;
    assert!((y - 0.5).abs() < 0.05, "the box did not rest at y = {y}");
}

#[test]
fn overlapping_groups_keep_the_contact() {
    let (mut scene, ground, box_entity) = resting_scene();
    // Both are in group 0 and both filter for group 0: the masks agree, so the
    // contact survives a filter that is not the default.
    let filter = CollisionFilter {
        groups: CollisionGroups::new(GROUP_0, GROUP_0),
        sensor: false,
    };
    scene.get_mut(ground).expect("ground").insert(filter);
    scene.get_mut(box_entity).expect("box").insert(filter);

    let mut physics = Physics3D::new();
    for _ in 0..300 {
        physics.step(&mut scene, DT);
    }

    let y = translation(&scene, box_entity).y;
    assert!(
        (y - 0.5).abs() < 0.05,
        "overlapping groups did not meet: y = {y}"
    );
}

#[test]
fn a_sensor_passes_through_what_it_would_rest_on() {
    let (mut scene, _, box_entity) = resting_scene();
    scene
        .get_mut(box_entity)
        .expect("box")
        .insert(CollisionFilter {
            groups: CollisionGroups::ALL,
            sensor: true,
        });

    let mut physics = Physics3D::new();
    for _ in 0..120 {
        physics.step(&mut scene, DT);
    }

    let y = translation(&scene, box_entity).y;
    assert!(y < -1.0, "the sensor was stopped by a contact: y = {y}");
}
