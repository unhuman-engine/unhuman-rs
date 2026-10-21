//! The debug view is data: shapes, contacts and broad-phase boxes as segments.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::Physics3D;
use glam::Vec3;
use meats::{RigidBodyKind, Scene};

#[test]
fn every_simulated_box_contributes_twelve_shape_edges() {
    let mut scene = Scene::new();
    add_body(
        &mut scene,
        "box",
        Vec3::new(0.0, 5.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );
    let mut physics = Physics3D::new();
    physics.step(&mut scene, DT);

    let view = physics.debug_view();
    assert_eq!(view.shapes.len(), 12, "one box is twelve edges");
    assert_eq!(view.broad_phase.len(), 12, "one broad-phase cell");
}

#[test]
fn a_resting_contact_is_drawn_and_the_cells_are_two() {
    let mut scene = Scene::new();
    add_body(
        &mut scene,
        "ground",
        Vec3::new(0.0, -1.0, 0.0),
        RigidBodyKind::Static,
        box_shape(Vec3::new(50.0, 1.0, 50.0)),
    );
    add_body(
        &mut scene,
        "box",
        Vec3::new(0.0, 0.5, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );
    let mut physics = Physics3D::new();
    for _ in 0..120 {
        physics.step(&mut scene, DT);
    }

    let view = physics.debug_view();
    assert!(!view.contacts.is_empty(), "a resting contact drew nothing");
    assert_eq!(view.broad_phase.len(), 24, "one cell per collider");
    assert_eq!(view.shapes.len(), 24, "two boxes are two wireframes");
}
