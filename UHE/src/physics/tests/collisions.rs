//! Collisions as deferred messages: a landing reports enter then contacts, and
//! two bodies that never meet report nothing.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::{CollisionKind, Physics3D};
use glam::Vec3;
use meats::{Entity, RigidBodyKind, Scene};

fn pair(a: Entity, b: Entity) -> (u32, u32, u32, u32) {
    let (a, b) = if a.index() <= b.index() {
        (a, b)
    } else {
        (b, a)
    };
    (a.index(), a.generation(), b.index(), b.generation())
}

fn landing() -> (Scene, Entity, Entity) {
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
        Vec3::new(0.0, 2.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );
    (scene, ground, box_entity)
}

#[test]
fn a_box_landing_reports_an_enter_then_contacts() {
    let (mut scene, ground, box_entity) = landing();
    let wanted = pair(ground, box_entity);
    let mut physics = Physics3D::new();

    let (mut saw_enter, mut saw_contact) = (false, false);
    for _ in 0..120 {
        physics.step(&mut scene, DT);
        for event in physics.drain_collision_events() {
            if pair(event.a, event.b) != wanted {
                continue;
            }
            match event.kind {
                CollisionKind::Enter => saw_enter = true,
                CollisionKind::Contact => saw_contact = true,
                CollisionKind::Exit => {}
            }
        }
    }

    assert!(saw_enter, "the landing never reported an enter");
    assert!(saw_contact, "a resting contact was not reported");
}

#[test]
fn bodies_that_never_meet_report_nothing() {
    let mut scene = Scene::new();
    add_body(
        &mut scene,
        "left",
        Vec3::new(-50.0, 0.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );
    add_body(
        &mut scene,
        "right",
        Vec3::new(50.0, 0.0, 0.0),
        RigidBodyKind::Dynamic,
        box_shape(Vec3::splat(0.5)),
    );

    let mut physics = Physics3D::new();
    let mut events = Vec::new();
    for _ in 0..60 {
        physics.step(&mut scene, DT);
        events.extend(physics.drain_collision_events());
    }

    assert!(
        events.is_empty(),
        "reported {events:?} with nothing touching"
    );
}

#[test]
fn the_event_point_sits_between_the_two_bodies() {
    let (mut scene, ground, box_entity) = landing();
    let wanted = pair(ground, box_entity);
    let mut physics = Physics3D::new();

    let mut point = None;
    for _ in 0..120 {
        physics.step(&mut scene, DT);
        for event in physics.drain_collision_events() {
            if pair(event.a, event.b) == wanted && event.kind == CollisionKind::Enter {
                point = Some(event.point);
            }
        }
    }

    let point = point.expect("no enter");
    // The box rests at y = 0.5 and the ground's top is y = 0, so a midpoint lies
    // in between and never below the ground.
    assert!(point.y > -0.5 && point.y < 0.6, "point at {point:?}");
}
