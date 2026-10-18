//! Shared scene fixtures for the physics tests: a body with a transform, and
//! readers for it. Damping is zeroed so the numbers a test asserts are the
//! integrator's, not the component default's.

#![allow(dead_code)]

use glam::Vec3;
use meats::{Collider, Entity, RigidBody, RigidBodyKind, Scene, Transform};

/// The engine's fixed step (60 Hz, `adam::Time::new`).
pub const DT: f32 = 1.0 / 60.0;

pub fn box_shape(half_extent: Vec3) -> femur::ColliderShape {
    femur::ColliderShape::Box { half_extent }
}

pub fn add_body(
    scene: &mut Scene,
    name: &str,
    translation: Vec3,
    kind: RigidBodyKind,
    shape: femur::ColliderShape,
) -> Entity {
    let entity = scene.spawn(name);
    let mut built = scene.get_mut(entity).expect("spawned");
    built
        .get_mut::<Transform>()
        .expect("every entity has a transform")
        .translation = translation;
    built.insert(RigidBody {
        kind,
        mass: 1.0,
        velocity: Vec3::ZERO,
        damping: 0.0,
        gravity_scale: 1.0,
    });
    built.insert(Collider {
        shape,
        offset: Vec3::ZERO,
        friction: 0.5,
        restitution: 0.0,
    });
    entity
}

pub fn translation(scene: &Scene, entity: Entity) -> Vec3 {
    scene
        .get(entity)
        .expect("entity")
        .get::<Transform>()
        .expect("transform")
        .translation
}

pub fn rotation(scene: &Scene, entity: Entity) -> Vec3 {
    scene
        .get(entity)
        .expect("entity")
        .get::<Transform>()
        .expect("transform")
        .rotation
}
