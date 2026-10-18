//! Joints: the scene's [`Joint`] component, and the Rapier impulse joint it
//! builds.
//!
//! Rapier has two joint families. This uses the impulse joints, the ones for a
//! pair of independent bodies; the multibody joints own a tree of links and are
//! not what a scene's pairwise joint describes.
//!
//! A joint is keyed by the entity that carries the component, which may be one of
//! the two bodies or an entity of its own. Both bodies must exist as Rapier
//! bodies first, so this runs after the body pass; a joint whose bodies are not
//! ready is dropped until they are.

use std::collections::{HashMap, HashSet};

use meats::{Entity, Scene};
use rapier3d::prelude::{
    FixedJointBuilder, GenericJoint, ImpulseJointHandle, ImpulseJointSet, RevoluteJointBuilder,
    RigidBodyHandle, SphericalJointBuilder,
};

use crate::body::Entry;
use crate::components::{Joint, JointKind};

/// The joint a host entity gave the backend.
pub(crate) struct JointEntry {
    pub handle: ImpulseJointHandle,
    pub source: JointSource,
}

/// The two resolved bodies and the description the joint was built from. A
/// changed handle means a body was rebuilt, and the joint with it.
#[derive(PartialEq)]
pub(crate) struct JointSource {
    pub a: RigidBodyHandle,
    pub b: RigidBodyHandle,
    pub joint: Joint,
}

/// Reconciles the Rapier joints with the scene's [`Joint`] components.
pub(crate) fn sync(
    scene: &Scene,
    bodies: &HashMap<Entity, Entry>,
    joints: &mut HashMap<Entity, JointEntry>,
    set: &mut ImpulseJointSet,
) {
    let mut alive: HashSet<Entity> = HashSet::new();
    for entity in scene.entities() {
        let Some(joint) = scene.get(entity).and_then(|e| e.get::<Joint>().map(|j| *j)) else {
            continue;
        };
        alive.insert(entity);

        let a = bodies.get(&joint.body_a).and_then(|entry| entry.body);
        let b = bodies.get(&joint.body_b).and_then(|entry| entry.body);
        let (Some(a), Some(b)) = (a, b) else {
            // A body was refused or destroyed: the joint cannot exist, and it is
            // rebuilt once both bodies are back.
            remove(joints, set, entity);
            continue;
        };

        let source = JointSource { a, b, joint };
        if joints
            .get(&entity)
            .is_some_and(|entry| entry.source == source)
        {
            continue;
        }
        remove(joints, set, entity);
        let handle = set.insert(a, b, generic(joint), true);
        joints.insert(entity, JointEntry { handle, source });
    }

    let dead: Vec<Entity> = joints
        .keys()
        .filter(|entity| !alive.contains(entity))
        .copied()
        .collect();
    for entity in dead {
        remove(joints, set, entity);
    }
}

/// Drops a host's joint. The Rapier joint may already be gone if a body was
/// removed with its attached joints.
fn remove(joints: &mut HashMap<Entity, JointEntry>, set: &mut ImpulseJointSet, entity: Entity) {
    if let Some(entry) = joints.remove(&entity) {
        set.remove(entry.handle, true);
    }
}

/// Maps a scene joint to a Rapier one.
///
/// The anchor goes in the body's local space; the hinge axis, in Rapier's
/// convention, is the local axis of both bodies, so it reads as the world axis
/// only while they share an orientation.
fn generic(joint: Joint) -> GenericJoint {
    match joint.kind {
        JointKind::Fixed => FixedJointBuilder::new()
            .local_anchor1(joint.anchor_a)
            .local_anchor2(joint.anchor_b)
            .build()
            .into(),
        JointKind::Hinge { axis } => RevoluteJointBuilder::new(axis)
            .local_anchor1(joint.anchor_a)
            .local_anchor2(joint.anchor_b)
            .build()
            .into(),
        JointKind::Ball => SphericalJointBuilder::new()
            .local_anchor1(joint.anchor_a)
            .local_anchor2(joint.anchor_b)
            .build()
            .into(),
    }
}
