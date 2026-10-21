//! Collisions as deferred messages, not a second event system.
//!
//! Rapier reports touching pairs; the engine drains them into `pings` after the
//! step. The backend owns no bus, so an event is a value in a queue the caller
//! takes: enter and exit are the difference against the previous step's touching
//! set, and an unchanged pair is a `Contact` each step it holds.
//!
//! Order is by entity index, so a run produces the same message order every
//! build. The simulation does not read these, so a dropped queue cannot change a
//! stepped frame.

use std::collections::{HashMap, HashSet};

use glam::Vec3;
use meats::Entity;
use rapier3d::prelude::{ColliderHandle, ColliderSet, NarrowPhase};

use crate::body::Entry;
use crate::convert;

/// Two bodies began touching.
///
/// `point` is the midpoint of the two colliders, coarse but glam: a per-contact
/// world point would name Rapier's manifold types outside `convert`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CollisionEvent {
    pub a: Entity,
    pub b: Entity,
    pub kind: CollisionKind,
    pub point: Vec3,
}

/// Whether a pair is new this step, still touching, or gone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollisionKind {
    Enter,
    Contact,
    Exit,
}

/// Diffs this step's touching pairs against `touching`, appends the events to
/// `out`, and updates `touching` to this step's set.
pub(crate) fn collect(
    narrow_phase: &NarrowPhase,
    colliders: &ColliderSet,
    entries: &HashMap<Entity, Entry>,
    touching: &mut HashSet<(Entity, Entity)>,
    out: &mut Vec<CollisionEvent>,
) {
    let owners: HashMap<ColliderHandle, Entity> = entries
        .iter()
        .filter_map(|(entity, entry)| Some((entry.collider?, *entity)))
        .collect();

    let mut current: HashMap<(Entity, Entity), Vec3> = HashMap::new();
    for pair in narrow_phase.contact_pairs() {
        if !pair.has_any_active_contact() {
            continue;
        }
        if let Some(key) = key_for(&owners, pair.collider1, pair.collider2) {
            current.insert(
                key,
                point_between(colliders, pair.collider1, pair.collider2),
            );
        }
    }
    for (c1, c2, intersecting) in narrow_phase.intersection_pairs() {
        if !intersecting {
            continue;
        }
        if let Some(key) = key_for(&owners, c1, c2) {
            current.insert(key, point_between(colliders, c1, c2));
        }
    }

    let mut step: Vec<CollisionEvent> = Vec::new();
    for (key, point) in &current {
        let kind = if touching.contains(key) {
            CollisionKind::Contact
        } else {
            CollisionKind::Enter
        };
        step.push(event(*key, kind, *point));
    }
    for key in touching.iter() {
        if !current.contains_key(key) {
            step.push(event(*key, CollisionKind::Exit, Vec3::ZERO));
        }
    }
    step.sort_by_key(|event| (event.a.index(), event.a.generation(), event.b.index()));

    touching.clear();
    touching.extend(current.into_keys());
    out.extend(step);
}

fn key_for(
    owners: &HashMap<ColliderHandle, Entity>,
    c1: ColliderHandle,
    c2: ColliderHandle,
) -> Option<(Entity, Entity)> {
    let a = *owners.get(&c1)?;
    let b = *owners.get(&c2)?;
    Some(
        if (a.index(), a.generation()) <= (b.index(), b.generation()) {
            (a, b)
        } else {
            (b, a)
        },
    )
}

/// The midpoint of two colliders, matching what an event reports.
pub(crate) fn point_between(
    colliders: &ColliderSet,
    c1: ColliderHandle,
    c2: ColliderHandle,
) -> Vec3 {
    let p1 = colliders.get(c1).map(convert::collider_translation);
    let p2 = colliders.get(c2).map(convert::collider_translation);
    match (p1, p2) {
        (Some(a), Some(b)) => (a + b) * 0.5,
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => Vec3::ZERO,
    }
}

fn event((a, b): (Entity, Entity), kind: CollisionKind, point: Vec3) -> CollisionEvent {
    CollisionEvent { a, b, kind, point }
}
