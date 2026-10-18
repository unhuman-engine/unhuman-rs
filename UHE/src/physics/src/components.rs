//! The physics-only components `meats` does not carry, inserted into a scene
//! like any other component.
//!
//! `meats` owns the body and collider description. Joints, collision filtering
//! and per-body CCD have no field there, and this crate cannot grow one, so they
//! are defined here. Every type is glam: the crate's public surface names no
//! Rapier type.

use glam::Vec3;
use meats::Entity;

/// What a [`Joint`] constrains, and the free axis a hinge leaves.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum JointKind {
    /// Locks position and rotation: the two bodies keep their relative transform.
    Fixed,
    /// A revolute joint: locks position and two rotations, leaving one axis. The
    /// axis is expressed in the local space of both bodies, so it only reads as a
    /// world direction while they share an orientation.
    Hinge { axis: Vec3 },
    /// A ball-and-socket joint: locks the two anchors together, leaving rotation.
    Ball,
}

/// A joint between two rigid bodies, keyed by the entity that carries it.
///
/// `body_a` and `body_b` are the joined entities; `anchor_a` and `anchor_b` are
/// the connection point in each body's local space. The component may sit on a
/// body or on its own entity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Joint {
    pub body_a: Entity,
    pub body_b: Entity,
    pub kind: JointKind,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
}

/// Rapier collision layers: a collider is in `memberships` and meets a collider
/// only when each has a bit the other filters for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CollisionGroups {
    pub memberships: u32,
    pub filter: u32,
}

impl CollisionGroups {
    /// In every group and filtering every group: what a collider with no filter
    /// gets, so adding a `CollisionFilter` cannot silently stop collisions.
    pub const ALL: Self = Self {
        memberships: u32::MAX,
        filter: u32::MAX,
    };

    pub const fn new(memberships: u32, filter: u32) -> Self {
        Self {
            memberships,
            filter,
        }
    }

    /// In no group: a collider nothing can meet.
    pub const fn none() -> Self {
        Self {
            memberships: 0,
            filter: 0,
        }
    }
}

impl Default for CollisionGroups {
    fn default() -> Self {
        Self::ALL
    }
}

/// Per-entity collision filtering: the groups a collider is in, and whether it is
/// a sensor (detects overlaps but never pushes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CollisionFilter {
    pub groups: CollisionGroups,
    pub sensor: bool,
}

impl Default for CollisionFilter {
    fn default() -> Self {
        Self {
            groups: CollisionGroups::ALL,
            sensor: false,
        }
    }
}

/// Continuous collision detection for fast, small bodies that would otherwise
/// pass through a thin wall between two steps.
///
/// Rapier already runs CCD for any fast dynamic body against fixed colliders;
/// `enabled` widens the sweep to every body, which is what a kinematic or moving
/// wall needs. `prediction` is soft CCD, a cheaper predictive pass for thin
/// slow-moving objects; zero disables it.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Ccd {
    pub enabled: bool,
    pub prediction: f32,
}
