//! Rigid bodies and colliders: the scene's description, and the Rapier body it
//! builds.
//!
//! `Source` is everything about an entity that decides the body it gets; an equal
//! `Source` means the body is left alone, a different one rebuilds it.
//! `ScenePose` is kept as the scene's own translation and Euler degrees, so the
//! comparison that catches an editor edit stays bit-equal for an untouched body.

use glam::Vec3;
use meats::{Collider, Entity, RigidBody, RigidBodyKind, Scene, Transform};
use rapier3d::prelude::{
    ColliderBuilder, ColliderSet, RigidBodyBuilder, RigidBodyHandle, RigidBodySet,
};

use crate::components::{Ccd, CollisionFilter};
use crate::convert;
use crate::filter;
use crate::shape::{Refusal, ShapeSlot};

const DEFAULT_FRICTION: f32 = 0.2;
const DEFAULT_RESTITUTION: f32 = 0.0;

/// A pose as the scene stores one: a translation, and `Transform::rotation` Euler
/// degrees.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct ScenePose {
    pub translation: Vec3,
    pub rotation: Vec3,
}

/// Everything about an entity that decides the body it gets. A change rebuilds it.
#[derive(PartialEq)]
pub(crate) struct Source {
    pub kind: RigidBodyKind,
    pub mass: f32,
    pub velocity: Vec3,
    pub damping: f32,
    pub gravity_scale: f32,
    pub friction: f32,
    pub restitution: f32,
    pub offset: Vec3,
    pub shape: ShapeSlot,
    pub ccd: Ccd,
    pub filter: CollisionFilter,
}

/// A body in the world, and the pose the backend last wrote into the scene.
pub(crate) struct Entry {
    /// `None` when the entity's collider was refused; the entity stays tracked so
    /// the refusal is reported when the description changes.
    pub body: Option<RigidBodyHandle>,
    /// What the body was built from.
    pub source: Source,
    /// The pose this backend last wrote into the scene.
    pub written: ScenePose,
    /// The sync pass that last saw the entity's rigid body.
    pub seen: u64,
}

/// The physics view of an entity: its body description, and where it currently is.
pub(crate) fn read_source(scene: &Scene, entity: Entity) -> Option<(Source, ScenePose)> {
    let built = scene.get(entity)?;
    let rigid_body = built.get::<RigidBody>()?;
    let transform = built.get::<Transform>()?;

    let pose = ScenePose {
        translation: transform.translation,
        rotation: transform.rotation,
    };

    let collider = built.get::<Collider>();
    let source = Source {
        kind: rigid_body.kind,
        mass: rigid_body.mass,
        velocity: rigid_body.velocity,
        damping: rigid_body.damping,
        gravity_scale: rigid_body.gravity_scale,
        friction: collider.as_ref().map_or(DEFAULT_FRICTION, |c| c.friction),
        restitution: collider
            .as_ref()
            .map_or(DEFAULT_RESTITUTION, |c| c.restitution),
        // The component's `Offset` becomes the collider's position on the body,
        // which is what the field is for.
        offset: collider.as_ref().map_or(Vec3::ZERO, |c| c.offset),
        shape: ShapeSlot::from_collider(collider.as_deref(), transform.scale),
        ccd: built.get::<Ccd>().map_or(Ccd::default(), |ccd| *ccd),
        filter: built
            .get::<CollisionFilter>()
            .map_or(CollisionFilter::default(), |filter| *filter),
    };
    Some((source, pose))
}

/// Builds the Rapier body and collider for `source`, or refuses it by name.
pub(crate) fn insert(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    source: &Source,
    pose: ScenePose,
) -> Result<RigidBodyHandle, Refusal> {
    let shape = source.shape.shared()?;

    let body = match source.kind {
        RigidBodyKind::Static => RigidBodyBuilder::fixed(),
        RigidBodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
        RigidBodyKind::Dynamic => RigidBodyBuilder::dynamic(),
    }
    .pose(convert::rapier_pose(pose.translation, pose.rotation))
    .linvel(source.velocity)
    .gravity_scale(source.gravity_scale)
    // One `damping` on the component drives both linear and angular damping.
    .linear_damping(source.damping)
    .angular_damping(source.damping)
    // Fast bodies already sweep fixed colliders automatically; `enabled` widens
    // the sweep to every body, `prediction` is the cheaper soft-CCD distance.
    .ccd_enabled(source.ccd.enabled)
    .soft_ccd_prediction(source.ccd.prediction);
    let handle = bodies.insert(body);

    let mut collider = ColliderBuilder::new(shape)
        .friction(source.friction)
        .restitution(source.restitution)
        .translation(source.offset)
        .collision_groups(filter::interaction_groups(&source.filter.groups))
        .sensor(source.filter.sensor);
    if source.kind == RigidBodyKind::Dynamic {
        // Jolt set the body's mass from the component and left inertia to the
        // shape; a Rapier collider mass is the same split.
        collider = collider.mass(source.mass);
    }
    colliders.insert_with_parent(collider, handle, bodies);

    Ok(handle)
}

/// Pushes a scene pose into a body, waking it: a user edit should not be ignored.
pub(crate) fn push_pose(bodies: &mut RigidBodySet, handle: RigidBodyHandle, pose: ScenePose) {
    if let Some(body) = bodies.get_mut(handle) {
        body.set_position(convert::rapier_pose(pose.translation, pose.rotation), true);
    }
}

/// A body's pose in the scene's own representation.
pub(crate) fn pose_of(bodies: &RigidBodySet, handle: RigidBodyHandle) -> Option<ScenePose> {
    let body = bodies.get(handle)?;
    let (translation, rotation) = convert::scene_pose(body.position());
    Some(ScenePose {
        translation,
        rotation,
    })
}
