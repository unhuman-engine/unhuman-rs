//! The 3D world: bodies in, transforms out.
//!
//! Sync is per step, both directions, in that order: scene -> Rapier, step,
//! Rapier -> scene. Bodies, joints and collision filtering all live in the scene
//! as components; this owns the Rapier sets and reconciles them.
//!
//! Two markers make an unchanged body cost nothing and a user edit survive:
//!
//! * `Entry::source`: everything the body was built from. Equal to the scene's
//!   current description means the body is left alone; a difference rebuilds it.
//! * `Entry::written`: the pose this backend last put in the scene. A scene pose
//!   that disagrees is an editor edit, and it is pushed into Rapier rather than
//!   overwritten by the next step.

use std::collections::HashMap;

use adam::PhysicsBackend;
use glam::Vec3;
use meats::{Entity, RigidBodyKind, Scene};
use rapier3d::prelude::{
    BroadPhaseBvh, CCDSolver, ColliderSet, ImpulseJointSet, IntegrationParameters, IslandManager,
    MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodyHandle, RigidBodySet, SoftBodySet,
};

use crate::body::{self, Entry, ScenePose, Source};
use crate::joints::{self, JointEntry};
use crate::shape::Refusal;

/// Jolt's default gravity, which the 3D world used. Box2D's `-9.8` is the 2D
/// value, and this backend is 3D only.
pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// A body in the Rapier world.
///
/// The id stays in the backend and the scene component stays a description, so a
/// serialised scene never carries a stale index, and `meats` stays free of any
/// dependency on `rapier3d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyHandle(RigidBodyHandle);

/// Rigid-body simulation on Rapier.
pub struct Physics3D {
    pipeline: PhysicsPipeline,
    params: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    soft_bodies: SoftBodySet,
    ccd_solver: CCDSolver,
    entries: HashMap<Entity, Entry>,
    joints: HashMap<Entity, JointEntry>,
    generation: u64,
}

impl Physics3D {
    pub fn new() -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            // One fixed parameter set for the world's whole life: the step length
            // is the only thing `step` touches. That, and one pipeline, is what
            // makes two runs of the same scene produce the same numbers: Rapier
            // holds no generator of its own, so there is nothing else to seed.
            params: IntegrationParameters::default(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            soft_bodies: SoftBodySet::new(),
            ccd_solver: CCDSolver::new(),
            entries: HashMap::new(),
            joints: HashMap::new(),
            generation: 0,
        }
    }

    /// The body an entity currently has, if any.
    ///
    /// The editor wants this to show a body's state and a game wants it to apply
    /// an impulse; both go through a handle rather than through a scene
    /// component, so a scene file never records a body a reload cannot reproduce.
    pub fn handle(&self, entity: Entity) -> Option<BodyHandle> {
        Some(BodyHandle(self.entries.get(&entity)?.body?))
    }

    fn sync_in(&mut self, scene: &Scene) {
        self.generation += 1;
        let generation = self.generation;

        for entity in scene.entities() {
            let Some((source, pose)) = body::read_source(scene, entity) else {
                continue;
            };

            // Two markers, one decision: an equal source means keep the body, a
            // different pose means the scene moved it (an editor edit, or a
            // script), and neither means there is nothing to do at all. The push
            // is deferred because it needs `self.bodies` while `entry` borrows
            // `self.entries`.
            let mut rebuild = false;
            let mut push = None;
            if let Some(entry) = self.entries.get_mut(&entity) {
                if entry.source == source {
                    entry.seen = generation;
                    if entry.written != pose {
                        entry.written = pose;
                        push = entry.body.map(|handle| (handle, pose));
                    }
                } else {
                    rebuild = true;
                }
            } else {
                rebuild = true;
            }

            if rebuild {
                self.rebuild(entity, source, pose, generation);
            } else if let Some((handle, pose)) = push {
                body::push_pose(&mut self.bodies, handle, pose);
            }
        }

        let dead: Vec<Entity> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.seen != generation)
            .map(|(entity, _)| *entity)
            .collect();
        for entity in dead {
            self.despawn(entity);
        }

        joints::sync(
            scene,
            &self.entries,
            &mut self.joints,
            &mut self.impulse_joints,
        );
    }

    fn sync_out(&mut self, scene: &mut Scene) {
        for entity in scene.entities() {
            let Some(entry) = self.entries.get_mut(&entity) else {
                continue;
            };
            // Static bodies are never moved by the simulation, and skipping them
            // leaves a static body's pose bit-identical to what the scene asked
            // for.
            if entry.source.kind == RigidBodyKind::Static {
                continue;
            }
            let Some(handle) = entry.body else {
                continue;
            };
            let Some(pose) = body::pose_of(&self.bodies, handle) else {
                continue;
            };
            entry.written = pose;
            write_pose(scene, entity, pose);
        }
    }

    fn rebuild(&mut self, entity: Entity, source: Source, pose: ScenePose, generation: u64) {
        self.despawn(entity);

        match body::insert(&mut self.bodies, &mut self.colliders, &source, pose) {
            Ok(handle) => {
                self.entries.insert(
                    entity,
                    Entry {
                        body: Some(handle),
                        source,
                        written: pose,
                        seen: generation,
                    },
                );
            }
            Err(Refusal::Unsupported(name)) => {
                log::warn!("femur: entity {entity:?}: collider shape {name} is unsupported");
                self.track_refused(entity, source, pose, generation);
            }
            Err(Refusal::Missing) => {
                log::debug!("femur: entity {entity:?}: rigid body with no collider; no body");
                self.track_refused(entity, source, pose, generation);
            }
        }
    }

    /// Tracks an entity whose collider was refused, so the refusal is reported
    /// when the description changes rather than once per frame.
    fn track_refused(&mut self, entity: Entity, source: Source, pose: ScenePose, seen: u64) {
        self.entries.insert(
            entity,
            Entry {
                body: None,
                source,
                written: pose,
                seen,
            },
        );
    }

    fn despawn(&mut self, entity: Entity) {
        let Some(entry) = self.entries.remove(&entity) else {
            return;
        };
        let Some(handle) = entry.body else {
            return;
        };
        // `true`: an entity that lost its `RigidBody` should not leave its
        // collider in the world. Attached impulse joints go with it.
        self.bodies.remove(
            handle,
            &mut self.islands,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.soft_bodies,
            true,
        );
    }
}

impl Default for Physics3D {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsBackend for Physics3D {
    fn name(&self) -> &'static str {
        "rapier3d"
    }

    fn step(&mut self, scene: &mut Scene, dt: f32) {
        self.sync_in(scene);

        // The engine's accumulator decides the rate; this only takes what it is
        // given, clamped at zero so a bad delta cannot run time backwards (Rapier
        // treats a zero step as "apply the changes, detect collisions, advance
        // nothing").
        self.params.dt = dt.max(0.0);

        // Rapier's solver already generates no constraint between two fixed
        // bodies, so no collision layers are needed to exclude that pair.
        self.pipeline.step(
            GRAVITY,
            &self.params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.soft_bodies,
            &mut self.ccd_solver,
            &(),
            &(),
        );

        self.sync_out(scene);
    }

    fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

/// Writes the simulated pose back, leaving the scale alone.
///
/// The scene stores Euler angles, so both directions go through the one
/// convention `meats` defines.
fn write_pose(scene: &mut Scene, entity: Entity, pose: ScenePose) {
    let Some(mut built) = scene.get_mut(entity) else {
        return;
    };
    let Some(transform) = built.get_mut::<meats::Transform>() else {
        return;
    };
    transform.translation = pose.translation;
    transform.rotation = pose.rotation;
}
