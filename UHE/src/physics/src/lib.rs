//! Physics: the rigid-body backend, as a Plugin.
//!
//! Rigid bodies, colliders, joints and collision filtering run on Rapier.
//! Everything a scene carries is a `meats` component; joints, collision groups,
//! allowed axes, per-body CCD and the collision queue are this crate's own (see
//! `components` and `contacts`), because `meats` has no field for them.
//!
//! # Decisions the rewrite made
//!
//! * **Units.** The metre is canonical. A scene is metres, Rapier is SI metres,
//!   so there is no scene-to-engine conversion. A model has its own scale, which
//!   converts once, through `Transform::scale`, at the shape (`shape::scaled_shape`).
//!   There is deliberately no global physics scale: the scene is already metres,
//!   and a second scale would be a second thing to keep consistent.
//! * **One fixed step.** `adam::Time` owns the accumulator and the fixed rate;
//!   [`Physics3D::step`] consumes the `dt` it is handed and integrates once. The
//!   backend has no accumulator of its own, so there is one clock and one rate.
//! * **Determinism.** A stepped frame is reproducible for a fixed build: Rapier
//!   is bit-deterministic, this backend holds no generator, and the step runs on
//!   one thread. It breaks on a different build (Rapier or SIMD width), on a
//!   different `dt`, and on any attempt to step in parallel or from more than one
//!   thread at once. The `two_identical_runs_agree_bit_for_bit` test is the claim.
//! * **The `glam`/Rapier boundary is one file.** `convert` is the only module
//!   that names a Rapier math type; it has tests in both directions.
//! * **Collisions are messages.** `contacts` diffs touching pairs and queues
//!   enter, contact and exit events; the engine drains them into `pings`. The
//!   backend does not own an event bus.
//! * **Scene data and runtime state.** `Transform`, `RigidBody` and `Collider`
//!   are scene data. `Ccd`, `CollisionFilter`, `AllowedDOFs` and `Joint` are
//!   authored physics data. `BodyHandle`, the touching set and the collision
//!   queue are runtime state, and a save never contains them.
//!
//! `tests/` is the gate: gravity over N steps, a box coming to rest, two identical
//! runs agreeing bit for bit, a static body that does not move, joints holding
//! bodies together, a collision group excluding a contact, CCD stopping a fast
//! body at a thin wall, a constrained body staying constrained, a sleeping body
//! keeping the scene in sync, and the file's shape being the simulated shape.

mod body;
mod components;
mod contacts;
mod convert;
mod debug;
mod filter;
mod joints;
mod plugin;
mod shape;
mod world;

pub use components::{AllowedDOFs, Ccd, CollisionFilter, CollisionGroups, Joint, JointKind};
pub use contacts::{CollisionEvent, CollisionKind};
pub use debug::{DebugLine, PhysicsDebug};
pub use plugin::FemurPlugin;
pub use shape::ColliderShape;
pub use world::{BodyHandle, Physics3D, GRAVITY};
