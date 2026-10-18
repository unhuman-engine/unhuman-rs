//! Physics: the rigid-body backend, as a Plugin.
//!
//! Rigid bodies, colliders, joints and collision filtering run on Rapier.
//! Everything a scene carries is a `meats` component; joints, collision groups
//! and per-body CCD are components this crate defines (see `components`), because
//! `meats` has no field for them.
//!
//! `tests/` is the gate: gravity over N steps, a box coming to rest, two identical
//! runs agreeing bit for bit, a static body that does not move, joints holding
//! bodies together, a collision group excluding a contact, and CCD stopping a fast
//! body at a thin wall.

mod body;
mod components;
mod convert;
mod filter;
mod joints;
mod plugin;
mod shape;
mod world;

pub use components::{Ccd, CollisionFilter, CollisionGroups, Joint, JointKind};
pub use plugin::FemurPlugin;
pub use shape::ColliderShape;
pub use world::{BodyHandle, Physics3D, GRAVITY};
