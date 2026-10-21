//! `meats` - the scene crate.
//!
//! An ECS on `hecs`, a single-parent hierarchy with a per-frame world-transform
//! cache, the component set, and the `.unhuman` YAML serializer.
//!
//! `.unhuman` is the one scene document: a save and a scene asset are the same
//! text. A cooked binary form would be a derived import, not a second format;
//! see [`serialize`].
//!
//! Two deliberate differences:
//!
//! * `Entity` is this crate's own generational handle rather than a bare
//!   `hecs::Entity`, so a handle taken before a destroy is refused instead of
//!   addressing whatever now occupies the slot.
//! * `NativeScript` holds a `Box<dyn Script>`. Function pointers plus a raw
//!   `ScriptableEntity*` would own a lifetime nothing here needs, so this crate
//!   needs no `unsafe`.

pub mod camera;
pub mod components;
pub mod entity;
pub mod scene;
pub mod serialize;

pub use camera::{Camera, ProjectionType, SceneCamera};
pub use components::{
    Animator, AnimatorState, AudioSource, Collider, ColliderShape, Id, Light, LightKind,
    MeshRenderer, NativeScript, RigidBody, RigidBodyKind, Script, SpriteAnimation, SpriteRenderer,
    Tag, Text, Transform,
};
pub use entity::{Entity, EntityRef, EntityRefMut};
pub use scene::{euler_degrees, AssetError, LightSample, MeshSample, Scene};
pub use serialize::{from_yaml, load, save, to_yaml, Error, Result, FORMAT_VERSION};

/// `hecs` is re-exported so a backend can name `hecs::Entity` in a query through
/// [`Scene::world`] without taking its own dependency on the ECS.
pub use hecs;
pub use hecs::{Ref, RefMut};
