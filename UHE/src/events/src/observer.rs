//! ECS observers, tied to the real `meats` scene.
//!
//! The seam `meats` should expose: call [`Observers::entity_created`],
//! [`Observers::entity_destroyed`], [`Observers::component_added`] and
//! [`Observers::component_removed`] from `Scene::spawn`, `Scene::destroy`,
//! `Scene::insert` and `Scene::remove`. `meats` does not call them yet, and
//! `pings` must not be a dependency of `meats`, so [`ObservedScene`] is that
//! seam on this side: it owns a real `Scene` and forwards those four operations
//! through the registry. Once `Scene` calls the hooks itself, `ObservedScene`
//! can be dropped and callers use `Scene` directly.
//!
//! `hecs` is reached through `meats`'s re-export, so `pings` takes no direct
//! dependency on the ECS.

use std::any::TypeId;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use meats::{Entity, Scene};

/// Lifecycle and component hooks for one entity.
///
/// Every method defaults to a no-op, so an observer overrides only what it
/// cares about. The component hooks carry a [`TypeId`], not a generic parameter:
/// a generic method would not be object safe, and the registry stores trait
/// objects.
pub trait Observer: Send + Sync + 'static {
    fn on_created(&mut self, _scene: &Scene, _entity: Entity) {}

    fn on_destroyed(&mut self, _scene: &Scene, _entity: Entity) {}

    fn on_component_added(&mut self, _scene: &Scene, _entity: Entity, _component: TypeId) {}

    fn on_component_removed(&mut self, _scene: &Scene, _entity: Entity, _component: TypeId) {}
}

struct Slot {
    alive: Arc<AtomicBool>,
    observer: Box<dyn Observer>,
}

/// Stops its observer seeing hooks once dropped. Dropping it during a hook is
/// safe: the observer is called from a snapshot list, and compaction happens on
/// the next hook.
pub struct ObserverHandle {
    alive: Arc<AtomicBool>,
}

impl Drop for ObserverHandle {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
    }
}

/// The live observers, invoked in subscription order.
#[derive(Default)]
pub struct Observers {
    slots: Vec<Slot>,
}

impl Observers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, observer: impl Observer) -> ObserverHandle {
        let alive = Arc::new(AtomicBool::new(true));
        self.slots.push(Slot {
            alive: Arc::clone(&alive),
            observer: Box::new(observer),
        });
        ObserverHandle { alive }
    }

    pub fn entity_created(&mut self, scene: &Scene, entity: Entity) {
        self.live();
        for slot in &mut self.slots {
            slot.observer.on_created(scene, entity);
        }
    }

    pub fn entity_destroyed(&mut self, scene: &Scene, entity: Entity) {
        self.live();
        for slot in &mut self.slots {
            slot.observer.on_destroyed(scene, entity);
        }
    }

    pub fn component_added(&mut self, scene: &Scene, entity: Entity, component: TypeId) {
        self.live();
        for slot in &mut self.slots {
            slot.observer.on_component_added(scene, entity, component);
        }
    }

    pub fn component_removed(&mut self, scene: &Scene, entity: Entity, component: TypeId) {
        self.live();
        for slot in &mut self.slots {
            slot.observer.on_component_removed(scene, entity, component);
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Drops the handlers whose handle is gone; a push does the same for the
    /// bus, so the two registries age the same way.
    fn live(&mut self) {
        self.slots.retain(|slot| slot.alive.load(Ordering::Acquire));
    }
}

/// A [`Scene`] whose four structural operations notify [`Observers`].
///
/// This is the integration that works against `meats` today without `meats`
/// depending on `pings`. Forwarding is deliberate and narrow: only spawn,
/// destroy, insert and remove can change what an observer watches.
pub struct ObservedScene {
    scene: Scene,
    observers: Observers,
}

impl ObservedScene {
    pub fn new() -> Self {
        Self {
            scene: Scene::new(),
            observers: Observers::new(),
        }
    }

    pub fn with_scene(scene: Scene) -> Self {
        Self {
            scene,
            observers: Observers::new(),
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub fn observers(&self) -> &Observers {
        &self.observers
    }

    pub fn observers_mut(&mut self) -> &mut Observers {
        &mut self.observers
    }

    pub fn spawn(&mut self, name: &str) -> Entity {
        let entity = self.scene.spawn(name);
        self.observers.entity_created(&self.scene, entity);
        entity
    }

    /// Notifies every doomed entity before the subtree comes down, so an
    /// observer can still read the components it is being told about.
    pub fn destroy(&mut self, entity: Entity) {
        for doomed in self.subtree(entity) {
            if self.scene.contains(doomed) {
                self.observers.entity_destroyed(&self.scene, doomed);
            }
        }
        self.scene.destroy(entity);
    }

    /// Fires `component_added` on a replace too: `Scene::insert` cannot tell an
    /// overwrite from a first insert, so the hook means "this is now present".
    pub fn insert<T: meats::hecs::Component>(&mut self, entity: Entity, component: T) -> bool {
        let inserted = self.scene.insert(entity, component);
        if inserted {
            self.observers
                .component_added(&self.scene, entity, TypeId::of::<T>());
        }
        inserted
    }

    pub fn remove<T: meats::hecs::Component>(&mut self, entity: Entity) -> Option<T> {
        let removed = self.scene.remove::<T>(entity);
        if removed.is_some() {
            self.observers
                .component_removed(&self.scene, entity, TypeId::of::<T>());
        }
        removed
    }

    fn subtree(&self, root: Entity) -> Vec<Entity> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if !self.scene.contains(entity) || !seen.insert(entity) {
                continue;
            }
            out.push(entity);
            stack.extend_from_slice(self.scene.children(entity));
        }
        out
    }
}

impl Default for ObservedScene {
    fn default() -> Self {
        Self::new()
    }
}
