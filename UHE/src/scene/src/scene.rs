//! The scene: a `hecs::World` plus the single-parent hierarchy, the generational
//! entity slot map, and the per-frame world-transform cache.
//!
//! Renderer, physics world and editor state are not part of it: those are
//! plugins here.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use hecs::World;

use crate::camera::Camera;
use crate::components::{
    Animator, AnimatorState, AudioSource, Collider, Id, Light, LightKind, MeshRenderer,
    NativeScript, RigidBody, SpriteAnimation, SpriteRenderer, Tag, Text, Transform,
};
use crate::entity::{Entity, EntityRef, EntityRefMut};

/// Cycles are refused on the way in, but a loaded file can still contain one,
/// so the walks cap their depth rather than spin forever.
const MAX_DEPTH: usize = 4096;

/// A light as a renderer wants it: world-space, with the direction the entity's
/// rotation gives local -Z.
#[derive(Clone, Copy, Debug)]
pub struct LightSample {
    pub entity: Entity,
    pub kind: LightKind,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    pub position: Vec3,
    pub direction: Vec3,
}

/// A mesh as a renderer wants it.
#[derive(Clone, Copy, Debug)]
pub struct MeshSample {
    pub entity: Entity,
    pub mesh: vault::AssetId,
    pub material: Option<vault::AssetId>,
    pub tint: glam::Vec4,
    pub transform: Mat4,
    pub animator: Option<AnimatorState>,
    pub visible: bool,
    pub cast_shadows: bool,
}

/// A path the registry could not resolve. Recorded, never a panic: a scene that
/// references a deleted asset still loads and still renders the rest.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AssetError {
    pub entity: Entity,
    pub path: String,
}

impl std::fmt::Display for AssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unresolved asset `{}` on {:?}", self.path, self.entity)
    }
}

impl std::error::Error for AssetError {}

/// Copies every cloneable component from one entity to another, leaving
/// `NativeScript` behind. A macro because it is a dozen near-identical
/// statements and a trait would need a bound every component would have to
/// carry.
macro_rules! copy_components {
    ($source:expr, $target:expr, $from:expr, $to:expr, $($ty:ty),* $(,)?) => {
        $(
            if let Ok(component) = $source.world.get::<&$ty>($from) {
                let _ = $target.world.insert_one($to, duplicate(&*component));
            }
        )*
    };
}

/// Clones through a generic so `copy_components!` works for the `Copy`
/// components too without a `clone_on_copy` lint at every expansion.
fn duplicate<T: Clone>(value: &T) -> T {
    value.clone()
}

pub struct Scene {
    world: World,
    /// `index -> hecs entity`, `None` once freed.
    slots: Vec<Option<hecs::Entity>>,
    /// `index -> generation`, bumped on reuse so a stale handle is refused.
    generations: Vec<u32>,
    free: Vec<u32>,
    /// Reverse of `slots`, so a `hecs::Entity` from a raw query maps back.
    entities: HashMap<hecs::Entity, Entity>,
    parents: HashMap<Entity, Entity>,
    children: HashMap<Entity, Vec<Entity>>,
    /// `Id` -> `Entity`. Hierarchy and the serializer resolve ids through this
    /// instead of scanning components.
    id_index: HashMap<u64, Entity>,
    /// Rebuilt by [`Scene::update_transforms`]; consulted by
    /// [`Scene::world_transform`] and invalidated by every structural change.
    world_cache: HashMap<Entity, Mat4>,
    /// Component names a loaded file carried but this build does not implement.
    /// Kept per entity so the next save can name what it dropped.
    dropped: HashMap<Entity, Vec<String>>,
    viewport: (u32, u32),
    name: String,
    next_id: u64,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            slots: Vec::new(),
            generations: Vec::new(),
            free: Vec::new(),
            entities: HashMap::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
            id_index: HashMap::new(),
            world_cache: HashMap::new(),
            dropped: HashMap::new(),
            viewport: (0, 0),
            name: "Untitled".to_string(),
            next_id: seed_id(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    pub fn viewport_size(&self) -> (u32, u32) {
        self.viewport
    }

    // --- entities ---------------------------------------------------------

    /// Spawns an entity with an `Id`, a `Tag` and a `Transform`.
    pub fn spawn(&mut self, name: &str) -> Entity {
        let id = self.next_unique_id();
        let tag = if name.is_empty() { "Entity" } else { name };
        let hecs = self
            .world
            .spawn((Id(id), Tag::new(tag), Transform::default()));
        let entity = self.register(hecs);
        self.id_index.insert(id, entity);
        self.world_cache.clear();
        entity
    }

    /// Destroys the whole subtree, and every script gets `on_destroy` while its
    /// components are still readable.
    pub fn destroy(&mut self, entity: Entity) {
        let mut doomed = Vec::new();
        self.collect_subtree(entity, &mut doomed);
        // Children first, so a parent is never destroyed before its child.
        for doomed in doomed.into_iter().rev() {
            self.destroy_one(doomed);
        }
    }

    pub fn clear(&mut self) {
        self.destroy_all_scripts();
        self.world.clear();
        self.slots.clear();
        self.generations.clear();
        self.free.clear();
        self.entities.clear();
        self.parents.clear();
        self.children.clear();
        self.id_index.clear();
        self.world_cache.clear();
        self.dropped.clear();
    }

    pub fn contains(&self, entity: Entity) -> bool {
        self.hecs_of(entity).is_some()
    }

    /// Every live entity, ordered by slot index so two calls with no mutation in
    /// between agree; a render graph that sees a different pass list each frame
    /// is a render graph that cannot compile.
    pub fn entities(&self) -> Vec<Entity> {
        let mut entities: Vec<Entity> = self.entities.values().copied().collect();
        entities.sort_unstable_by_key(|entity| (entity.index, entity.generation));
        entities
    }

    pub fn get(&self, entity: Entity) -> Option<EntityRef<'_>> {
        self.contains(entity).then_some(EntityRef {
            scene: self,
            entity,
        })
    }

    pub fn get_mut(&mut self, entity: Entity) -> Option<EntityRefMut<'_>> {
        self.contains(entity).then_some(EntityRefMut {
            scene: self,
            entity,
        })
    }

    pub fn id_of(&self, entity: Entity) -> Option<u64> {
        self.id_component(entity).map(|id| id.0)
    }

    pub fn entity_of_id(&self, id: u64) -> Option<Entity> {
        self.id_index
            .get(&id)
            .copied()
            .filter(|entity| self.contains(*entity))
    }

    pub fn entity_of(&self, hecs: hecs::Entity) -> Option<Entity> {
        self.entities.get(&hecs).copied()
    }

    /// Component names a file carried that this build skipped while loading.
    /// A save writes them back under `DroppedComponents` so the loss is visible.
    pub fn dropped_components(&self, entity: Entity) -> &[String] {
        self.dropped.get(&entity).map(Vec::as_slice).unwrap_or(&[])
    }

    pub(crate) fn record_dropped_component(&mut self, entity: Entity, name: &str) {
        let names = self.dropped.entry(entity).or_default();
        if !names.iter().any(|seen| seen == name) {
            names.push(name.to_string());
        }
    }

    /// The raw world, for a backend that needs a multi-component query
    /// (`world.query_mut::<(&mut Transform, &RigidBody)>()`).
    ///
    /// Read and write components through it freely, but never spawn or despawn:
    /// the slot map, the hierarchy and the id index are [`Scene`]'s, and an
    /// entity created behind its back has no entry in any of them.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// See [`world`](Self::world).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    // --- components -------------------------------------------------------

    pub fn has<T: hecs::Component>(&self, entity: Entity) -> bool {
        match self.hecs_of(entity) {
            Some(hecs) => self.world.satisfies::<&T>(hecs),
            None => false,
        }
    }

    /// Add or replace a component. A fresh camera is handed the viewport it was
    /// created in.
    pub fn insert<T: hecs::Component>(&mut self, entity: Entity, mut component: T) -> bool {
        let Some(hecs) = self.hecs_of(entity) else {
            return false;
        };
        let any: &mut dyn std::any::Any = &mut component;
        if let Some(camera) = any.downcast_mut::<Camera>() {
            // A camera that pins its aspect does not take the viewport's, the
            // same rule `set_viewport_size` applies on a resize.
            if !camera.fixed_aspect_ratio {
                camera
                    .camera
                    .set_viewport_size(self.viewport.0 as f32, self.viewport.1 as f32);
            }
        }
        self.world.insert_one(hecs, component).is_ok()
    }

    /// Removing a script runs its `on_destroy` first, so the instance is not
    /// dropped without notice.
    pub fn remove<T: hecs::Component>(&mut self, entity: Entity) -> Option<T> {
        let hecs = self.hecs_of(entity)?;
        let mut removed = self.world.remove_one::<T>(hecs).ok()?;
        let any: &mut dyn std::any::Any = &mut removed;
        if let Some(script) = any.downcast_mut::<NativeScript>() {
            script.script.on_destroy(entity, self);
        }
        Some(removed)
    }

    // --- hierarchy --------------------------------------------------------

    /// Attach `child` under `parent` keeping its **local** transform. A cycle is
    /// refused, with a warning.
    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>) {
        self.attach(child, parent, false);
    }

    /// Keep the entity where it is in the world while switching parents, which
    /// is what an editor drag-drop wants.
    pub fn set_parent_preserving_world(&mut self, child: Entity, parent: Option<Entity>) {
        self.attach(child, parent, true);
    }

    pub fn parent(&self, entity: Entity) -> Option<Entity> {
        self.parents.get(&entity).copied()
    }

    pub fn children(&self, entity: Entity) -> &[Entity] {
        self.children.get(&entity).map(Vec::as_slice).unwrap_or(&[])
    }

    /// True when `entity` is a strict descendant of `ancestor`.
    pub fn is_descendant_of(&self, entity: Entity, ancestor: Entity) -> bool {
        let mut cursor = entity;
        for _ in 0..MAX_DEPTH {
            match self.parents.get(&cursor).copied() {
                Some(parent) if parent == ancestor => return true,
                Some(parent) => cursor = parent,
                None => return false,
            }
        }
        false
    }

    // --- transforms -------------------------------------------------------

    /// World matrix for `entity`: the cached one if [`Scene::update_transforms`]
    /// has run since the last structural change, otherwise a walk up the parent
    /// chain multiplying the local matrices back down.
    pub fn world_transform(&self, entity: Entity) -> Mat4 {
        match self.world_cache.get(&entity) {
            Some(world) => *world,
            None => self.walk_world_transform(entity),
        }
    }

    /// Called once per frame; a second call with no mutation in between
    /// produces the same cache.
    pub fn update_transforms(&mut self) {
        self.world_cache.clear();

        let mut roots = Vec::new();
        for entity in self.entities() {
            match self.parents.get(&entity).copied() {
                Some(parent) if self.contains(parent) => {}
                _ => roots.push(entity),
            }
        }

        // Iterative so a deep glTF hierarchy cannot blow the stack.
        let mut stack: Vec<(Entity, Mat4)> = roots
            .into_iter()
            .rev()
            .map(|entity| (entity, Mat4::IDENTITY))
            .collect();
        while let Some((entity, parent_world)) = stack.pop() {
            let world = parent_world * self.local_matrix(entity);
            self.world_cache.insert(entity, world);
            if let Some(kids) = self.children.get(&entity) {
                for child in kids.iter().rev() {
                    if self.contains(*child) {
                        stack.push((*child, world));
                    }
                }
            }
        }
    }

    /// Cameras that do not pin their aspect take the new viewport size.
    pub fn set_viewport_size(&mut self, width: u32, height: u32) {
        self.viewport = (width, height);
        for entity in self.entities() {
            let Some(hecs) = self.hecs_of(entity) else {
                continue;
            };
            if let Ok(mut camera) = self.world.get::<&mut Camera>(hecs) {
                if !camera.fixed_aspect_ratio {
                    camera.camera.set_viewport_size(width as f32, height as f32);
                }
            }
        }
    }

    // --- queries ----------------------------------------------------------

    /// The first entity carrying a primary [`Camera`].
    pub fn primary_camera(&self) -> Option<Entity> {
        self.entities().into_iter().find(|entity| {
            self.hecs_of(*entity)
                .and_then(|hecs| self.world.get::<&Camera>(hecs).ok())
                .is_some_and(|camera| camera.primary)
        })
    }

    /// Extracts the scene's own light components. Disabled lights and entities
    /// with no transform are skipped; the direction is the entity's world
    /// rotation applied to local -Z, the light axis.
    pub fn lights(&self) -> Vec<LightSample> {
        let mut lights = Vec::new();
        for entity in self.entities() {
            let Some(hecs) = self.hecs_of(entity) else {
                continue;
            };
            let Ok(light) = self.world.get::<&Light>(hecs) else {
                continue;
            };
            if !light.enabled || !self.has::<Transform>(entity) {
                continue;
            }
            let (_, rotation, position) =
                self.world_transform(entity).to_scale_rotation_translation();
            lights.push(LightSample {
                entity,
                kind: light.kind,
                color: light.color,
                intensity: light.intensity,
                radius: light.radius,
                inner_cone_angle: light.inner_cone_angle,
                outer_cone_angle: light.outer_cone_angle,
                position,
                direction: (rotation * Vec3::NEG_Z).normalize_or_zero(),
            });
        }
        lights
    }

    /// The meshes a renderer should submit, in slot order.
    pub fn meshes(&self) -> Vec<MeshSample> {
        let mut meshes = Vec::new();
        for entity in self.entities() {
            let Some(hecs) = self.hecs_of(entity) else {
                continue;
            };
            let Ok(renderer) = self.world.get::<&MeshRenderer>(hecs) else {
                continue;
            };
            if !renderer.visible {
                continue;
            }
            let Some(mesh) = renderer.mesh else {
                continue;
            };
            let animator = self
                .world
                .get::<&Animator>(hecs)
                .ok()
                .map(|animator| AnimatorState::from(&*animator));
            meshes.push(MeshSample {
                entity,
                mesh,
                material: renderer.material,
                tint: renderer.tint,
                transform: self.world_transform(entity),
                animator,
                visible: renderer.visible,
                cast_shadows: renderer.cast_shadows,
            });
        }
        meshes
    }

    // --- assets -----------------------------------------------------------

    /// Resolve every path a component carries through the registry, recording
    /// what it cannot find. Idempotent: a handle that is already set is left
    /// alone, so calling it twice changes nothing and re-reports the same
    /// missing paths.
    pub fn resolve_assets(&mut self, assets: &mut vault::Assets) -> Vec<AssetError> {
        let mut errors = Vec::new();
        for entity in self.entities() {
            let Some(hecs) = self.hecs_of(entity) else {
                continue;
            };

            if let Ok(mut renderer) = self.world.get::<&mut MeshRenderer>(hecs) {
                let path = renderer.mesh_path.clone();
                resolve(&mut renderer.mesh, &path, assets, &mut errors, entity);
                let path = renderer.material_path.clone();
                resolve(&mut renderer.material, &path, assets, &mut errors, entity);
            }
            if let Ok(mut sprite) = self.world.get::<&mut SpriteRenderer>(hecs) {
                let path = sprite.texture_path.clone();
                resolve(&mut sprite.texture, &path, assets, &mut errors, entity);
            }
            if let Ok(mut text) = self.world.get::<&mut Text>(hecs) {
                let path = text.font_path.clone();
                resolve(&mut text.font, &path, assets, &mut errors, entity);
            }
            if let Ok(mut audio) = self.world.get::<&mut AudioSource>(hecs) {
                let path = audio.clip_path.clone();
                resolve(&mut audio.clip, &path, assets, &mut errors, entity);
            }
            if let Ok(mut animator) = self.world.get::<&mut Animator>(hecs) {
                let path = animator.clip_path.clone();
                resolve(&mut animator.clip, &path, assets, &mut errors, entity);
            }
            if let Ok(mut animation) = self.world.get::<&mut SpriteAnimation>(hecs) {
                let path = animation.sprite_sheet_path.clone();
                resolve(&mut animation.sheet, &path, assets, &mut errors, entity);
            }
        }
        errors
    }

    // --- scripts ----------------------------------------------------------

    /// Runs `on_create` on the first call for an entity, then `on_update`.
    ///
    /// The script is moved out of the world for the duration of the call, because
    /// a hook takes `&mut Scene` and the component lives in that scene.
    pub fn update_scripts(&mut self, dt: f32) {
        for entity in self.entities() {
            let Some(hecs) = self.hecs_of(entity) else {
                continue;
            };
            let Ok(mut script) = self.world.remove_one::<NativeScript>(hecs) else {
                continue;
            };
            if !script.take_created() {
                script.script.on_create(entity, self);
            }
            script.script.on_update(entity, self, dt);
            if self.world.contains(hecs) {
                let _ = self.world.insert_one(hecs, script);
            } else {
                // The hook destroyed its own entity; `destroy_one` found no
                // script to notify, so this is where `on_destroy` belongs.
                script.script.on_destroy(entity, self);
            }
        }
    }

    // --- internals --------------------------------------------------------

    pub(crate) fn hecs_of(&self, entity: Entity) -> Option<hecs::Entity> {
        let slot = self.slots.get(entity.index as usize)?.as_ref()?;
        if self.generations[entity.index as usize] == entity.generation {
            Some(*slot)
        } else {
            None
        }
    }

    /// Restore a serialized `Id`. The serializer owns id assignment because a
    /// file's ids are the ones its hierarchy refers to.
    pub(crate) fn set_id(&mut self, entity: Entity, id: u64) {
        let Some(hecs) = self.hecs_of(entity) else {
            return;
        };
        let previous = self.world.get::<&Id>(hecs).ok().map(|id| id.0);
        if let Ok(mut component) = self.world.get::<&mut Id>(hecs) {
            component.0 = id;
        }
        if let Some(previous) = previous {
            if previous != id {
                self.id_index.remove(&previous);
            }
        }
        self.id_index.insert(id, entity);
    }

    pub(crate) fn id_is_taken(&self, id: u64) -> bool {
        self.id_index.contains_key(&id)
    }

    fn next_unique_id(&mut self) -> u64 {
        loop {
            self.next_id = self.next_id.wrapping_add(1);
            if self.next_id != 0 && !self.id_index.contains_key(&self.next_id) {
                return self.next_id;
            }
        }
    }

    fn register(&mut self, hecs: hecs::Entity) -> Entity {
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                self.slots.push(None);
                self.generations.push(1);
                (self.slots.len() - 1) as u32
            }
        };
        let generation = self.generations[index as usize];
        self.slots[index as usize] = Some(hecs);
        let entity = Entity { index, generation };
        self.entities.insert(hecs, entity);
        entity
    }

    fn release(&mut self, entity: Entity) {
        let index = entity.index as usize;
        if index >= self.slots.len() || self.generations[index] != entity.generation {
            return;
        }
        if let Some(hecs) = self.slots[index].take() {
            self.entities.remove(&hecs);
        }
        self.generations[index] = self.generations[index].wrapping_add(1);
        self.free.push(entity.index);
    }

    fn collect_subtree(&self, root: Entity, out: &mut Vec<Entity>) {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if !self.contains(entity) || !seen.insert(entity) {
                continue;
            }
            out.push(entity);
            if let Some(kids) = self.children.get(&entity) {
                stack.extend(kids.iter().copied());
            }
        }
    }

    fn destroy_one(&mut self, entity: Entity) {
        let Some(hecs) = self.hecs_of(entity) else {
            return;
        };

        let script = self.world.remove_one::<NativeScript>(hecs).ok();
        if let Some(mut script) = script {
            script.script.on_destroy(entity, self);
        }

        if let Some(parent) = self.parents.remove(&entity) {
            detach_child(&mut self.children, parent, entity);
        }
        self.children.remove(&entity);

        if let Some(id) = self.world.get::<&Id>(hecs).ok().map(|id| id.0) {
            self.id_index.remove(&id);
        }
        self.dropped.remove(&entity);

        let _ = self.world.despawn(hecs);
        self.release(entity);
        self.world_cache.clear();
    }

    fn destroy_all_scripts(&mut self) {
        for entity in self.entities() {
            let Some(hecs) = self.hecs_of(entity) else {
                continue;
            };
            let script = self.world.remove_one::<NativeScript>(hecs).ok();
            if let Some(mut script) = script {
                script.script.on_destroy(entity, self);
            }
        }
    }

    fn attach(&mut self, child: Entity, parent: Option<Entity>, preserve_world: bool) {
        if !self.contains(child) {
            return;
        }
        if let Some(parent) = parent {
            if !self.contains(parent) {
                return;
            }
            if parent == child || self.is_descendant_of(parent, child) {
                log::warn!(
                    "meats: refused to parent {child:?} under its own descendant {parent:?}"
                );
                return;
            }
            if self.parents.get(&child) == Some(&parent) {
                return;
            }
        }

        let world = preserve_world.then(|| self.world_transform(child));

        if let Some(previous) = self.parents.remove(&child) {
            detach_child(&mut self.children, previous, child);
        }
        if let Some(parent) = parent {
            self.parents.insert(child, parent);
            self.children.entry(parent).or_default().push(child);
        }
        if let Some(world) = world {
            self.set_local_transform_from_world(child, world);
        }
        self.world_cache.clear();
    }

    /// Rewrites the local transform so the entity keeps the given world matrix.
    fn set_local_transform_from_world(&mut self, entity: Entity, world: Mat4) {
        if !self.has::<Transform>(entity) {
            return;
        }
        let local = match self.parent(entity) {
            Some(parent) => self.world_transform(parent).inverse() * world,
            None => world,
        };
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        let Some(hecs) = self.hecs_of(entity) else {
            return;
        };
        if let Ok(mut transform) = self.world.get::<&mut Transform>(hecs) {
            transform.translation = translation;
            transform.rotation = euler_degrees(rotation);
            transform.scale = scale;
        }
    }

    fn local_matrix(&self, entity: Entity) -> Mat4 {
        match self
            .hecs_of(entity)
            .and_then(|hecs| self.world.get::<&Transform>(hecs).ok())
        {
            Some(transform) => transform.matrix(),
            None => Mat4::IDENTITY,
        }
    }

    fn walk_world_transform(&self, entity: Entity) -> Mat4 {
        if !self.contains(entity) {
            return Mat4::IDENTITY;
        }
        let mut chain = Vec::new();
        let mut cursor = entity;
        for _ in 0..MAX_DEPTH {
            if !self.contains(cursor) {
                break;
            }
            chain.push(cursor);
            match self.parents.get(&cursor).copied() {
                Some(parent) => cursor = parent,
                None => break,
            }
        }
        let mut world = Mat4::IDENTITY;
        for node in chain.iter().rev() {
            world *= self.local_matrix(*node);
        }
        world
    }

    fn id_component(&self, entity: Entity) -> Option<Id> {
        let hecs = self.hecs_of(entity)?;
        self.world.get::<&Id>(hecs).ok().map(|id| *id)
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene")
            .field("name", &self.name)
            .field("entities", &self.entities.len())
            .finish_non_exhaustive()
    }
}

impl Drop for Scene {
    /// A script's `on_destroy` runs even when the scene is simply dropped. Rust
    /// frees the box either way; this is about the callback.
    fn drop(&mut self) {
        self.destroy_all_scripts();
    }
}

impl Clone for Scene {
    /// Used for play-mode snapshots. The hierarchy and every component come
    /// across; `NativeScript` does not, because `Box<dyn Script>` cannot be
    /// cloned, so the copy's runtime instantiates its own.
    fn clone(&self) -> Self {
        let mut clone = Scene::new();
        clone.name = self.name.clone();
        clone.viewport = self.viewport;
        clone.next_id = self.next_id;

        let mut map: HashMap<Entity, Entity> = HashMap::new();
        for entity in self.entities() {
            let Some(source) = self.hecs_of(entity) else {
                continue;
            };
            let hecs = clone.world.spawn(());
            let target = clone.register(hecs);
            // The `Id` goes across verbatim so the hierarchy references below
            // still resolve; it is `Copy`, so it is copied rather than cloned.
            if let Ok(id) = self.world.get::<&Id>(source) {
                let _ = clone.world.insert_one(hecs, *id);
            }
            copy_components!(
                self,
                clone,
                source,
                hecs,
                Tag,
                Transform,
                Camera,
                MeshRenderer,
                SpriteRenderer,
                Text,
                Light,
                RigidBody,
                Collider,
                AudioSource,
                Animator,
                SpriteAnimation,
            );
            map.insert(entity, target);
        }

        for (source, target) in &map {
            if let Some(parent) = self.parents.get(source) {
                if let Some(parent) = map.get(parent) {
                    clone.parents.insert(*target, *parent);
                }
            }
            if let Some(kids) = self.children.get(source) {
                let kids: Vec<Entity> = kids
                    .iter()
                    .filter_map(|kid| map.get(kid).copied())
                    .collect();
                if !kids.is_empty() {
                    clone.children.insert(*target, kids);
                }
            }
            if let Some(id) = self.id_component(*source) {
                clone.id_index.insert(id.0, *target);
            }
            if let Some(names) = self.dropped.get(source) {
                clone.dropped.insert(*target, names.clone());
            }
        }
        clone
    }
}

fn detach_child(children: &mut HashMap<Entity, Vec<Entity>>, parent: Entity, child: Entity) {
    let mut empty = false;
    if let Some(siblings) = children.get_mut(&parent) {
        siblings.retain(|sibling| *sibling != child);
        empty = siblings.is_empty();
    }
    if empty {
        children.remove(&parent);
    }
}

fn resolve(
    slot: &mut Option<vault::AssetId>,
    path: &str,
    assets: &mut vault::Assets,
    errors: &mut Vec<AssetError>,
    entity: Entity,
) {
    if slot.is_some() || path.is_empty() {
        return;
    }
    match assets.load(path) {
        Ok(id) => *slot = Some(id),
        Err(_) => errors.push(AssetError {
            entity,
            path: path.to_string(),
        }),
    }
}

fn seed_id() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(1);
    nanos | 1
}

/// The inverse of [`Transform::rotation_quat`], returned in degrees.
///
/// Hand-written because glam's `EulerRot` sequences do not reproduce the
/// X-then-Y-then-Z order.
pub fn euler_degrees(rotation: Quat) -> Vec3 {
    let (x, y, z, w) = (rotation.x, rotation.y, rotation.z, rotation.w);
    let pitch = (2.0 * (y * z + w * x)).atan2(w * w - x * x - y * y + z * z);
    let yaw = (-2.0 * (x * z - w * y)).clamp(-1.0, 1.0).asin();
    let roll = (2.0 * (x * y + w * z)).atan2(w * w + x * x - y * y - z * z);
    Vec3::new(pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees())
}
