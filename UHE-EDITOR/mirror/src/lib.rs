//! The editor model: the outliner, the property panel, the content browser,
//! the theme, the gizmo, the play state and the undo history.
//!
//! Toolkit-agnostic on purpose: the state is a plain struct and any UI binary is
//! one renderer of it. Nothing in this crate names a UI library, draws a widget,
//! or opens a window.
//!
//! `Editor` holds no scene reference. Every call that needs the scene takes
//! `&meats::Scene` or `&mut meats::Scene`, which keeps the editor out of the
//! engine's ownership and lets a headless run build the same model.

pub mod commands;
pub mod console;
pub mod content_browser;
pub mod gizmo;
pub mod hierarchy;
pub mod inspector;
pub mod picking;
pub mod play_state;
pub mod profiler;
pub mod selection;
pub mod theme;

pub use commands::{Command, History, DEFAULT_CAPACITY};
pub use console::{Console, Level, LogEntry};
pub use content_browser::{ContentBrowser, DragPayload, Entry};
pub use gizmo::{snap_angle, snap_scale, snap_translate, Axis, GizmoMode, GizmoTransform, Snap};
pub use hierarchy::{Hierarchy, Row};
pub use inspector::{Inspector, Property, PropertyKind, PropertyValue};
pub use picking::Ray;
pub use play_state::{PlayState, PlayStatus};
pub use profiler::{
    Counters, FrameStats, PassTime, Profiler, Stats, SystemLoad, Timings, DEFAULT_SAMPLE_INTERVAL,
};
pub use selection::Selection;
pub use theme::{Palette, Rgba, Theme, ThemeError, ThemeId};

use std::path::{Path, PathBuf};
use std::sync::Arc;

use adam::{Engine, FrameCtx, Plugin};
use glam::Mat4;
use meats::{Entity, Scene};

use commands::set_tag;
use gizmo::TransformGizmo;
use hierarchy::{is_within_subtree, tag_name};

pub struct Editor {
    pub selection: Selection,
    pub hierarchy: Hierarchy,
    pub inspector: Inspector,
    pub content_browser: ContentBrowser,
    pub theme: Theme,
    pub gizmo: TransformGizmo,
    pub play: PlayState,
    pub history: History,
    pub stats: Stats,
    pub profiler: Profiler,
    pub console: Console,
    /// The logger attachment the console shares; dropped with the editor so a
    /// closed editor stops receiving records.
    console_sink: adam::logging::SinkId,
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

impl Editor {
    pub fn new() -> Self {
        let console = Console::new();
        // The first line says where the file log is. A process launched from a
        // desktop entry has no terminal, so that path is how a failure before
        // the window opened is found again.
        if let Some(path) = adam::logging::log_path() {
            console.info(format!("log file: {}", path.display()));
        }
        let console_sink = adam::logging::attach(Arc::new(console.clone()));

        Self {
            selection: Selection::new(),
            hierarchy: Hierarchy::new(),
            inspector: Inspector::new(),
            content_browser: ContentBrowser::new(PathBuf::from("assets")),
            theme: Theme::new(),
            gizmo: TransformGizmo::new(),
            play: PlayState::new(),
            history: History::new(),
            stats: Stats::default(),
            profiler: Profiler::new(),
            console,
            console_sink,
        }
    }

    pub fn selected_entity(&self) -> Option<Entity> {
        self.selection.selected
    }

    pub fn select(&mut self, entity: Option<Entity>) {
        self.selection.select(entity);
    }

    pub fn hover(&mut self, entity: Option<Entity>) {
        self.selection.hover(entity);
    }

    pub fn hovered(&self) -> Option<Entity> {
        self.selection.hovered
    }

    /// Picks the entity under a ray: the nearest mesh whose bounds it crosses,
    /// or nothing. A hit selects, a miss clears the selection, and the result is
    /// the new hover either way. See [`picking`].
    pub fn pick(&mut self, scene: &Scene, assets: &vault::Assets, ray: Ray) -> Option<Entity> {
        let hit = picking::pick(scene, assets, ray);
        self.selection.hover(hit);
        self.selection.select(hit);
        hit
    }

    pub fn rows(&self, scene: &Scene) -> Vec<Row> {
        self.hierarchy.rows(scene)
    }

    pub fn set_expanded(&mut self, entity: Entity, expanded: bool) {
        self.hierarchy.set_expanded(entity, expanded);
    }

    pub fn toggle_expanded(&mut self, entity: Entity) {
        self.hierarchy.toggle_expanded(entity);
    }

    pub fn properties(&self, scene: &Scene) -> Vec<Property> {
        match self.selection.selected {
            Some(entity) => self.inspector.properties(scene, entity),
            None => Vec::new(),
        }
    }

    /// Edits the selected entity's property and records it for undo.
    pub fn set_property(&mut self, scene: &mut Scene, property: &Property, value: PropertyValue) {
        let Some(entity) = self.selection.selected else {
            return;
        };
        let before = property.value(scene, entity);
        if before == value {
            return;
        }
        self.inspector.set(scene, entity, property, value.clone());
        self.history.push(Command::SetProperty {
            entity,
            component: property.component,
            property: property.name,
            before,
            after: value,
        });
    }

    pub fn rename(&mut self, scene: &mut Scene, entity: Entity, name: String) {
        let before = tag_name(scene, entity);
        if name.is_empty() || name == before {
            return;
        }
        set_tag(scene, entity, &name);
        self.inspector.mark_dirty(entity);
        self.history.push(Command::Rename {
            entity,
            before,
            after: name,
        });
    }

    /// Opens the outliner's inline rename editor on `entity`.
    pub fn begin_rename(&mut self, scene: &Scene, entity: Entity) {
        self.hierarchy.begin_rename(scene, entity);
    }

    pub fn cancel_rename(&mut self) {
        self.hierarchy.cancel_rename();
    }

    /// Closes the inline rename editor. `false` when the name did not change,
    /// which is also when no history entry is worth making.
    pub fn commit_rename(&mut self, scene: &mut Scene) -> bool {
        let Some(mut command) = self.hierarchy.commit_rename(scene) else {
            return false;
        };
        command.apply(scene, &mut self.inspector);
        self.history.push(command);
        true
    }

    pub fn create_entity(&mut self, scene: &mut Scene, parent: Option<Entity>) -> Entity {
        let entity = scene.spawn(EMPTY_ENTITY);
        if let Some(parent) = parent {
            scene.set_parent(entity, Some(parent));
            self.hierarchy.set_expanded(parent, true);
        }
        self.selection.select(Some(entity));
        self.inspector.mark_dirty(entity);
        self.history.push(Command::CreateEntity {
            entity,
            parent,
            name: EMPTY_ENTITY.to_string(),
        });
        entity
    }

    /// Spawns an entity drawing the model at `path`, loaded through the
    /// registry. `path` is the registry's virtual path, not a filesystem one.
    pub fn add_model(
        &mut self,
        scene: &mut Scene,
        assets: &mut vault::Assets,
        path: &str,
        name: &str,
    ) -> Entity {
        let entity = scene.spawn(name);
        scene.insert(
            entity,
            meats::MeshRenderer {
                mesh_path: path.to_string(),
                ..Default::default()
            },
        );
        self.resolve_dropped(scene, assets, entity, name);
        entity
    }

    /// Spawns an entity drawing the image at `path`. `path` is the registry's
    /// virtual path.
    pub fn add_image(
        &mut self,
        scene: &mut Scene,
        assets: &mut vault::Assets,
        path: &str,
        name: &str,
    ) -> Entity {
        let entity = scene.spawn(name);
        scene.insert(
            entity,
            meats::SpriteRenderer {
                texture_path: path.to_string(),
                ..Default::default()
            },
        );
        self.resolve_dropped(scene, assets, entity, name);
        entity
    }

    /// Loads the component paths a freshly spawned asset entity carries, then
    /// makes the drop undoable and visible.
    fn resolve_dropped(
        &mut self,
        scene: &mut Scene,
        assets: &mut vault::Assets,
        entity: Entity,
        name: &str,
    ) {
        for error in scene.resolve_assets(assets) {
            log::warn!("mirror: {error}");
        }
        self.selection.select(Some(entity));
        self.inspector.mark_dirty(entity);
        self.history.push(Command::CreateEntity {
            entity,
            parent: None,
            name: name.to_string(),
        });
    }

    /// Destroys the entity and its subtree. A selection inside that subtree is
    /// dropped first, so the properties panel cannot draw a dangling handle.
    pub fn destroy_entity(&mut self, scene: &mut Scene, entity: Entity) {
        let parent = scene.parent(entity);
        let name = tag_name(scene, entity);
        if let Some(selected) = self.selection.selected {
            if is_within_subtree(scene, entity, selected) {
                self.selection.select(None);
            }
        }
        scene.destroy(entity);
        self.history.push(Command::DestroyEntity {
            entity,
            parent,
            name,
        });
    }

    /// Re-parents an entity. Refused when the target is inside the entity's own
    /// subtree, which would make a cycle.
    pub fn reparent(&mut self, scene: &mut Scene, entity: Entity, parent: Option<Entity>) -> bool {
        if let Some(parent) = parent {
            if !self.hierarchy.can_drop(scene, entity, parent) {
                return false;
            }
        }
        let before = scene.parent(entity);
        if before == parent {
            return true;
        }
        scene.set_parent(entity, parent);
        self.history.push(Command::Reparent {
            entity,
            before,
            after: parent,
        });
        true
    }

    pub fn play(&mut self, scene: &mut Scene) {
        self.play.play(scene);
    }

    pub fn pause(&mut self) {
        self.play.pause();
    }

    pub fn resume(&mut self) {
        self.play.resume();
    }

    pub fn stop(&mut self, scene: &mut Scene) {
        self.play.stop(scene);
        // A runtime-spawned entity the user had selected is not in the authored
        // scene that just came back.
        self.forget_dead_selection(scene);
    }

    /// Queues exactly one fixed step; only a paused editor honours it.
    pub fn step(&mut self) {
        self.play.step_once();
    }

    /// Advances the play state and the frame timings, returning the seconds to
    /// simulate. Property edits are flushed here too: a transform written by
    /// the panel has to reach the world transforms before the renderer reads
    /// them this frame.
    pub fn tick(&mut self, scene: &mut Scene, dt: f32) -> f32 {
        self.profiler.begin_frame(dt);
        self.profiler.publish(std::time::Instant::now());
        self.stats = *self.profiler.stats();
        if !self.inspector.dirty().is_empty() {
            scene.update_transforms();
            self.inspector.clear_dirty();
        }
        self.play.tick(dt)
    }

    /// Writes a drag's world transform back into the selected entity's local
    /// `Transform`, through its parent, and records what changed.
    ///
    /// The gizmo works in world space so a nested entity still lines up with what
    /// is on screen; this multiplies by the inverse of the parent's world
    /// transform, and `Transform` stores euler degrees.
    pub fn apply_gizmo(&mut self, scene: &mut Scene, target: &GizmoTransform) -> bool {
        let Some(entity) = self.selection.selected else {
            return false;
        };
        let local = self.local_under_parent(scene, entity, target);
        let mut changed = false;
        for property in self.inspector.properties(scene, entity) {
            let value = match (property.component, property.name) {
                ("Transform", "Translation") => PropertyValue::Vec3(local.translation.to_array()),
                ("Transform", "Rotation") => {
                    // The component's quaternion is `Rz*Ry*Rx`: glam's
                    // `EulerRot::XYZEx`, not the `XYZ` anyone reaches for. The
                    // inverse is `meats`' to own.
                    PropertyValue::Vec3(meats::euler_degrees(local.rotation).to_array())
                }
                ("Transform", "Scale") => PropertyValue::Vec3(local.scale.to_array()),
                _ => continue,
            };
            if property.value(scene, entity) == value {
                continue;
            }
            self.set_property(scene, &property, value);
            changed = true;
        }
        changed
    }

    fn local_under_parent(
        &self,
        scene: &Scene,
        entity: Entity,
        target: &GizmoTransform,
    ) -> GizmoTransform {
        let Some(parent) = scene.parent(entity) else {
            return *target;
        };
        let world = Mat4::from_scale_rotation_translation(
            target.scale,
            target.rotation,
            target.translation,
        );
        let (scale, rotation, translation) =
            (scene.world_transform(parent).inverse() * world).to_scale_rotation_translation();
        GizmoTransform {
            translation,
            rotation,
            scale,
        }
    }

    pub fn undo(&mut self, scene: &mut Scene) -> bool {
        let undone = self.history.undo(scene, &mut self.inspector);
        if undone {
            self.forget_dead_selection(scene);
        }
        undone
    }

    pub fn redo(&mut self, scene: &mut Scene) -> bool {
        let redone = self.history.redo(scene, &mut self.inspector);
        if redone {
            self.forget_dead_selection(scene);
        }
        redone
    }

    /// Replaces the scene with one loaded from disk, resolving its asset paths
    /// against the registry. A missing asset is a warning.
    pub fn open_scene(
        &mut self,
        scene: &mut Scene,
        assets: &mut vault::Assets,
        path: &Path,
    ) -> meats::Result<()> {
        let loaded = meats::load(path)?;
        self.reset(scene);
        *scene = loaded;
        for error in scene.resolve_assets(assets) {
            log::warn!("mirror: {error}");
        }
        self.history.clear();
        Ok(())
    }

    pub fn save_scene(&self, scene: &Scene, path: &Path) -> meats::Result<()> {
        meats::save(scene, path)
    }

    pub fn new_scene(&mut self, scene: &mut Scene) {
        self.reset(scene);
        scene.clear();
        self.history.clear();
    }

    /// Replaces the scene with [`starter_scene`] and resolves its asset paths.
    /// A fresh launch and File > New both land here, so the panels always have
    /// something to show instead of an empty hierarchy.
    pub fn new_starter_scene(&mut self, scene: &mut Scene, assets: &mut vault::Assets) {
        self.new_scene(scene);
        starter_scene(scene);
        for error in scene.resolve_assets(assets) {
            self.console.warn(error.to_string());
        }
        self.console.info("new scene");
    }

    fn reset(&mut self, scene: &mut Scene) {
        self.play.stop(scene);
        self.selection.clear();
        self.hierarchy.reset();
        self.inspector.clear_dirty();
        self.gizmo.end();
    }

    fn forget_dead_selection(&mut self, scene: &Scene) {
        if let Some(selected) = self.selection.selected {
            if scene.get(selected).is_none() {
                self.selection.select(None);
            }
        }
    }
}

impl Drop for Editor {
    fn drop(&mut self) {
        adam::logging::detach(self.console_sink);
    }
}

/// The entities a fresh scene starts with: a camera, a directional light and a
/// visible model.
///
/// The camera is not primary, so the editor's orbit camera keeps driving the
/// viewport; a primary scene camera would take the view over on the first
/// frame. The mesh path is a registry path, so it stays unresolved until a
/// caller runs [`meats::Scene::resolve_assets`].
pub fn starter_scene(scene: &mut Scene) {
    let camera = scene.spawn("Camera");
    scene.insert(
        camera,
        meats::Camera {
            primary: false,
            ..Default::default()
        },
    );
    set_transform(
        scene,
        camera,
        glam::Vec3::new(0.0, 2.0, -6.0),
        glam::Vec3::ZERO,
    );

    let light = scene.spawn("Directional Light");
    scene.insert(light, meats::Light::default());
    set_transform(
        scene,
        light,
        glam::Vec3::new(0.0, 3.0, 0.0),
        glam::Vec3::new(-50.0, -30.0, 0.0),
    );

    let model = scene.spawn("Sphere");
    scene.insert(
        model,
        meats::MeshRenderer {
            mesh_path: "models/Sphere.glb".to_string(),
            ..Default::default()
        },
    );
    set_transform(scene, model, glam::Vec3::ZERO, glam::Vec3::ZERO);
}

fn set_transform(scene: &mut Scene, entity: Entity, translation: glam::Vec3, rotation: glam::Vec3) {
    if let Some(mut entry) = scene.get_mut(entity) {
        if let Some(transform) = entry.get_mut::<meats::Transform>() {
            transform.translation = translation;
            transform.rotation = rotation;
        }
    }
}

/// The registry's virtual path for a file, mounting its directory if it sits
/// outside every root.
///
/// A file already under a root keeps its root-relative path. Anything else is
/// addressed by file name after its directory is mounted, so a file dropped
/// from anywhere still loads. Callers store the returned virtual path: that is
/// what a saved scene needs to resolve on the next launch, not the absolute
/// path the drop carried.
pub fn import_path(assets: &mut vault::Assets, path: &Path) -> Option<String> {
    let roots = assets.roots().to_vec();
    for root in &roots {
        if let Ok(relative) = path.strip_prefix(root) {
            return Some(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    let parent = path.parent()?;
    if !roots.iter().any(|root| root.as_path() == parent) {
        assets.mount(parent);
    }
    Some(path.file_name()?.to_string_lossy().into_owned())
}

/// The world-space transform the gizmo draws at for `entity`.
pub fn gizmo_target(scene: &Scene, entity: Entity) -> GizmoTransform {
    let (scale, rotation, translation) = scene
        .world_transform(entity)
        .to_scale_rotation_translation();
    GizmoTransform {
        translation,
        rotation,
        scale,
    }
}

const EMPTY_ENTITY: &str = "Empty Entity";

pub struct MirrorPlugin {
    editor: Editor,
}

impl Default for MirrorPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl MirrorPlugin {
    pub fn new() -> Self {
        Self {
            editor: Editor::new(),
        }
    }

    pub fn editor(&self) -> &Editor {
        &self.editor
    }

    pub fn editor_mut(&mut self) -> &mut Editor {
        &mut self.editor
    }
}

impl Plugin for MirrorPlugin {
    fn name(&self) -> &'static str {
        "mirror"
    }

    /// No GPU work and no window: the editor model is built here and the panel
    /// state is what the toolkit layer renders.
    fn build(&mut self, _eng: &mut Engine) -> adam::Result<()> {
        self.editor.theme = match Theme::load(Path::new(theme::DEFAULT_PATH)) {
            Ok(theme) => theme,
            Err(error) => {
                log::warn!("mirror: {error}; using the default theme");
                Theme::new()
            }
        };
        Ok(())
    }

    fn on_frame(&mut self, eng: &mut Engine, ctx: &mut FrameCtx) {
        self.editor.tick(&mut eng.scene, ctx.delta.as_secs_f32());
    }
}
