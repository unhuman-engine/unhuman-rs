//! What the keys and the menu do to the model.
//!
//! The layout commands write through to disk here, and [`EditorApp::run_command`]
//! is the one place a registry [`Command`] becomes an `Editor` call, so the
//! toolbar, the menu and the key handler cannot drift apart.

use std::path::PathBuf;

use gpui::Context;
use mirror::gizmo::GizmoMode;

use crate::app::EditorApp;
use crate::commands::{Command, COMMANDS};
use crate::layout::{Dock, PanelId};

impl EditorApp {
    // ---- panels: open, close, move -----------------------------------------

    pub(crate) fn toggle_panel(&mut self, panel: PanelId, cx: &mut Context<Self>) {
        self.layout.toggle(panel);
        self.after_layout_change(cx);
    }

    pub(crate) fn close_panel(&mut self, panel: PanelId, cx: &mut Context<Self>) {
        self.layout.close(panel);
        self.after_layout_change(cx);
    }

    pub(crate) fn move_panel(&mut self, panel: PanelId, dock: Dock, cx: &mut Context<Self>) {
        self.layout.place(panel, dock);
        self.menu = None;
        self.after_layout_change(cx);
    }

    fn after_layout_change(&mut self, cx: &mut Context<Self>) {
        self.save_layout();
        // Opening the profiler is what turns GPU timestamps on; a panel that has
        // never been shown must not make the frame wait for a collection.
        if self.layout.is_open(PanelId::Profiler) {
            self.enable_profiler_gpu_timing();
        }
        cx.notify();
    }

    /// Writes the layout to disk when it changed, so a rearranged editor comes
    /// back after a relaunch. A write failure is a warning, not a crash.
    pub(crate) fn save_layout(&mut self) {
        if self.layout.take_dirty() {
            if let Err(error) = self.layout.save(&self.layout_path) {
                log::warn!("editor: could not save layout: {error}");
            }
        }
    }

    // ---- commands ----------------------------------------------------------

    /// Runs a command typed on the console's command line. The name is resolved
    /// against the same registry the menu and the keys read; the result or the
    /// error is echoed into the console.
    pub(crate) fn run_console_command(&mut self, line: &str, cx: &mut Context<Self>) {
        let line = line.trim();
        if line.is_empty() {
            self.editor.console.command_clear();
            cx.notify();
            return;
        }
        self.editor.console.remember_command(line);

        let command = match parse_console_command(line) {
            Ok(command) => command,
            Err(error) => {
                self.editor.console.error(error);
                cx.notify();
                return;
            }
        };

        let outcome = match command {
            Command::Undo => match self.editor.history.undo_label() {
                Some(label) => {
                    self.run_command(command, cx);
                    Ok(format!("undid {label}"))
                }
                None => Err("nothing to undo".to_string()),
            },
            Command::Redo => match self.editor.history.redo_label() {
                Some(label) => {
                    self.run_command(command, cx);
                    Ok(format!("redid {label}"))
                }
                None => Err("nothing to redo".to_string()),
            },
            Command::Delete => {
                if self.editor.selection.selected.is_none() {
                    Err("nothing selected".to_string())
                } else {
                    self.run_command(command, cx);
                    Ok("deleted".to_string())
                }
            }
            Command::Rename => {
                if self.editor.selection.selected.is_none() {
                    Err("nothing selected".to_string())
                } else {
                    self.run_command(command, cx);
                    Ok("rename started".to_string())
                }
            }
            // The save branch already writes its warning to the console.
            Command::SaveScene => {
                self.run_command(command, cx);
                Ok(String::new())
            }
            _ => {
                self.run_command(command, cx);
                Ok("ok".to_string())
            }
        };

        match outcome {
            Ok(message) if message.is_empty() => {}
            Ok(message) => self
                .editor
                .console
                .info(format!("{}: {message}", command.label())),
            Err(error) => self
                .editor
                .console
                .error(format!("{}: {error}", command.label())),
        }
        cx.notify();
    }

    /// Runs a registry command. Both the key handler and the menu call this, so
    /// the two cannot diverge.
    pub(crate) fn run_command(&mut self, command: Command, cx: &mut Context<Self>) {
        match command {
            Command::NewScene => {
                let engine = self.engine.engine_mut();
                self.editor
                    .new_starter_scene(&mut engine.scene, &mut engine.assets);
            }
            Command::SaveScene => {
                // No file dialog in gpui and no path to write to yet; the engine
                // can save, the editor has nowhere to ask for a name.
                self.editor
                    .console
                    .warn("Save is not wired: no file dialog");
            }
            Command::Undo => {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.undo(scene);
            }
            Command::Redo => {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.redo(scene);
            }
            Command::Rename => {
                if let Some(entity) = self.editor.selection.selected {
                    let scene = &self.engine.engine().scene;
                    self.editor.begin_rename(scene, entity);
                }
            }
            Command::Delete => {
                if let Some(entity) = self.editor.selection.selected {
                    let scene = &mut self.engine.engine_mut().scene;
                    self.editor.destroy_entity(scene, entity);
                }
            }
            Command::Translate => self.editor.gizmo.mode = GizmoMode::Translate,
            Command::Rotate => self.editor.gizmo.mode = GizmoMode::Rotate,
            Command::Scale => self.editor.gizmo.mode = GizmoMode::Scale,
            Command::Play => {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.play(scene);
            }
            Command::Pause => self.editor.pause(),
            Command::Step => self.editor.step(),
            Command::Stop => {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.stop(scene);
            }
            Command::ToggleProfiler => {
                self.toggle_panel(PanelId::Profiler, cx);
                return;
            }
            Command::ToggleShortcuts => {
                self.toggle_panel(PanelId::Shortcuts, cx);
                return;
            }
        }
        cx.notify();
    }

    // ---- file drops --------------------------------------------------------

    /// Routes a dropped file by extension: a scene replaces the scene, a model
    /// or image becomes an entity. An unknown extension is logged and ignored.
    pub(crate) fn drop_files(&mut self, paths: &[PathBuf], cx: &mut Context<Self>) {
        for path in paths {
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase);
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "Entity".to_string());

            match extension.as_deref() {
                Some("unhuman") => {
                    let engine = self.engine.engine_mut();
                    match self
                        .editor
                        .open_scene(&mut engine.scene, &mut engine.assets, path)
                    {
                        Ok(()) => log::info!("editor: opened scene {}", path.display()),
                        Err(error) => log::warn!("editor: scene {}: {error}", path.display()),
                    }
                }
                Some("gltf") | Some("glb") => {
                    let engine = self.engine.engine_mut();
                    match mirror::import_path(&mut engine.assets, path) {
                        Some(virtual_path) => {
                            self.editor.add_model(
                                &mut engine.scene,
                                &mut engine.assets,
                                &virtual_path,
                                &name,
                            );
                            log::info!("editor: added model {virtual_path}");
                        }
                        None => log::warn!("editor: model {} is not loadable", path.display()),
                    }
                }
                Some("png") | Some("jpg") | Some("jpeg") | Some("bmp") | Some("gif")
                | Some("ktx2") => {
                    let engine = self.engine.engine_mut();
                    match mirror::import_path(&mut engine.assets, path) {
                        Some(virtual_path) => {
                            self.editor.add_image(
                                &mut engine.scene,
                                &mut engine.assets,
                                &virtual_path,
                                &name,
                            );
                            log::info!("editor: added image {virtual_path}");
                        }
                        None => log::warn!("editor: image {} is not loadable", path.display()),
                    }
                }
                other => log::warn!(
                    "editor: unsupported dropped file ({}) {}",
                    other.unwrap_or("no extension"),
                    path.display()
                ),
            }
        }
        cx.notify();
    }

    /// Adds a model from the content browser to the scene, the same path a drop
    /// from outside the window takes.
    pub(crate) fn add_asset(&mut self, path: &str, cx: &mut Context<Self>) {
        let engine = self.engine.engine_mut();
        let name = std::path::Path::new(path)
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Entity".to_string());
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".png")
            || lower.ends_with(".jpg")
            || lower.ends_with(".jpeg")
            || lower.ends_with(".ktx2")
        {
            self.editor
                .add_image(&mut engine.scene, &mut engine.assets, path, &name);
        } else {
            self.editor
                .add_model(&mut engine.scene, &mut engine.assets, path, &name);
        }
        cx.notify();
    }
}

/// Resolves a typed line against the registry. A label or the variant name
/// matches, and a unique prefix is enough: `save` finds `Save Scene As...`.
fn parse_console_command(line: &str) -> Result<Command, String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.is_empty() {
        return Err("empty command".to_string());
    }

    // A multi-word label first, longest match, so `new scene` is one name.
    for split in (1..=tokens.len()).rev() {
        let name = normalize(&tokens[..split].join(" "));
        if let Some(command) = exact_command(&name) {
            if split < tokens.len() {
                return Err(format!(
                    "{} takes no arguments: {}",
                    command.label(),
                    tokens[split..].join(" ")
                ));
            }
            return Ok(command);
        }
    }

    let name = normalize(tokens[0]);
    let matches = commands_starting_with(&name);
    match matches.as_slice() {
        [] => Err(format!("unknown command: {}", tokens[0])),
        [command] if tokens.len() == 1 => Ok(*command),
        [command] => Err(format!(
            "{} takes no arguments: {}",
            command.label(),
            tokens[1..].join(" ")
        )),
        _ => {
            let names: Vec<&str> = matches.iter().map(|command| command.label()).collect();
            Err(format!(
                "ambiguous command `{}`: {}",
                tokens[0],
                names.join(", ")
            ))
        }
    }
}

fn normalize(text: &str) -> String {
    text.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn command_names(command: Command) -> [String; 2] {
    [
        normalize(command.label()),
        normalize(&format!("{command:?}")),
    ]
}

fn exact_command(name: &str) -> Option<Command> {
    COMMANDS.iter().map(|spec| spec.command).find(|command| {
        command_names(*command)
            .iter()
            .any(|candidate| candidate == name)
    })
}

fn commands_starting_with(name: &str) -> Vec<Command> {
    let mut matches: Vec<Command> = Vec::new();
    for spec in COMMANDS {
        if command_names(spec.command)
            .iter()
            .any(|candidate| candidate.starts_with(name))
            && !matches.contains(&spec.command)
        {
            matches.push(spec.command);
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use meats::{Entity, Scene};
    use mirror::inspector::PropertyValue;
    use mirror::Editor;

    fn tag_of(scene: &Scene, entity: Entity) -> String {
        scene
            .get(entity)
            .and_then(|entry| entry.tag())
            .unwrap_or_default()
            .to_string()
    }

    /// The four edits the panels make (property, inline rename, reparent and
    /// create) all go through the same history, so undoing them in reverse
    /// restores the scene and the rows the panels read.
    #[test]
    fn the_four_panel_mutations_are_all_undoable() {
        let mut scene = Scene::new();
        let mut editor = Editor::new();
        let root = scene.spawn("Root");
        let child = scene.spawn("Child");
        let other = scene.spawn("Other");
        scene.set_parent(child, Some(root));
        editor.set_expanded(root, true);
        editor.set_expanded(other, true);

        editor.select(Some(child));
        let translation = editor
            .properties(&scene)
            .into_iter()
            .find(|property| property.component == "Transform" && property.name == "Translation")
            .expect("a spawned entity has a Transform");
        editor.set_property(
            &mut scene,
            &translation,
            PropertyValue::Vec3([1.0, 2.0, 3.0]),
        );

        editor.begin_rename(&scene, child);
        editor.hierarchy.rename_buffer = "Renamed".into();
        assert!(editor.commit_rename(&mut scene));

        assert!(editor.reparent(&mut scene, child, Some(other)));
        let created = editor.create_entity(&mut scene, None);
        assert_eq!(editor.history.len(), 4);

        assert!(editor.undo(&mut scene));
        assert!(scene.get(created).is_none());
        assert_eq!(scene.entities().len(), 3);
        assert_eq!(editor.selected_entity(), None);
        assert_eq!(scene.parent(child), Some(other));
        assert_eq!(tag_of(&scene, child), "Renamed");

        assert!(editor.undo(&mut scene));
        assert_eq!(scene.parent(child), Some(root));
        assert!(editor
            .rows(&scene)
            .iter()
            .any(|row| row.entity == child && row.name == "Renamed"));

        assert!(editor.undo(&mut scene));
        assert_eq!(tag_of(&scene, child), "Child");
        assert!(editor
            .rows(&scene)
            .iter()
            .any(|row| row.entity == child && row.name == "Child"));

        assert!(editor.undo(&mut scene));
        assert_eq!(
            translation.value(&scene, child),
            PropertyValue::Vec3([0.0, 0.0, 0.0])
        );
        assert!(!editor.history.can_undo());
    }

    #[test]
    fn the_command_line_resolves_registry_names() {
        assert_eq!(parse_console_command("undo"), Ok(Command::Undo));
        assert_eq!(parse_console_command("redo"), Ok(Command::Redo));
        assert_eq!(parse_console_command("save"), Ok(Command::SaveScene));
        assert_eq!(parse_console_command("new scene"), Ok(Command::NewScene));
        assert_eq!(parse_console_command("translate"), Ok(Command::Translate));
        assert_eq!(parse_console_command("prof"), Ok(Command::ToggleProfiler));
    }

    #[test]
    fn the_command_line_reports_unknown_and_ambiguous_names() {
        let unknown = parse_console_command("bogus").unwrap_err();
        assert!(unknown.contains("unknown command"));
        assert!(unknown.contains("bogus"));

        let ambiguous = parse_console_command("t").unwrap_err();
        assert!(ambiguous.contains("ambiguous"));
    }

    #[test]
    fn a_command_with_arguments_is_refused() {
        let error = parse_console_command("undo now").unwrap_err();
        assert!(error.contains("takes no arguments"));
    }
}
