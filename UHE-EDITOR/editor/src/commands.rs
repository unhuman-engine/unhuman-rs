//! The editor's command registry: one list that the shortcut handler, the menu
//! bar and the Help > Shortcuts panel all read, so a binding is changed in one
//! place and nowhere else.
//!
//! `mirror::commands` is the *undo history* (the record of edits), not this. The
//! history labels its entries with [`mirror::Command::label`], and the Edit menu
//! reads those; this is the set of things the user can ask the editor to do.

use gpui::Keystroke;

/// One thing the editor can do. The menu, the shortcut table and the key
/// handler switch on it; adding a variant forces every one of them to handle it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Command {
    NewScene,
    SaveScene,
    Undo,
    Redo,
    Rename,
    Delete,
    Translate,
    Rotate,
    Scale,
    Play,
    Pause,
    Step,
    Stop,
    ToggleProfiler,
    ToggleShortcuts,
}

/// What the menu and the shortcut list print, and what keys select it. An empty
/// `bindings` is a command with no key: reachable from the menu alone.
pub struct Spec {
    pub command: Command,
    pub label: &'static str,
    pub category: &'static str,
    pub bindings: &'static [&'static str],
}

pub const COMMANDS: &[Spec] = &[
    Spec {
        command: Command::NewScene,
        label: "New Scene",
        category: "File",
        bindings: &["ctrl-n"],
    },
    Spec {
        command: Command::SaveScene,
        label: "Save Scene As...",
        category: "File",
        bindings: &[],
    },
    Spec {
        command: Command::Undo,
        label: "Undo",
        category: "Edit",
        bindings: &["ctrl-z"],
    },
    Spec {
        command: Command::Redo,
        label: "Redo",
        category: "Edit",
        bindings: &["ctrl-shift-z"],
    },
    Spec {
        command: Command::Rename,
        label: "Rename",
        category: "Edit",
        bindings: &["f2"],
    },
    Spec {
        command: Command::Delete,
        label: "Delete",
        category: "Edit",
        bindings: &["delete", "backspace"],
    },
    Spec {
        command: Command::Translate,
        label: "Move",
        category: "Gizmo",
        bindings: &["w"],
    },
    Spec {
        command: Command::Rotate,
        label: "Rotate",
        category: "Gizmo",
        bindings: &["e"],
    },
    Spec {
        command: Command::Scale,
        label: "Scale",
        category: "Gizmo",
        bindings: &["r"],
    },
    Spec {
        command: Command::Play,
        label: "Play",
        category: "Play",
        bindings: &[],
    },
    Spec {
        command: Command::Pause,
        label: "Pause",
        category: "Play",
        bindings: &[],
    },
    Spec {
        command: Command::Step,
        label: "Step",
        category: "Play",
        bindings: &[],
    },
    Spec {
        command: Command::Stop,
        label: "Stop",
        category: "Play",
        bindings: &[],
    },
    Spec {
        command: Command::ToggleProfiler,
        label: "Profiler",
        category: "View",
        bindings: &[],
    },
    Spec {
        command: Command::ToggleShortcuts,
        label: "Shortcuts",
        category: "View",
        bindings: &["f1"],
    },
];

impl Command {
    pub fn spec(self) -> &'static Spec {
        COMMANDS
            .iter()
            .find(|spec| spec.command == self)
            .expect("every command is in the registry")
    }

    pub fn label(self) -> &'static str {
        self.spec().label
    }

    /// The first key that selects this command, for the menu's hint column.
    pub fn binding(self) -> Option<&'static str> {
        self.spec().bindings.first().copied()
    }

    /// Every command in a category, in registry order.
    pub fn in_category(category: &str) -> impl Iterator<Item = Command> + '_ {
        COMMANDS
            .iter()
            .filter(move |spec| spec.category == category)
            .map(|spec| spec.command)
    }
}

/// The command a keypress selects, matching gpui's own key spelling
/// (`Keystroke::unparse`, e.g. `ctrl-shift-z`).
pub fn command_for_keystroke(keystroke: &Keystroke) -> Option<Command> {
    command_for_binding(&keystroke.unparse())
}

pub fn command_for_binding(binding: &str) -> Option<Command> {
    COMMANDS
        .iter()
        .find(|spec| spec.bindings.contains(&binding))
        .map(|spec| spec.command)
}

/// The categories the menu bar shows, in order.
pub const CATEGORIES: &[&str] = &["File", "Edit", "Gizmo", "Play", "View"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_has_exactly_one_spec() {
        for spec in COMMANDS {
            assert_eq!(spec.command.spec().label, spec.label);
        }
    }

    #[test]
    fn bindings_select_their_command() {
        assert_eq!(command_for_binding("ctrl-z"), Some(Command::Undo));
        assert_eq!(command_for_binding("ctrl-shift-z"), Some(Command::Redo));
        assert_eq!(command_for_binding("f2"), Some(Command::Rename));
        assert_eq!(command_for_binding("w"), Some(Command::Translate));
        assert_eq!(command_for_binding("delete"), Some(Command::Delete));
        assert_eq!(command_for_binding("backspace"), Some(Command::Delete));
        assert_eq!(command_for_binding("q"), None);
    }

    #[test]
    fn a_keystroke_matches_through_unparse() {
        let keystroke = Keystroke::parse("ctrl-shift-z").expect("a valid keystroke");
        assert_eq!(command_for_keystroke(&keystroke), Some(Command::Redo));
    }

    #[test]
    fn categories_cover_every_command() {
        for spec in COMMANDS {
            assert!(
                CATEGORIES.contains(&spec.category),
                "{} has no menu category",
                spec.label
            );
        }
    }
}
