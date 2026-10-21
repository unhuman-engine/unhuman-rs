//! The layout's `imgui.ini`-style file: `to_ini`, `from_ini`, `load`/`save`.
//!
//! The format is `key= value` lines, the same shape `imgui.ini` uses, so the
//! file is hand-editable. It is a child module of [`super`] because it reaches
//! into the layout's private docks; the editor only sees the `Layout` methods
//! and [`config_path`].

use std::path::{Path, PathBuf};

use super::{Dock, Layout, PanelId};

impl Layout {
    /// Serializes to the `imgui.ini`-style file this editor reads back.
    pub fn to_ini(&self) -> String {
        let mut text = String::from("# unhuman editor layout; edit while the editor is closed\n");
        for dock in Dock::ALL {
            let names: Vec<&str> = self.docks[dock.index()]
                .iter()
                .map(|panel| panel.key())
                .collect();
            text.push_str(&format!("{}= {}\n", dock.key(), names.join(",")));
            if let Some(active) = self.active[dock.index()] {
                text.push_str(&format!("active_{}= {}\n", dock.key(), active.key()));
            }
        }
        text.push_str(&format!("left_width= {}\n", self.left_width));
        text.push_str(&format!("right_width= {}\n", self.right_width));
        text.push_str(&format!("bottom_height= {}\n", self.bottom_height));
        text
    }

    /// Parses [`Layout::to_ini`]. Unknown keys and values are ignored, so a file
    /// from a newer or older build still opens.
    pub fn from_ini(text: &str) -> Layout {
        let mut layout = Layout {
            docks: [Vec::new(), Vec::new(), Vec::new()],
            active: [None, None, None],
            ..Layout::default()
        };
        // The default's panels must not leak through when the file names none.
        for dock in Dock::ALL {
            layout.docks[dock.index()].clear();
            layout.active[dock.index()] = None;
        }

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let (key, value) = (key.trim(), value.trim());

            if let Some(dock) = key.strip_prefix("active_").and_then(Dock::from_key) {
                layout.active[dock.index()] = PanelId::from_key(value);
                continue;
            }
            if let Some(dock) = Dock::from_key(key) {
                layout.docks[dock.index()] = value
                    .split(',')
                    .filter_map(|name| PanelId::from_key(name.trim()))
                    .collect();
                continue;
            }
            match key {
                "left_width" => parse_size(value, &mut layout.left_width),
                "right_width" => parse_size(value, &mut layout.right_width),
                "bottom_height" => parse_size(value, &mut layout.bottom_height),
                _ => (),
            }
        }
        layout
    }

    /// Loads a layout, falling back to the default when the file is missing or
    /// unreadable: a corrupt `imgui.ini` should cost a layout, not a launch.
    pub fn load(path: &Path) -> Layout {
        match std::fs::read_to_string(path) {
            Ok(text) => Layout::from_ini(&text),
            Err(_) => Layout::default(),
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, self.to_ini())
    }
}

fn parse_size(value: &str, slot: &mut f32) {
    if let Ok(parsed) = value.parse::<f32>() {
        if parsed.is_finite() && parsed > 0.0 {
            *slot = parsed;
        }
    }
}

/// The layout file: `$UHE_EDITOR_LAYOUT`, else the XDG config dir, else the
/// working directory. A missing home is the working directory rather than an
/// error, so a headless run still has somewhere to write.
pub fn config_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("UHE_EDITOR_LAYOUT") {
        return PathBuf::from(explicit);
    }
    if let Ok(config) = std::env::var("XDG_CONFIG_HOME") {
        if !config.is_empty() {
            return PathBuf::from(config).join("unhuman").join("editor.ini");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".config")
                .join("unhuman")
                .join("editor.ini");
        }
    }
    PathBuf::from("editor.ini")
}
