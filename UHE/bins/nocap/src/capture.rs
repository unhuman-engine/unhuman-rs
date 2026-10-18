//! The capture file: a frame sequence, its input, its timing, and a digest.
//!
//! ## Format
//!
//! Line oriented ASCII, `:`-separated events, so a capture is diffable and can
//! be written by hand:
//!
//! ```text
//! nocap 1
//! plugins aimlab femur
//! scene assets/scene/aimlab.unhuman
//! frame <dt_micros> <digest_hex> <event> <event> ...
//! ```
//!
//! `scene` is present only when one was loaded. A frame with no input still
//! writes both its numbers. `#` starts a comment and blank lines are ignored.
//!
//! ## Why it replays the same
//!
//! A frame is a pure function of three things: the state before it, the events
//! delivered to it, and the fixed step it is advanced by. The file carries all
//! three. The engine's `Time` samples the wall clock, so replay cannot go
//! through `App::frame`; it drives `Engine` and the plugins at the recorded step
//! directly, and is therefore independent of how fast the machine is. The
//! digest is captured and re-checked, so a capture that drifts fails by name at
//! the frame that drifted rather than silently.

use std::path::PathBuf;

use pings::Event;

use crate::event;

/// File magic and the version the parser accepts.
pub const MAGIC: &str = "nocap";
pub const VERSION: u32 = 1;

#[derive(Clone, PartialEq, Debug)]
pub struct Capture {
    /// Plugin names in registration order, so a replay of a different
    /// composition is refused instead of compared against nothing.
    pub plugins: Vec<String>,
    /// Path the scene came from, when a capture was recorded with one.
    pub scene: Option<PathBuf>,
    pub frames: Vec<Frame>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Frame {
    pub dt_micros: u64,
    /// Digest of the scene after the frame. Replay recomputes and compares it.
    pub digest: u64,
    pub events: Vec<Event>,
}

impl Capture {
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("{MAGIC} {VERSION}\n"));
        out.push_str(&format!("plugins {}\n", self.plugins.join(" ")));
        if let Some(scene) = &self.scene {
            out.push_str(&format!("scene {}\n", scene.display()));
        }
        for frame in &self.frames {
            out.push_str(&format!("frame {} {:016x}", frame.dt_micros, frame.digest));
            for event in &frame.events {
                out.push(' ');
                out.push_str(&event::encode(event));
            }
            out.push('\n');
        }
        out
    }

    /// Parses [`to_text`](Self::to_text)'s output. Any malformed line is an
    /// error: a half-read capture is not a capture.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().filter(|line| {
            let line = line.trim();
            !line.is_empty() && !line.starts_with('#')
        });

        let header = lines.next().ok_or("empty capture")?;
        if header.trim() != format!("{MAGIC} {VERSION}") {
            return Err(format!("not a {MAGIC} {VERSION} capture: {header:?}"));
        }

        let plugins = lines
            .next()
            .and_then(|line| line.strip_prefix("plugins "))
            .ok_or("missing `plugins` line")?
            .split_whitespace()
            .map(str::to_string)
            .collect();

        let mut capture = Self {
            plugins,
            scene: None,
            frames: Vec::new(),
        };

        for line in lines {
            if let Some(path) = line.strip_prefix("scene ") {
                // Only the first scene line wins; a second is ignored rather
                // than fatal, because the writer emits at most one.
                if capture.scene.is_none() {
                    capture.scene = Some(PathBuf::from(path));
                }
                continue;
            }
            let rest = line
                .strip_prefix("frame ")
                .ok_or_else(|| format!("unknown capture line {line:?}"))?;
            capture.frames.push(parse_frame(rest)?);
        }

        Ok(capture)
    }
}

fn parse_frame(rest: &str) -> Result<Frame, String> {
    let mut fields = rest.split_whitespace();
    let dt_micros = fields
        .next()
        .ok_or("frame without a step")?
        .parse()
        .map_err(|_| "frame step is not a number".to_string())?;
    let digest = u64::from_str_radix(fields.next().ok_or("frame without a digest")?, 16)
        .map_err(|_| "frame digest is not hex".to_string())?;

    let events = fields.map(event::decode).collect::<Result<Vec<_>, _>>()?;
    Ok(Frame {
        dt_micros,
        digest,
        events,
    })
}
