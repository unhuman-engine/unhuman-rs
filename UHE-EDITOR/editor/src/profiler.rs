//! The live profiler panel: the loop's period split into the engine's frame,
//! its readback and the UI's own paint, then the counters and OS rows
//! `mirror::Profiler` keeps.
//!
//! The rows come from [`mirror::FrameStats`], so the panel and the model cannot
//! disagree about what a number means. A row the backend does not fill is shown
//! as "not collected", not as a zero that reads as "free".

use gpui::{div, prelude::*, px, AnyElement, IntoElement, SharedString};
use mirror::FrameStats;

use crate::panels::ROW_HEIGHT;
use crate::theme::EditorTheme;

/// The upload time and scene counts this frame, alongside `mirror`'s rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameNumbers {
    pub upload_ms: f32,
    pub entities: usize,
    pub meshes: usize,
}

fn millis(ms: f32) -> String {
    format!("{ms:.2} ms")
}

/// A count the renderer did not report is not zero.
fn count(value: u32) -> String {
    if value == 0 {
        "not collected".to_string()
    } else {
        value.to_string()
    }
}

fn option(value: Option<f32>, unit: &str) -> String {
    match value {
        Some(value) => format!("{value:.1}{unit}"),
        None => "not available".to_string(),
    }
}

fn memory(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) => format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0)),
        None => "not available".to_string(),
    }
}

pub fn panel(
    stats: &FrameStats,
    fps: f32,
    numbers: FrameNumbers,
    theme: &EditorTheme,
) -> impl IntoElement {
    let timings = &stats.timings;
    let system = &stats.system;
    let mut rows: Vec<(&str, String)> = vec![
        ("loop", millis(timings.loop_ms)),
        ("engine frame", millis(timings.engine_ms)),
        ("render", millis(timings.render_ms)),
        (
            "readback",
            timings
                .readback_ms
                .map(millis)
                .unwrap_or_else(|| "not collected".to_string()),
        ),
        ("ui paint", millis(timings.ui_ms())),
        ("image upload", millis(numbers.upload_ms)),
        ("fps", format!("{fps:.1}")),
        ("entities", numbers.entities.to_string()),
        ("meshes", numbers.meshes.to_string()),
        ("draw calls", count(stats.counters.draw_calls)),
        ("passes", count(stats.counters.passes)),
        ("instances", stats.counters.instances.to_string()),
        ("cpu", option(system.cpu_percent, "%")),
        ("rss", memory(system.resident_bytes)),
        ("gpu load", option(system.gpu_percent, "%")),
    ];

    // The seam, spelled out: an empty pass list is "no timestamp queries", not
    // a frame made of free passes.
    if stats.gpu_passes.is_empty() {
        rows.push(("gpu passes", "not collected".to_string()));
    } else {
        for pass in &stats.gpu_passes {
            rows.push((pass.pass.as_str(), millis(pass.ms)));
        }
    }

    let mut children: Vec<AnyElement> = Vec::with_capacity(rows.len());
    for (label, value) in rows {
        children.push(
            div()
                .flex_none()
                .h(px(ROW_HEIGHT))
                .px_2()
                .flex()
                .items_center()
                .justify_between()
                .gap_3()
                .text_sm()
                .child(
                    div()
                        .flex_none()
                        .text_color(theme.text_dim)
                        .child(SharedString::from(label.to_string())),
                )
                .child(
                    div()
                        .flex_none()
                        .text_color(theme.text)
                        .child(SharedString::from(value)),
                )
                .into_any_element(),
        );
    }

    div()
        .id("profiler-list")
        .flex()
        .flex_col()
        .size_full()
        .overflow_y_scroll()
        .bg(theme.panel)
        .children(children)
}
