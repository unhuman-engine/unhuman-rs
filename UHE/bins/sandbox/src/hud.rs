//! The aim lab's on-screen readout.
//!
//! Reads the demo's fixed `Report` and queues lines into the renderer's HUD.
//! Windowed and headless share it, so the two paths show the same thing.

use adam::App;
use aimlab::{AimLabPlugin, Report, MAX_AMMO};
use bifrost::BifrostPlugin;
use glam::{Vec2, Vec4};

use crate::crosshair;

/// Pixels from the top-left of the viewport.
const MARGIN: f32 = 16.0;
/// Pixels per line.
const LINE_HEIGHT: f32 = 22.0;

/// Queues this frame's HUD. A no-op without an aim lab or a loaded face.
pub fn push(app: &mut App) {
    let Some(report) = app.plugin::<AimLabPlugin>().map(AimLabPlugin::report) else {
        return;
    };
    let viewport = viewport(app);
    let Some(renderer) = app.plugin_mut::<BifrostPlugin>() else {
        return;
    };

    renderer.clear_hud();

    let ammo = if report.ammo.reloading {
        format!("Reloading {:.0}%", report.ammo.reload_progress() * 100.0)
    } else {
        format!("Ammo {}/{}", report.ammo.rounds, MAX_AMMO)
    };
    let mut lines = vec![
        format!("Score {}", report.score.points),
        format!(
            "Hits {}   Misses {}   Shots {}",
            report.score.hits, report.score.misses, report.score.shots
        ),
        format!("Accuracy {:.1}%", report.accuracy()),
        ammo,
        String::new(),
        "Controls".to_owned(),
    ];
    lines.extend(Report::CONTROLS.iter().map(|line| (*line).to_owned()));

    for (index, line) in lines.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        let y = MARGIN + index as f32 * LINE_HEIGHT;
        renderer.push_hud(line, Vec2::new(MARGIN, y), LINE_HEIGHT, Vec4::ONE);
    }

    if let (Some(hud), Some(viewport)) = (renderer.hud(), viewport) {
        crosshair::push(hud, viewport, Report::CROSSHAIR);
    }
}

/// The frame's target size. Width is rounded up the way the renderer pads rows,
/// because the overlay's screen projection is built from that same extent.
fn viewport(app: &App) -> Option<Vec2> {
    let extent = app.engine().gpu.as_ref()?.extent();
    let width = chassis::copy_row_pitch(extent.width.max(1), 4) / 4;
    Some(Vec2::new(width as f32, extent.height.max(1) as f32))
}
