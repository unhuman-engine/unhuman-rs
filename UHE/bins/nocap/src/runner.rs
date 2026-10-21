//! Record, replay and the plain headless summary.

use std::path::Path;
use std::time::Duration;

use adam::App;
use aimlab::AimLabPlugin;
use femur::FemurPlugin;
use wire::WirePlugin;

use crate::capture::{Capture, Frame};
use crate::config::Config;
use crate::{input, plugins};

/// The recorded step. Microseconds so the value a capture writes is the value
/// replay advances by, with no `f64` round-trip in between.
pub const STEP_MICROS: u64 = 16_666;

/// Records `--steps` frames of scripted input to `path`, each with its digest.
pub fn record(config: &Config, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let steps = config
        .frames()
        .ok_or("--record needs a finite --steps (0 means until a plugin quits)")?;

    let mut session = plugins::session();
    if let Some(scene) = &config.scene {
        load_scene(&mut session, scene)?;
    }
    session.start()?;

    let step = Duration::from_micros(STEP_MICROS);
    let mut frames = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let events = input::scripted(session.frame());
        session.step(step, &events);
        frames.push(Frame {
            dt_micros: STEP_MICROS,
            digest: session.digest(),
            events,
        });
    }
    session.stop();

    let capture = Capture {
        plugins: session.plugin_names(),
        scene: config.scene.clone(),
        frames,
    };
    std::fs::write(path, capture.to_text())
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    println!(
        "nocap: recorded {} frames to {}",
        capture.frames.len(),
        path.display()
    );
    Ok(())
}

/// Replays `path` with no window and no GPU, checking every frame's digest.
pub fn replay(config: &Config, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let capture = Capture::from_text(
        &std::fs::read_to_string(path)
            .map_err(|error| format!("read {}: {error}", path.display()))?,
    )?;

    let mut session = plugins::session();
    let names = session.plugin_names();
    if names != capture.plugins {
        return Err(format!(
            "capture plugins {:?} do not match this binary's {:?}; \
             a replay of a different composition would compare against nothing",
            capture.plugins, names
        )
        .into());
    }

    if let Some(scene) = config.scene.as_ref().or(capture.scene.as_ref()) {
        load_scene(&mut session, scene)?;
    }
    session.start()?;

    for (index, frame) in capture.frames.iter().enumerate() {
        session.step(Duration::from_micros(frame.dt_micros), &frame.events);
        let got = session.digest();
        if got != frame.digest {
            return Err(format!(
                "replay diverged at frame {}: captured {:016x}, got {:016x}",
                index + 1,
                frame.digest,
                got
            )
            .into());
        }
    }
    session.stop();

    println!(
        "nocap: replayed {} frames, digests match",
        capture.frames.len()
    );
    Ok(())
}

/// The plain run: a headless engine, the game plugins, a summary line.
pub fn summary(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::new()
        .add(AimLabPlugin::new())
        .add(WirePlugin::new())
        .add(FemurPlugin);

    // Aimlab's paths spell the `assets/` prefix, so the repository root has to
    // be a root too; `assets` stays as the fallback for asset-relative paths.
    app.engine_mut().assets.mount(".");
    app.engine_mut().assets.mount("assets");

    log::info!("nocap: plugins {}", app.plugin_names().join(", "));

    if config.list_plugins {
        for name in app.plugin_names() {
            println!("{name}");
        }
        return Ok(());
    }

    if let Some(path) = &config.scene {
        let adam::Engine {
            scene,
            assets,
            jobs,
            ..
        } = app.engine_mut();
        *scene =
            meats::load(path).map_err(|error| format!("--scene {}: {error}", path.display()))?;
        for error in scene.resolve_assets_async(jobs, assets) {
            log::warn!("nocap: {error}");
        }
    }

    app.run(config.frames())?;

    let engine = app.engine();
    // Sleeping bodies count: there is no awake-body accessor, and zero is the
    // honest "physics did nothing".
    println!(
        "nocap: frames={} entities={} bodies={}",
        engine.time.frame(),
        engine.scene.entities().len(),
        engine
            .physics
            .as_ref()
            .map_or(0, |physics| physics.body_count()),
    );
    Ok(())
}

fn load_scene(
    session: &mut crate::session::Session,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let adam::Engine {
        scene,
        assets,
        jobs,
        ..
    } = session.engine_mut();
    *scene = meats::load(path).map_err(|error| format!("--scene {}: {error}", path.display()))?;
    for error in scene.resolve_assets_async(jobs, assets) {
        log::warn!("nocap: {error}");
    }
    Ok(())
}
