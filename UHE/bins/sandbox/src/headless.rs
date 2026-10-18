//! Headless run: no window, and a GPU only when the factory can build one.
//!
//! `gpu: None` is not a degraded mode; it is the same engine with no device, and
//! it is what lets a captured run replay on a machine that has none.

use crate::args::Config;
use crate::plugins;

/// Engine name handed to the backend factory, for driver logs.
const ENGINE_NAME: &std::ffi::CStr = c"unhuman-sandbox";

pub fn run(config: &Config) -> Result<(), String> {
    let mut app = plugins::app();
    if let Some(path) = &config.scene {
        plugins::load_scene(&mut app, path)?;
    }

    // The factory resolves `--backend` against the same environment and settings
    // a windowed run would, then builds a device. A missing driver is a warning,
    // not a failure: the loop still advances, with no `on_render` hook.
    let environment = std::env::var(chassis::backend::ENVIRONMENT_VARIABLE).ok();
    let extent = engine::default_headless_extent();
    match engine::create_gpu_from_process(&config.raw, environment.as_deref(), ENGINE_NAME, extent)
    {
        Ok((gpu, selection)) => {
            log::info!(
                "sandbox: headless gpu {} ({})",
                selection.choice.name(),
                selection.source
            );
            app.engine_mut().gpu = Some(gpu);
        }
        Err(error) => {
            log::warn!("sandbox: no GPU ({error}); running headless on the CPU");
        }
    }

    app.run(config.frames())
        .map_err(|error| error.to_string())?;

    let engine = app.engine();
    println!(
        "sandbox: headless frames={} entities={} bodies={} gpu={}",
        engine.time.frame(),
        engine.scene.entities().len(),
        engine
            .physics
            .as_ref()
            .map_or(0, |physics| physics.body_count()),
        !engine.is_headless(),
    );
    Ok(())
}
