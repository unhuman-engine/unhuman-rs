//! Windowed run: a real window, a real swapchain, one frame loop.
//!
//! The GPU cannot exist before the window: a `wgpu` surface is created from the
//! `winit` window and handed over inside the frame callback. `App::start` runs
//! from the first frame, once there is a device for plugins to build against.

use std::sync::{Arc, Mutex};

use chassis::backend::{BackendChoice, BackendSelection};
use eyes::{FrameView, WindowDesc, WindowExt};

use crate::args::Config;
use crate::plugins;

const TITLE: &str = "unhuman sandbox";
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

pub fn run(config: &Config) -> Result<(), String> {
    let (settings, skipped) = engine::load_settings();
    if let Some(reason) = skipped {
        eprintln!("sandbox: ignoring settings file ({reason})");
    }
    let environment = std::env::var(chassis::backend::ENVIRONMENT_VARIABLE).ok();
    let selection = BackendSelection::resolve(
        &config.raw,
        environment.as_deref(),
        settings
            .as_ref()
            .and_then(chassis::settings::Settings::backend),
    )
    .map_err(|error| error.to_string())?;

    if selection.choice != BackendChoice::Wgpu {
        return Err(format!(
            "the windowed path is wgpu-only and {} asked for {}; creating a Vulkan \
             surface is not wired",
            selection.source,
            selection.choice.name()
        ));
    }
    log::info!(
        "sandbox: backend {} ({})",
        selection.choice.name(),
        selection.source
    );

    let instance = stormglass::create_instance();

    let mut app = plugins::app();
    if let Some(path) = &config.scene {
        plugins::load_scene(&mut app, path)?;
    }
    log::info!("sandbox: plugins {}", app.plugin_names().join(", "));

    // The frame callback is `'static` and owns the app, so a failure inside it
    // has to be handed back out through something shared. Failure here is
    // terminal, and an exit status of 0 after "no wgpu device" is a lie a script
    // would believe.
    let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let reported = Arc::clone(&failure);
    let frame = move |view: &mut FrameView<'_>| -> bool {
        let fail = |message: String| {
            if let Ok(mut slot) = reported.lock() {
                *slot = Some(message);
            }
            false
        };

        if app.engine().gpu.is_none() {
            let window = view.window.clone();
            let (width, height) = window.size();
            let extent = chassis::Extent2d {
                width: width.max(1),
                height: height.max(1),
            };
            let surface = match instance.create_surface(window.clone()) {
                Ok(surface) => surface,
                Err(error) => return fail(format!("no wgpu surface: {error}")),
            };
            match stormglass::WgpuGpu::with_surface(&instance, surface, extent) {
                Ok(gpu) => app.engine_mut().gpu = Some(Box::new(gpu)),
                Err(error) => return fail(format!("wgpu: {error}")),
            }
            if let Err(error) = app.start() {
                return fail(error.to_string());
            }
        }

        // The window layer pushes; `App::frame` drains into `Engine::input`, so
        // there is one copy of the held and pressed state and not two.
        for event in view.events {
            app.engine_mut().events.push(event.clone());
        }
        if let Err(error) = app.frame() {
            return fail(error.to_string());
        }
        if app.is_running() {
            return true;
        }
        // The loop is over: run the plugins' teardown hook while the window and
        // the device are still alive.
        if let Err(error) = app.stop() {
            return fail(error.to_string());
        }
        false
    };

    let desc = WindowDesc {
        title: TITLE.to_owned(),
        width: WIDTH,
        height: HEIGHT,
        vsync: true,
    };
    eyes::run_windowed(desc, frame).map_err(|error| format!("window: {error}"))?;

    match failure.lock().ok().and_then(|mut slot| slot.take()) {
        Some(message) => Err(message),
        None => Ok(()),
    }
}
