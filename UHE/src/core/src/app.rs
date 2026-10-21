use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::plugin::{FrameCtx, Plugin};
/// The composition root: owns the plugin registry and the loop.
///
/// `Engine` owns the data, so iterating plugins and borrowing the engine are
/// disjoint borrows. Field order is drop order: plugins must drop before the
/// engine, or a renderer leaks the GPU resources its device still holds.
pub struct App {
    plugins: Vec<Box<dyn Plugin>>,
    engine: Engine,
    running: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            engine: Engine::new(),
            running: true,
        }
    }

    pub fn with_engine(engine: Engine) -> Self {
        Self {
            plugins: Vec::new(),
            engine,
            running: true,
        }
    }

    /// Registration order is execution order.
    #[allow(clippy::should_implement_trait)]
    pub fn add<P: Plugin>(mut self, plugin: P) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    pub fn plugin_names(&self) -> Vec<&'static str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    /// A registered plugin, downcast to its concrete type.
    ///
    /// `None` when no plugin of that type is registered. This is how a caller
    /// reads a plugin's own output: the editor pulls the renderer's last frame
    /// through it, without `adam` naming a renderer.
    pub fn plugin_mut<T: Plugin>(&mut self) -> Option<&mut T> {
        self.plugins
            .iter_mut()
            .find_map(|plugin| (plugin.as_mut() as &mut dyn std::any::Any).downcast_mut::<T>())
    }

    /// The same plugin, for reading: a caller that only needs the camera or the
    /// last frame does not have to borrow the whole `App` mutably to ask.
    pub fn plugin<T: Plugin>(&self) -> Option<&T> {
        self.plugins
            .iter()
            .find_map(|plugin| (plugin.as_ref() as &dyn std::any::Any).downcast_ref::<T>())
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn quit(&mut self) {
        self.running = false;
    }

    /// Builds and starts every plugin in registration order.
    pub fn start(&mut self) -> Result<()> {
        for plugin in &mut self.plugins {
            plugin.build(&mut self.engine).map_err(|e| Error::Plugin {
                name: plugin.name(),
                message: e.to_string(),
            })?;
        }
        for plugin in &mut self.plugins {
            plugin
                .on_start(&mut self.engine)
                .map_err(|e| Error::Plugin {
                    name: plugin.name(),
                    message: e.to_string(),
                })?;
        }
        Ok(())
    }

    /// Advances one frame: fixed simulation substeps, then hooks, then GPU.
    pub fn frame(&mut self) -> Result<()> {
        let delta = self.engine.time.tick();
        let mut ctx = FrameCtx {
            frame: self.engine.time.frame(),
            delta,
            step: delta,
            scale: self.engine.time.scale(),
            quit: false,
        };

        // Input state is owned by the engine and fed from the event bus: one
        // consumer, one source of truth. The window pushes events and keeps no
        // second copy of the held/pressed state.
        self.engine.input.begin_frame();
        for event in self.engine.events.drain() {
            self.engine.input.apply(&event);
        }

        // Fixed-step substeps first, so rendering sees a settled simulation
        // state regardless of frame rate.
        while let Some(step) = self.engine.time.consume_fixed_step() {
            ctx.step = step;
            self.engine.simulate_fixed_step(step);
            for plugin in &mut self.plugins {
                plugin.on_update(&mut self.engine, &mut ctx);
            }
        }

        // Physics wrote the last step's pose; blend the last two steps so a
        // frame between steps does not show the same pose twice.
        let alpha = self.engine.time.alpha();
        if let Some(physics) = self.engine.physics.as_mut() {
            physics.interpolate(&mut self.engine.scene, alpha);
        }

        ctx.step = self.engine.time.fixed_dt();
        for plugin in &mut self.plugins {
            plugin.on_frame(&mut self.engine, &mut ctx);
        }

        // `on_render` runs once and only with a GPU: with no GPU there is
        // nothing to record into. The `gpu` borrow is re-taken per use because
        // the hook needs `&mut self.engine` whole and pulls the GPU out itself.
        //
        // The order is the frame's contract: acquire, let every plugin record
        // (a renderer hands its frame to `Gpu::record_frame`), then `submit`,
        // the one place a frame is presented. Two present sites would present
        // a frame twice.
        if self.engine.gpu.is_some() {
            if let Some(gpu) = self.engine.gpu.as_mut() {
                gpu.frame_begin()?;
            }
            for plugin in &mut self.plugins {
                plugin.on_render(&mut self.engine, &mut ctx);
            }
            if let Some(gpu) = self.engine.gpu.as_mut() {
                gpu.submit()?;
            }
        }

        if ctx.quit {
            self.running = false;
        }

        self.engine.input.end_frame();
        Ok(())
    }

    /// Runs `steps` frames, or until a hook requests a quit when `steps` is
    /// `None`.
    pub fn run(&mut self, steps: Option<u64>) -> Result<()> {
        self.start()?;
        let mut remaining = steps;
        while self.running {
            self.frame()?;
            if let Some(n) = &mut remaining {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    break;
                }
            }
        }
        self.stop()
    }

    /// Deterministic fixed-step run with no window, no vsync and no present.
    pub fn run_headless(&mut self, steps: u64) -> Result<()> {
        self.running = true;
        self.run(Some(steps))
    }

    pub fn stop(&mut self) -> Result<()> {
        for plugin in &mut self.plugins {
            plugin.on_stop(&mut self.engine);
        }
        Ok(())
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
