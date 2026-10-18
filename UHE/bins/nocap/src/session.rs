//! A deterministic fixed-step driver over `Engine` and the plugin set.
//!
//! `adam::App` advances frames from the wall clock (`Time::tick`), so two runs
//! of the same input take different numbers of fixed steps and diverge. This
//! driver injects the step instead, using the same `Engine`, the same `Plugin`
//! hooks and the same order `App::frame` uses, minus the clock.

use std::time::Duration;

use adam::{Engine, FrameCtx, Plugin, Result};
use meats::{MeshRenderer, Tag, Transform};
use pings::Event;

pub struct Session {
    engine: Engine,
    plugins: Vec<Box<dyn Plugin>>,
    frame: u64,
}

impl Session {
    pub fn new() -> Self {
        Self {
            engine: Engine::headless(),
            plugins: Vec::new(),
            frame: 0,
        }
    }

    /// Registration order is execution order, as in [`adam::App::add`].
    #[allow(clippy::should_implement_trait)]
    pub fn add<P: Plugin>(mut self, plugin: P) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    pub fn start(&mut self) -> Result<()> {
        for plugin in &mut self.plugins {
            plugin.build(&mut self.engine)?;
        }
        for plugin in &mut self.plugins {
            plugin.on_start(&mut self.engine)?;
        }
        Ok(())
    }

    /// One frame at an injected step: input, one fixed simulation step, the
    /// update hooks, then the per-frame hook.
    pub fn step(&mut self, dt: Duration, events: &[Event]) {
        self.engine.input.begin_frame();
        for event in events {
            self.engine.events.push(event.clone());
        }
        for event in self.engine.events.drain() {
            self.engine.input.apply(&event);
        }

        self.engine.simulate_fixed_step(dt);
        let mut ctx = FrameCtx {
            frame: self.frame + 1,
            delta: dt,
            step: dt,
            scale: 1.0,
            quit: false,
        };
        for plugin in &mut self.plugins {
            plugin.on_update(&mut self.engine, &mut ctx);
        }
        for plugin in &mut self.plugins {
            plugin.on_frame(&mut self.engine, &mut ctx);
        }
        self.engine.input.end_frame();
        self.frame += 1;
    }

    pub fn stop(&mut self) {
        for plugin in &mut self.plugins {
            plugin.on_stop(&mut self.engine);
        }
    }

    pub fn plugin_names(&self) -> Vec<String> {
        self.plugins
            .iter()
            .map(|plugin| plugin.name().to_string())
            .collect()
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// A hash of the scene's observable state: which entities exist, their
    /// tags, transforms and mesh visibility. Handles are excluded, because a
    /// path that fails to resolve is still the same scene; entity `Id`s are
    /// excluded because `Scene::spawn` seeds them from the wall clock, so they
    /// differ between two runs of the same input.
    ///
    /// FNV-1a, hand-rolled: a hash that a capture file depends on must not
    /// change with the toolchain, and `DefaultHasher` explicitly may.
    pub fn digest(&self) -> u64 {
        let scene = &self.engine.scene;
        let mut hash = Fnv::new();
        let entities = scene.entities();
        hash.write_u64(entities.len() as u64);

        for entity in entities {
            let Some(view) = scene.get(entity) else {
                continue;
            };
            if let Some(tag) = view.get::<Tag>() {
                hash.write_str(&tag.tag);
            }
            if let Some(transform) = view.get::<Transform>() {
                for value in [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.scale.x,
                    transform.scale.y,
                    transform.scale.z,
                ] {
                    hash.write_f32(value);
                }
            }
            if let Some(mesh) = view.get::<MeshRenderer>() {
                hash.write_str(&mesh.mesh_path);
                hash.write_u64(u64::from(mesh.visible));
            }
        }
        hash.finish()
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn write_f32(&mut self, value: f32) {
        // Bits, not a decimal: a hash of a float must see every bit that differs.
        self.write_u64(u64::from(value.to_bits()));
    }

    fn write_str(&mut self, value: &str) {
        self.write_u64(value.len() as u64);
        self.write(value.as_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    /// The determinism the capture layer rests on: same input, same step, same
    /// digests, with no clock and no device in the loop.
    #[test]
    fn two_identical_runs_produce_identical_digests() {
        let run = || {
            let mut session = crate::plugins::session();
            session.start().unwrap();
            let mut digests = Vec::new();
            for frame in 0..40_u64 {
                let events = crate::input::scripted(frame);
                session.step(Duration::from_micros(crate::runner::STEP_MICROS), &events);
                digests.push(session.digest());
            }
            session.stop();
            digests
        };

        assert_eq!(run(), run());
    }
}
