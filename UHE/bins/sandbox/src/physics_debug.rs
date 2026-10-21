//! The physics debug view's source and push site.
//!
//! `Engine::physics` is a `dyn PhysicsBackend`, and that trait has no debug hook:
//! only `Physics3D` can build the view. This plugin installs the same world the
//! engine steps while keeping a typed handle, so `push` can call `debug_view`
//! and hand the segments to bifrost between frames.

use std::sync::{Arc, Mutex, MutexGuard};

use adam::{App, Engine, PhysicsBackend, Plugin, Result};
use bifrost::BifrostPlugin;
use femur::{Physics3D, PhysicsDebug};
use meats::Scene;

/// Installs a shared `Physics3D` as the engine's backend.
///
/// Only the `--physics-debug` path registers this; otherwise `FemurPlugin`
/// installs an ordinary one and nothing here exists to read it.
pub struct PhysicsDebugPlugin {
    physics: Arc<Mutex<Physics3D>>,
}

impl PhysicsDebugPlugin {
    pub fn new() -> Self {
        Self {
            physics: Arc::new(Mutex::new(Physics3D::new())),
        }
    }

    pub fn debug_view(&self) -> PhysicsDebug {
        self.lock().debug_view()
    }

    fn lock(&self) -> MutexGuard<'_, Physics3D> {
        self.physics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Default for PhysicsDebugPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for PhysicsDebugPlugin {
    fn name(&self) -> &'static str {
        "femur"
    }

    fn build(&mut self, eng: &mut Engine) -> Result<()> {
        eng.physics = Some(Box::new(SharedPhysics(Arc::clone(&self.physics))));
        Ok(())
    }
}

/// The engine's half of the shared world: it owns the step, the plugin owns the
/// view.
struct SharedPhysics(Arc<Mutex<Physics3D>>);

impl PhysicsBackend for SharedPhysics {
    fn name(&self) -> &'static str {
        "rapier3d"
    }

    fn step(&mut self, scene: &mut Scene, dt: f32) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .step(scene, dt);
    }

    fn body_count(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .body_count()
    }
}

/// Hands this frame's debug view to bifrost. A no-op without `--physics-debug`.
pub fn push(app: &mut App) {
    let Some(physics) = app.plugin::<PhysicsDebugPlugin>() else {
        return;
    };
    let debug = physics.debug_view();
    log::debug!(
        "sandbox: physics debug {} segments",
        debug.shapes.len() + debug.contacts.len() + debug.broad_phase.len()
    );
    if let Some(renderer) = app.plugin_mut::<BifrostPlugin>() {
        renderer.set_physics_debug(debug);
    }
}
