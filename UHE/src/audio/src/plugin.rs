//! The plugin that installs [`AudioEngine`] and points the listener at the camera.

use adam::{Engine, FrameCtx, Listener, Plugin, Result};
use meats::Scene;

use crate::engine::{AudioEngine, DEFAULT_VOICES};

/// Installs [`AudioEngine`] into `Engine::audio` and drives it once a frame.
pub struct WirePlugin {
    capacity: usize,
}

impl WirePlugin {
    pub fn new() -> Self {
        Self {
            capacity: DEFAULT_VOICES,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { capacity }
    }
}

impl Default for WirePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for WirePlugin {
    fn name(&self) -> &'static str {
        "wire"
    }

    fn build(&mut self, eng: &mut Engine) -> Result<()> {
        eng.audio = Some(Box::new(AudioEngine::with_capacity(self.capacity)));
        Ok(())
    }

    /// Once a frame, not once per fixed step: the listener follows the camera, which
    /// moves per frame, and a frame whose accumulator reached no step runs no
    /// substeps at all. The frame's real step is what the Doppler shift integrates
    /// over.
    fn on_frame(&mut self, eng: &mut Engine, ctx: &mut FrameCtx) {
        let Some(audio) = eng.audio.as_mut() else {
            return;
        };
        audio.set_listener(listener_from(&eng.scene));
        audio.update(&eng.scene, &eng.assets, ctx.step_seconds());
    }
}

/// The primary camera's pose as the ear.
pub fn listener_from(scene: &Scene) -> Listener {
    let Some(camera) = scene.primary_camera() else {
        return Listener::default();
    };
    let transform = scene.world_transform(camera);
    Listener::new(
        transform.w_axis.truncate(),
        // Right-handed: a camera looks down its own -Z.
        -transform.z_axis.truncate(),
        transform.y_axis.truncate(),
    )
}
