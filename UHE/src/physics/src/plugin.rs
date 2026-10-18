//! The plugin that installs the backend.

use adam::{Engine, Plugin, Result};

use crate::world::Physics3D;

/// Installs `Physics3D` as the engine's physics backend.
///
/// The backend hangs off `Engine::physics`, not off this plugin, because the engine's
/// fixed step drives it (`Engine::simulate_fixed_step`): one owner of the step, and a
/// build with no physics is `physics: None`, not a stubbed plugin.
pub struct FemurPlugin;

impl Plugin for FemurPlugin {
    fn name(&self) -> &'static str {
        "femur"
    }

    fn build(&mut self, eng: &mut Engine) -> Result<()> {
        eng.physics = Some(Box::new(Physics3D::new()));
        Ok(())
    }
}
