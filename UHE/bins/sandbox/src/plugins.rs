//! The composition root's one job: name the plugins, in order, and mount assets.
//!
//! `sandbox` is not an engine. It picks the plugin set and hands it to
//! [`adam::App`]; every hook and every subsystem lives in a crate.

use std::path::Path;

use adam::App;
use aimlab::AimLabPlugin;
use bifrost::BifrostPlugin;
use femur::FemurPlugin;
use mirror::MirrorPlugin;
use wire::WirePlugin;

use crate::physics_debug::PhysicsDebugPlugin;

/// The game plugin set, in registration order, with the repository's `assets/`
/// mounted so relative scene and aimlab paths resolve exactly as in the editor.
///
/// The physics backend depends on `--physics-debug`: the two installers are
/// mutually exclusive, and the debug one has to be the world the engine steps
/// for its view to be current.
pub fn app_with_physics_debug(physics_debug: bool) -> App {
    let app = App::new()
        .add(AimLabPlugin::new())
        .add(BifrostPlugin::new().with_hud())
        .add(WirePlugin::new());
    let app = if physics_debug {
        app.add(PhysicsDebugPlugin::new())
    } else {
        app.add(FemurPlugin)
    };
    let mut app = app.add(MirrorPlugin::new());

    app.engine_mut().assets.mount("assets");
    app
}

/// Replace the engine's scene with one loaded from disk, then resolve its asset
/// paths against the registry the plugins will use.
pub fn load_scene(app: &mut App, path: &Path) -> Result<(), String> {
    let adam::Engine { scene, assets, .. } = app.engine_mut();
    *scene = meats::load(path).map_err(|error| format!("--scene {}: {error}", path.display()))?;
    for error in scene.resolve_assets(assets) {
        log::warn!("sandbox: {error}");
    }
    Ok(())
}
