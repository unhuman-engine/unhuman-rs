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

/// The game plugin set, in registration order, with the repository's `assets/`
/// mounted so relative scene and aimlab paths resolve exactly as in the editor.
pub fn app() -> App {
    let mut app = App::new()
        .add(AimLabPlugin::new())
        .add(BifrostPlugin::new())
        .add(WirePlugin::new())
        .add(FemurPlugin)
        .add(MirrorPlugin::new());

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
