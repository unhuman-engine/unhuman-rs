//! The plugin set a capture is recorded from and replayed against.
//!
//! Audio is deliberately absent: a sound device is the one piece of state a
//! replay machine cannot be assumed to have, and its absence changes nothing
//! the scene digest covers.

use aimlab::AimLabPlugin;
use femur::FemurPlugin;

use crate::session::Session;

/// The demo composition, in registration order, with `assets/` mounted so the
/// aim lab's relative mesh paths resolve as they do in the windowed binary.
pub fn session() -> Session {
    let mut session = Session::new().add(AimLabPlugin::new()).add(FemurPlugin);
    session.engine_mut().assets.mount("assets");
    session
}
