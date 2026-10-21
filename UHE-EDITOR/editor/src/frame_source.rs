//! The seam between the editor and the renderer's readback.
//!
//! `bifrost` reads a ring slot `READBACK_LAG` submissions back, so its render
//! submits the current frame and hands over the last completed one. The editor
//! never needs the frame from *this* render, only the newest one the renderer
//! has finished, and [`FrameSource::poll`] does not wait for one to arrive.
//!
//! TODO: bifrost should grow `Bifrost::take_frame_image() -> Option<FrameImage>`
//! (ownership, so the editor does not copy the frame a second time), and
//! [`BifrostSource`] should call it instead of cloning.

use bifrost::{Bifrost, BifrostPlugin, FrameImage};

/// The newest completed frame, or nothing if none is ready. Must not block.
pub trait FrameSource {
    fn poll(&mut self) -> Option<FrameImage>;
}

/// Adapts `bifrost`'s last completed frame to [`FrameSource`].
///
/// `poll` maps the newest frame the renderer has already read back; it exists so
/// the editor consumes the interface rather than `frame_image` directly.
pub struct BifrostSource<'a> {
    plugin: &'a mut BifrostPlugin,
    last_frame: u64,
}

impl<'a> BifrostSource<'a> {
    pub fn new(plugin: &'a mut BifrostPlugin) -> Self {
        // `frame_index` starts at zero and the first `render` bumps it to one,
        // so zero means "nothing seen yet".
        Self {
            plugin,
            last_frame: 0,
        }
    }
}

impl FrameSource for BifrostSource<'_> {
    fn poll(&mut self) -> Option<FrameImage> {
        let sequence = self
            .plugin
            .renderer()
            .map(Bifrost::frame_index)
            .unwrap_or(0);
        if sequence == self.last_frame {
            return None;
        }
        self.last_frame = sequence;
        // TODO: `take_frame_image` hands the bytes over without this clone.
        self.plugin.frame_image().cloned()
    }
}
