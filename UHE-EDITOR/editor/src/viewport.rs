//! The center viewport.
//!
//! The editor window belongs to gpui and the engine runs headless
//! (`engine::create_gpu_from_process` hands back `WgpuGpu::headless`, and
//! `chassis::Gpu::swapchain()` is `None` by design). The frame cannot be
//! presented by the engine, so it is read back: `bifrost` copies the composed
//! image into a readable buffer each frame and hands over a
//! [`bifrost::FrameImage`], and this module turns that into the gpui image the
//! panel draws.

use std::sync::Arc;

use bifrost::FrameImage;
use gpui::RenderImage;
use image::{Frame, ImageBuffer, Rgba};
use smallvec::SmallVec;

/// Shown in the middle of the viewport until the first frame is read back.
pub const MISSING_FRAME_REASON: &str =
    "no engine image yet: the renderer has not read a frame back";

/// A readback frame as a gpui image.
///
/// Takes the frame by value: `from_raw` wants the buffer, and a `clone` of a
/// full frame here would be a second CPU copy on top of the readback's own.
///
/// gpui's `RenderImage` is BGRA, which is the composition target's format, so
/// the bytes go in unswizzled. The buffer is already tightly packed
/// (`width * height * 4`), which is what `from_raw` needs.
pub fn render_image(frame: FrameImage) -> Arc<RenderImage> {
    let buffer: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width, frame.height, frame.bgra)
            .expect("a readback frame is width * height * 4 bytes");
    Arc::new(RenderImage::new(SmallVec::from_const([Frame::new(buffer)])))
}

/// The renderer's name, or why there is none.
pub fn backend_label(backend: Option<&str>) -> &str {
    backend.unwrap_or("no backend")
}

/// `60.0 fps`, for the toolbar and the status bar.
pub fn fps_label(fps: f32) -> String {
    format!("{fps:.1} fps")
}

/// `16.61 ms`, the frame the stats were sampled over.
pub fn frame_time_label(frame_ms: f32) -> String {
    format!("{frame_ms:.2} ms")
}

/// Backend, fps and frame time as one line for the status bar.
pub fn frame_status(backend: Option<&str>, fps: f32, frame_ms: f32) -> String {
    format!(
        "{} · {} · {}",
        backend_label(backend),
        fps_label(fps),
        frame_time_label(frame_ms)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_keeps_one_decimal() {
        assert_eq!(fps_label(60.0), "60.0 fps");
        assert_eq!(fps_label(59.949), "59.9 fps");
        assert_eq!(fps_label(0.0), "0.0 fps");
    }

    #[test]
    fn frame_time_keeps_two_decimals() {
        assert_eq!(frame_time_label(16.606), "16.61 ms");
        assert_eq!(frame_time_label(0.0), "0.00 ms");
    }

    #[test]
    fn status_names_the_backend_or_says_there_is_none() {
        assert_eq!(
            frame_status(Some("wgpu"), 60.0, 16.61),
            "wgpu · 60.0 fps · 16.61 ms"
        );
        assert_eq!(backend_label(None), "no backend");
        assert!(frame_status(None, 0.0, 0.0).starts_with("no backend"));
    }

    #[test]
    fn the_missing_frame_reason_names_a_cause() {
        assert!(MISSING_FRAME_REASON.contains("no engine image"));
    }

    #[test]
    fn a_frame_becomes_an_image_of_its_size() {
        let frame = FrameImage {
            width: 2,
            height: 3,
            bgra: vec![0; 2 * 3 * 4],
        };
        let image = render_image(frame);
        assert_eq!(image.as_bytes(0).map(<[u8]>::len), Some(2 * 3 * 4));
    }

    #[test]
    #[should_panic(expected = "width * height * 4")]
    fn a_frame_that_is_not_its_size_panics() {
        let frame = FrameImage {
            width: 2,
            height: 2,
            bgra: vec![0; 4],
        };
        let _ = render_image(frame);
    }
}
