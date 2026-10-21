//! Per-pass GPU timings on Vulkan, through `chassis::Gpu`.
//!
//! The seam returns pass names owned by the frame, so this pins that a real
//! frame produces named entries and a frame time after timing is enabled. A
//! skipped run prints why, so a machine without a timestamp-capable queue is
//! never mistaken for a passing one.

use bifrost::{Bifrost, EditorCamera, ViewSource};
use chassis::Extent2d;
use meats::Scene;
use stormbreaker::vulkan_gpu::VulkanGpu;
use vault::Assets;

fn extent() -> Extent2d {
    Extent2d {
        width: 64,
        height: 64,
    }
}

fn open() -> Option<VulkanGpu> {
    match VulkanGpu::create_headless(c"stormbreaker-gpu-timing", extent()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("SKIP: no vulkan device: {error}");
            None
        }
    }
}

/// One frame with timing on yields named passes and a positive frame time.
#[test]
fn a_frame_reports_named_pass_times() {
    let Some(mut gpu) = open() else { return };
    if !gpu.enable_gpu_timing() {
        eprintln!("SKIP: the vulkan queue cannot write timestamps");
        return;
    }

    let mut assets = Assets::new(".");
    let scene = Scene::new();
    let camera = EditorCamera::new(45.0, 16.0 / 9.0, 0.1, 1000.0);
    let mut renderer = Bifrost::new(&mut gpu, &mut assets).expect("bifrost builds");
    let jobs = grind::Jobs::new();

    renderer.set_gpu_timing(true);
    renderer
        .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
        .expect("timed frame");

    let timings = renderer.gpu_passes();
    assert!(
        !timings.is_empty(),
        "timing was enabled and a frame rendered, so passes must be named",
    );
    for timing in timings {
        assert!(
            !timing.name.is_empty(),
            "a graph pass has a name, and the seam must carry it",
        );
        assert!(
            timing.ms.is_finite() && timing.ms >= 0.0,
            "{} reported {} ms",
            timing.name,
            timing.ms,
        );
    }

    let frame_ms = renderer
        .gpu_frame_ms()
        .expect("a collected frame has a whole-frame GPU time");
    assert!(
        frame_ms.is_finite() && frame_ms > 0.0 && frame_ms < 1000.0,
        "a 64x64 frame is not zero and not a minute: {frame_ms} ms",
    );

    // The same values through the trait, which is what the editor reads.
    assert_eq!(gpu.gpu_timings().len(), timings.len());
    assert_eq!(gpu.gpu_frame_ms(), Some(frame_ms));

    renderer.destroy(&mut gpu);
}
