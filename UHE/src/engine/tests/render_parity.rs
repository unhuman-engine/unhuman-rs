//! One fixed frame, both real backends, read back and compared.
//!
//! `backend_parity.rs` proves the RHI surface agrees. It never records a
//! renderer frame, so a backend whose composition is black or whose readback
//! copies the wrong image still passes. This file renders the same scene through
//! `Bifrost` on wgpu and Vulkan and compares the pixels the CPU receives.
//!
//! A backend that cannot open prints one `SKIP:` line and is left out. A
//! backend that opens but draws nothing, reads back nothing, or agrees with the
//! other only by accident is a failure. Returning green without a frame is the
//! exact false pass this suite exists to remove.

use bifrost::passes::READBACK_LAG;
use bifrost::EditorCamera;
use chassis::backend::BackendChoice;
use meats::{Light, MeshRenderer, Scene};

mod common;
use common::{clear_bgra, difference, drawn_pixels, pixel, Harness};

/// Agreement bounds for a matching frame.
///
/// Two rasterisers can disagree at a triangle or grid-line edge by a sample.
/// The mean is the real guard: a different frame moves most pixels, and a
/// handful of edge texels does not. The outlier fraction stops a backend from
/// hiding a large area of disagreement behind a permissive maximum.
const MAX_MEAN_DIFFERENCE: f32 = 3.0;
const MAX_CHANNEL_DIFFERENCE: u8 = 128;
const MAX_OUTLIER_FRACTION: f32 = 0.02;
const OUTLIER_DELTA: u8 = 32;

/// One fixed scene, rendered on every backend that opens, read back and
/// checked: something drew, a known pixel is the composited clear colour, and
/// the two images agree.
#[test]
fn a_fixed_frame_is_drawn_and_read_back_on_every_backend() {
    let mut frames = Vec::new();

    for choice in BackendChoice::all() {
        if !choice.is_compiled_in() {
            eprintln!("SKIP: {choice:?}: not compiled in");
            continue;
        }
        let Some(mut harness) = Harness::open(choice, "parity", |context, scratch, assets| {
            scratch.write_triangle();
            let id = assets
                .load("tri.gltf")
                .unwrap_or_else(|error| panic!("{context}: load triangle: {error}"));

            let mut scene = Scene::new();
            let mesh = scene.spawn("tri");
            assert!(
                scene.insert(
                    mesh,
                    MeshRenderer {
                        mesh: Some(id),
                        ..MeshRenderer::default()
                    }
                ),
                "{context}: mesh entity"
            );
            // Default Light is directional along local -Z, which lights the
            // triangle's +Z face.
            let light = scene.spawn("light");
            assert!(scene.insert(light, Light::default()), "{context}: light");

            let mut camera = EditorCamera::new(45.0, 1.0, 0.1, 1000.0);
            // A shallow tilt keeps the top of the frame above the horizon, so a
            // known corner is the clear colour and not the grid. The pitch has
            // to be smaller than half the vertical field of view.
            camera.set_pitch(0.15);
            camera.set_yaw(0.6);

            (scene, camera)
        }) else {
            continue;
        };
        let context = harness.context.clone();

        // The ring reads READBACK_LAG submissions behind, so the first readable
        // frame needs that many plus one. Every frame is identical.
        harness.render(READBACK_LAG as usize + 1);
        let image = harness.image();
        harness.check_extent(&image);

        let drawn = drawn_pixels(&image);
        assert!(
            drawn > image.width as usize * image.height as usize / 20,
            "{context}: only {drawn} pixels differ from the clear colour; the frame drew nothing",
        );

        // A corner above the horizon: the composition should have passed the
        // scene's clear colour through unchanged.
        let clear = clear_bgra();
        let corner = pixel(&image, 1, 1);
        for (channel, (actual, expected)) in corner.iter().zip(clear).enumerate() {
            assert!(
                actual.abs_diff(expected) <= 6,
                "{context}: corner channel {channel} is {actual}, expected the \
                 clear colour {expected} (got {corner:?}, clear {clear:?})",
            );
        }

        // The centre is inside the triangle and must not be clear either.
        let centre = pixel(&image, image.width / 2, image.height / 2);
        assert_ne!(
            centre, clear,
            "{context}: the mesh did not reach the centre"
        );

        eprintln!(
            "{context}: {} of {} pixels drawn",
            drawn,
            image.width * image.height
        );
        frames.push((context, image));
        harness.destroy();
    }

    if frames.is_empty() {
        eprintln!("SKIP: no real GPU backend opened");
        return;
    }

    if let [(wgpu_name, wgpu), (vulkan_name, vulkan)] = frames.as_slice() {
        let (mean, max, outliers) = difference(wgpu, vulkan, OUTLIER_DELTA);
        eprintln!(
            "PARITY {wgpu_name}/{vulkan_name}: mean {mean:.3} max {max} outliers \
             {:.3}% (allowed {MAX_MEAN_DIFFERENCE:.1}/{MAX_CHANNEL_DIFFERENCE}/\
             {:.1}%)",
            outliers * 100.0,
            MAX_OUTLIER_FRACTION * 100.0,
        );
        assert!(
            mean <= MAX_MEAN_DIFFERENCE,
            "{wgpu_name} and {vulkan_name} differ by a mean of {mean:.3}; a backend \
             is drawing a different frame",
        );
        assert!(
            max <= MAX_CHANNEL_DIFFERENCE,
            "{wgpu_name} and {vulkan_name} differ by {max} on one channel; a \
             backend is drawing a different frame",
        );
        assert!(
            outliers <= MAX_OUTLIER_FRACTION,
            "{wgpu_name} and {vulkan_name} disagree past {OUTLIER_DELTA} on \
             {:.1}% of channels; that is a different image, not edge noise",
            outliers * 100.0,
        );
    }
}

/// The readback ring returns the older frame while the newest submission is in
/// flight.
///
/// This is the executable form of engine.md's P0 "fixed when": a frame reads the
/// pixels of a submission that is several frames old, not the one it
/// just issued. [`ReadbackRing`](bifrost::passes::ReadbackRing) stores each
/// frame's [`chassis::Submission`] and reads through
/// [`Resources::read_buffer_after`], which waits only for that token.
///
/// A ring that waited for the newest submission, or read the wrong slot, would
/// return the second frame's pixels here instead of the first's.
#[test]
fn the_readback_ring_returns_the_older_frame() {
    let mut ran = 0;

    for choice in BackendChoice::all() {
        if !choice.is_compiled_in() {
            continue;
        }
        let Some(mut harness) = Harness::open(choice, "ring", |context, scratch, assets| {
            scratch.write_triangle();
            let id = assets
                .load("tri.gltf")
                .unwrap_or_else(|error| panic!("{context}: load triangle: {error}"));

            let mut scene = Scene::new();
            let mesh = scene.spawn("tri");
            assert!(
                scene.insert(
                    mesh,
                    MeshRenderer {
                        mesh: Some(id),
                        ..MeshRenderer::default()
                    }
                ),
                "{context}: mesh entity"
            );
            let light = scene.spawn("light");
            assert!(scene.insert(light, Light::default()), "{context}: light");

            let mut camera = EditorCamera::new(45.0, 1.0, 0.1, 1000.0);
            camera.set_pitch(0.15);
            camera.set_yaw(0.6);

            (scene, camera)
        }) else {
            continue;
        };
        let context = harness.context.clone();

        // Frame A is static while the ring warms. Frame B changes the camera;
        // it uses a different per-frame slot, so rebuilding its sets does not
        // release any slot A is still reading.
        harness.render(READBACK_LAG as usize);
        harness.camera.set_yaw(-0.4);
        harness.render(1);

        // B is the newest submission and need not have retired; the readback
        // must still return A, not wait for B.
        let older = harness.image();

        // A readable frame exists at this point only because an older
        // submission retired; B itself has not been read.
        assert!(
            drawn_pixels(&older) > older.width as usize * older.height as usize / 20,
            "{context}: the older frame is empty",
        );

        // Advance until B is the ring's readable frame, and confirm the two
        // scenes are actually distinguishable. Without this the assertion above
        // would pass on a ring that read the newest slot while the scene stood
        // still.
        harness.render(READBACK_LAG as usize);
        let newest = harness.image();
        let (mean, max, _) = difference(&older, &newest, OUTLIER_DELTA);
        assert!(
            mean > 2.0 && max > 16,
            "{context}: the two camera frames differ by mean {mean:.3} max {max}; \
             the ring lag was not observable",
        );

        eprintln!(
            "{context}: readback after the newest frame returned the older frame \
             (mean difference to the eventual newest {mean:.3})"
        );
        harness.destroy();
        ran += 1;
    }

    if ran == 0 {
        eprintln!("SKIP: no real GPU backend opened");
    }
}
