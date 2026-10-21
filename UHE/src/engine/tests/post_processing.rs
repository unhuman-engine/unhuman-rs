//! Post-processing, measured in the pixels the CPU receives.

use bifrost::passes::READBACK_LAG;
use bifrost::EditorCamera;
use chassis::backend::BackendChoice;
use meats::{Light, MeshRenderer, Scene};

mod common;
use common::{mean_luminance, Harness};

#[test]
fn a_brighter_exposure_produces_a_brighter_frame() {
    let mut ran = 0;

    for choice in BackendChoice::all() {
        if !choice.is_compiled_in() {
            eprintln!("SKIP: {choice:?}: not compiled in");
            continue;
        }
        let Some(mut harness) = Harness::open(choice, "exposure", |context, scratch, assets| {
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

        harness.renderer.set_exposure(0.5);
        harness.render(READBACK_LAG as usize + 1);
        let dark = mean_luminance(&harness.image());

        harness.renderer.set_exposure(2.0);
        harness.render(READBACK_LAG as usize + 1);
        let bright = mean_luminance(&harness.image());

        harness.renderer.set_exposure(0.0);
        harness.render(READBACK_LAG as usize + 1);
        let black = mean_luminance(&harness.image());

        assert!(dark > 0.0, "{context}: exposure 0.5 rendered a black frame");
        assert!(
            bright > dark * 1.5,
            "{context}: exposure 2.0 ({bright}) is not clearly brighter than 0.5 ({dark})"
        );
        assert!(
            black < 0.5,
            "{context}: exposure 0.0 is not black ({black})"
        );
        ran += 1;
        harness.destroy();
    }

    assert!(ran > 0, "no backend opened, so exposure proved nothing");
}
