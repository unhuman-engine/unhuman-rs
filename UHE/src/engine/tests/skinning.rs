//! Skinning on a real device: a two-bone quad, two clip times, CPU reference.
//!
//! The model is generated into a scratch GLB-style glTF beside a binary buffer,
//! so the test needs no checked-in asset. The CPU reference applies the same
//! palette the renderer uploaded, which makes a missing shader attribute or an
//! unbound palette show up as a moved silhouette, not as a shader compile log.

use bifrost::passes::READBACK_LAG;
use bifrost::EditorCamera;
use chassis::backend::BackendChoice;
use glam::{Mat4, Vec3};
use meats::{Light, MeshRenderer, Scene};

mod common;
use common::{clear_bgra, drawn_pixels, pixel, project, Harness, HEIGHT, READBACK_WARMUP, WIDTH};

const VERTICES: [[f32; 3]; 4] = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
];
const JOINTS: [[u16; 4]; 4] = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]];
const WEIGHTS: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
];

fn camera() -> EditorCamera {
    let mut camera = EditorCamera::new(45.0, 1.0, 0.1, 100.0);
    camera.set_focal_point(Vec3::new(0.5, 0.0, 0.0));
    camera.set_distance(4.0);
    camera.set_pitch(0.0);
    camera.set_yaw(std::f32::consts::PI);
    camera
}

fn drawn_bounds(image: &bifrost::FrameImage) -> Option<((u32, u32), (u32, u32))> {
    let clear = clear_bgra();
    let mut min = (u32::MAX, u32::MAX);
    let mut max = (0, 0);
    let mut any = false;
    for y in 0..image.height {
        for x in 0..image.width {
            let value = pixel(image, x, y);
            if value
                .iter()
                .zip(clear)
                .all(|(actual, expected)| actual.abs_diff(expected) <= 2)
            {
                continue;
            }
            any = true;
            min.0 = min.0.min(x);
            min.1 = min.1.min(y);
            max.0 = max.0.max(x);
            max.1 = max.1.max(y);
        }
    }
    any.then_some((min, max))
}

fn cpu_bounds(palette: &[[[f32; 4]; 4]], view_projection: Mat4) -> ((u32, u32), (u32, u32)) {
    let matrices: Vec<Mat4> = palette.iter().map(Mat4::from_cols_array_2d).collect();
    let mut min = (u32::MAX, u32::MAX);
    let mut max = (0, 0);
    for (position, (joints, weights)) in VERTICES.iter().zip(JOINTS.iter().zip(WEIGHTS.iter())) {
        let mut skinned = Vec3::ZERO;
        for slot in 0..4 {
            skinned += matrices[joints[slot] as usize]
                .transform_point3(Vec3::from_array(*position))
                * weights[slot];
        }
        let (x, y) = project(view_projection, skinned, WIDTH, HEIGHT)
            .unwrap_or_else(|| panic!("skinned vertex {skinned} is outside the view"));
        min.0 = min.0.min(x);
        min.1 = min.1.min(y);
        max.0 = max.0.max(x);
        max.1 = max.1.max(y);
    }
    (min, max)
}

fn close(actual: (u32, u32), expected: (u32, u32)) -> bool {
    actual.0.abs_diff(expected.0) <= 2 && actual.1.abs_diff(expected.1) <= 2
}

#[test]
fn a_two_bone_quad_skins_at_two_clip_times() {
    let mut ran = 0;

    for choice in BackendChoice::all() {
        if !choice.is_compiled_in() {
            eprintln!("SKIP: {choice:?}: not compiled in");
            continue;
        }
        let Some(mut harness) = Harness::open(choice, "skin", |context, scratch, assets| {
            scratch.write_skinned_quad("skin");
            let id = assets
                .load("skin.gltf")
                .unwrap_or_else(|error| panic!("{context}: load skinned quad: {error}"));

            let mut scene = Scene::new();
            let mesh = scene.spawn("skin");
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
            (scene, camera())
        }) else {
            continue;
        };
        let context = harness.context.clone();
        let entity = harness.scene.meshes()[0].entity;
        let mesh = harness.scene.meshes()[0].mesh;

        harness.hide_grid();
        harness.render(READBACK_WARMUP + 1);
        eprintln!(
            "{context}: unskinned drew {}",
            drawn_pixels(&harness.image())
        );
        assert!(
            harness
                .renderer
                .attach_skinned(
                    &mut *harness.gpu,
                    &harness.assets,
                    entity,
                    mesh,
                    Some("Bend")
                )
                .unwrap_or_else(|error| panic!("{context}: attach skin: {error}")),
            "{context}: the synthetic model carried no skeleton or clip"
        );

        // The clip's t = 0 key is the bind pose; t = 1 moves BoneB one unit.
        harness.render(READBACK_WARMUP + 1);
        let bind = harness.image();
        harness.check_extent(&bind);
        let bind_palette = harness
            .renderer
            .skinned_palette(entity)
            .expect("attached palette")
            .to_vec();
        let expected_bind = cpu_bounds(&bind_palette, harness.camera.view_projection());

        harness.renderer.seek_skinned(entity, 1.0);
        harness.render(READBACK_WARMUP + 1);
        let bent = harness.image();
        harness.check_extent(&bent);
        let bent_palette = harness
            .renderer
            .skinned_palette(entity)
            .expect("attached palette")
            .to_vec();
        let expected_bent = cpu_bounds(&bent_palette, harness.camera.view_projection());

        eprintln!(
            "{context}: expected bind {expected_bind:?}, palette {:?}",
            &bind_palette[..bind_palette.len().min(2)]
        );
        let actual_bind = drawn_bounds(&bind).unwrap_or_else(|| {
            panic!(
                "{context}: the bind-pose quad drew {} pixels, expected {expected_bind:?}",
                drawn_pixels(&bind)
            )
        });
        let actual_bent = drawn_bounds(&bent).unwrap_or_else(|| {
            panic!(
                "{context}: the deformed quad drew {} pixels",
                drawn_pixels(&bent)
            )
        });

        eprintln!(
            "{context}: bind {actual_bind:?} expected {expected_bind:?}; \
             bent {actual_bent:?} expected {expected_bent:?}"
        );
        assert!(
            close(actual_bind.0, expected_bind.0) && close(actual_bind.1, expected_bind.1),
            "{context}: bind-pose silhouette {actual_bind:?} misses the CPU reference \
             {expected_bind:?}"
        );
        assert!(
            close(actual_bent.0, expected_bent.0) && close(actual_bent.1, expected_bent.1),
            "{context}: deformed silhouette {actual_bent:?} misses the CPU reference \
             {expected_bent:?}"
        );
        let bind_width = actual_bind.1 .0.saturating_sub(actual_bind.0 .0);
        let bent_width = actual_bent.1 .0.saturating_sub(actual_bent.0 .0);
        assert!(
            bent_width > bind_width + 4,
            "{context}: the clip did not stretch the quad (bind width {bind_width}, bent {bent_width})"
        );

        harness.destroy();
        ran += 1;
    }

    if ran == 0 {
        eprintln!("SKIP: no real GPU backend opened");
    }
    // Keep the imported lag constant meaningful to a reader and to the test's
    // warmup count: the image read is that many submissions behind.
    let _ = READBACK_LAG;
}
