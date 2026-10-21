//! Skinning on a real device: a two-bone quad, two clip times, CPU reference.
//!
//! The model is generated into a scratch GLB-style glTF beside a binary buffer,
//! so the test needs no checked-in asset. The CPU reference applies the same
//! palette the renderer uploaded, which makes a missing shader attribute or an
//! unbound palette show up as a moved silhouette, not as a shader compile log.

use bifrost::passes::READBACK_LAG;
use bifrost::{Bifrost, EditorCamera};
use chassis::backend::BackendChoice;
use glam::{Mat4, Vec3};
use meats::{Animator, Light, MeshRenderer, Scene};

mod common;
use common::{
    clear_bgra, drawn_pixels, pixel, project, Harness, Scratch, HEIGHT, READBACK_WARMUP, WIDTH,
};

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

fn push_f32(bin: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) -> usize {
    let offset = bin.len();
    for value in values {
        bin.extend_from_slice(&value.to_le_bytes());
    }
    offset
}

fn push_u16(bin: &mut Vec<u8>, values: impl IntoIterator<Item = u16>) -> usize {
    let offset = bin.len();
    for value in values {
        bin.extend_from_slice(&value.to_le_bytes());
    }
    offset
}

/// A skinned quad with Idle, Fire and Reload clips of 1.0, 0.5 and 2.0 seconds.
fn write_clip_model(scratch: &Scratch) {
    let positions = [
        [-1.0f32, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let normals = [[0.0f32, 0.0, 1.0]; 4];
    let joints = [[0u16, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]];
    let weights = [[1.0f32, 0.0, 0.0, 0.0]; 4];
    let indices = [0u16, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2];
    let inverse_bind = [
        [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0,
        ],
    ];
    let clips = [
        ("Idle", [0.0f32, 1.0], [[1.0f32, 0.0, 0.0], [1.1, 0.0, 0.0]]),
        ("Fire", [0.0, 0.5], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ("Reload", [0.0, 2.0], [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    ];

    let mut bin = Vec::new();
    let position_offset = push_f32(&mut bin, positions.iter().flatten().copied());
    let normal_offset = push_f32(&mut bin, normals.iter().flatten().copied());
    let joint_offset = push_u16(&mut bin, joints.iter().flatten().copied());
    let weight_offset = push_f32(&mut bin, weights.iter().flatten().copied());
    let index_offset = push_u16(&mut bin, indices.iter().copied());
    let bind_offset = push_f32(&mut bin, inverse_bind.iter().flatten().copied());
    let mut anim_offsets = Vec::new();
    for (_, times, translations) in &clips {
        let time_offset = push_f32(&mut bin, times.iter().copied());
        let translation_offset = push_f32(&mut bin, translations.iter().flatten().copied());
        anim_offsets.push((time_offset, translation_offset));
    }
    let byte_length = bin.len();
    std::fs::write(scratch.root().join("clips.bin"), &bin).expect("clips buffer");

    let mut views = vec![
        format!(
            r#"{{ "buffer": 0, "byteOffset": {position_offset}, "byteLength": 48, "target": 34962 }}"#
        ),
        format!(
            r#"{{ "buffer": 0, "byteOffset": {normal_offset}, "byteLength": 48, "target": 34962 }}"#
        ),
        format!(
            r#"{{ "buffer": 0, "byteOffset": {joint_offset}, "byteLength": 32, "target": 34962 }}"#
        ),
        format!(
            r#"{{ "buffer": 0, "byteOffset": {weight_offset}, "byteLength": 64, "target": 34962 }}"#
        ),
        format!(
            r#"{{ "buffer": 0, "byteOffset": {index_offset}, "byteLength": 24, "target": 34963 }}"#
        ),
        format!(r#"{{ "buffer": 0, "byteOffset": {bind_offset}, "byteLength": 128 }}"#),
    ];
    let mut accessors = vec![
        r#"{ "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [ -1.0, -1.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ] }"#.to_string(),
        r#"{ "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3" }"#.to_string(),
        r#"{ "bufferView": 2, "componentType": 5123, "count": 4, "type": "VEC4" }"#.to_string(),
        r#"{ "bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC4" }"#.to_string(),
        r#"{ "bufferView": 4, "componentType": 5123, "count": 12, "type": "SCALAR" }"#.to_string(),
        r#"{ "bufferView": 5, "componentType": 5126, "count": 2, "type": "MAT4" }"#.to_string(),
    ];
    let mut animations = Vec::new();
    for (index, (name, times, _)) in clips.iter().enumerate() {
        let (time_offset, translation_offset) = anim_offsets[index];
        let view = views.len();
        views.push(format!(
            r#"{{ "buffer": 0, "byteOffset": {time_offset}, "byteLength": 8 }}"#
        ));
        views.push(format!(
            r#"{{ "buffer": 0, "byteOffset": {translation_offset}, "byteLength": 24 }}"#
        ));
        let input = accessors.len();
        accessors.push(format!(
            r#"{{ "bufferView": {view}, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [ 0.0 ], "max": [ {} ] }}"#,
            times[1]
        ));
        accessors.push(format!(
            r#"{{ "bufferView": {}, "componentType": 5126, "count": 2, "type": "VEC3" }}"#,
            view + 1
        ));
        animations.push(format!(
            r#"{{ "name": "{name}", "samplers": [ {{ "input": {input}, "output": {}, "interpolation": "LINEAR" }} ], "channels": [ {{ "sampler": 0, "target": {{ "node": 2, "path": "translation" }} }} ] }}"#,
            input + 1
        ));
    }

    let gltf = format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [ {{ "nodes": [ 0, 3 ] }} ],
  "nodes": [
    {{ "name": "Root", "children": [ 1 ] }},
    {{ "name": "BoneA", "children": [ 2 ] }},
    {{ "name": "BoneB", "translation": [ 1.0, 0.0, 0.0 ] }},
    {{ "name": "clips", "mesh": 0, "skin": 0 }}
  ],
  "meshes": [ {{ "name": "clips", "primitives": [ {{ "attributes": {{ "POSITION": 0, "NORMAL": 1, "JOINTS_0": 2, "WEIGHTS_0": 3 }}, "indices": 4 }} ] }} ],
  "skins": [ {{ "name": "Skin", "inverseBindMatrices": 5, "joints": [ 1, 2 ], "skeleton": 0 }} ],
  "animations": [ {} ],
  "buffers": [ {{ "uri": "clips.bin", "byteLength": {byte_length} }} ],
  "bufferViews": [ {} ],
  "accessors": [ {} ]
}}"#,
        animations.join(", "),
        views.join(", "),
        accessors.join(", ")
    );
    std::fs::write(scratch.root().join("clips.gltf"), gltf).expect("clips glTF");
}

fn set_scene_clip(scene: &mut Scene, entity: meats::Entity, clip: &str, looping: bool) {
    let mut entry = scene.get_mut(entity).expect("entity exists");
    let animator = entry.get_mut::<Animator>().expect("the animator");
    animator.clip_path.clear();
    animator.clip_path.push_str(clip);
    animator.looping = looping;
}

fn apply_scene_animator(renderer: &mut Bifrost, scene: &Scene, entity: meats::Entity) {
    let entry = scene.get(entity).expect("entity exists");
    let animator = entry.get::<Animator>().expect("the animator");
    renderer.apply_animator(entity, &animator);
}

#[test]
fn the_plugin_switches_clips_by_name_and_keeps_the_last_valid_one() {
    let mut ran = 0;

    for choice in BackendChoice::all() {
        if !choice.is_compiled_in() {
            eprintln!("SKIP: {choice:?}: not compiled in");
            continue;
        }
        let Some(mut harness) = Harness::open(choice, "clip-switch", |context, scratch, assets| {
            write_clip_model(scratch);
            let id = assets
                .load("clips.gltf")
                .unwrap_or_else(|error| panic!("{context}: load clips: {error}"));

            let mut scene = Scene::new();
            let mesh = scene.spawn("clips");
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
            assert!(
                scene.insert(
                    mesh,
                    Animator {
                        clip_path: "Idle".into(),
                        playing: true,
                        looping: true,
                        ..Animator::default()
                    }
                ),
                "{context}: animator"
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

        assert!(
            harness
                .renderer
                .attach_skinned(
                    &mut *harness.gpu,
                    &harness.assets,
                    entity,
                    mesh,
                    Some("Idle")
                )
                .unwrap_or_else(|error| panic!("{context}: attach skin: {error}")),
            "{context}: the synthetic model carried no skeleton or clip"
        );
        assert_eq!(
            harness.renderer.skinned_animation_index(entity),
            Some(0),
            "{context}: Idle index"
        );
        assert_eq!(
            harness.renderer.skinned_clip_duration(entity),
            Some(1.0),
            "{context}: Idle duration"
        );

        set_scene_clip(&mut harness.scene, entity, "Fire", false);
        apply_scene_animator(&mut harness.renderer, &harness.scene, entity);
        assert_eq!(
            harness.renderer.skinned_animation_index(entity),
            Some(1),
            "{context}: Fire index"
        );
        assert_eq!(
            harness.renderer.skinned_clip_duration(entity),
            Some(0.5),
            "{context}: Fire duration"
        );
        assert_eq!(
            harness.renderer.skinned_animation_time(entity),
            Some(0.0),
            "{context}: a switch restarts the play head"
        );

        harness.renderer.advance_skinned(&harness.scene, 0.25);
        let time = harness
            .renderer
            .skinned_animation_time(entity)
            .expect("attached play head");
        assert!(
            (time - 0.25).abs() < 1e-6,
            "{context}: Fire play head {time}"
        );

        set_scene_clip(&mut harness.scene, entity, "Missing", true);
        apply_scene_animator(&mut harness.renderer, &harness.scene, entity);
        assert_eq!(
            harness.renderer.skinned_animation_index(entity),
            Some(1),
            "{context}: an unknown name keeps Fire playing"
        );

        set_scene_clip(&mut harness.scene, entity, "Reload", true);
        apply_scene_animator(&mut harness.renderer, &harness.scene, entity);
        assert_eq!(
            harness.renderer.skinned_animation_index(entity),
            Some(2),
            "{context}: Reload index"
        );
        assert_eq!(
            harness.renderer.skinned_clip_duration(entity),
            Some(2.0),
            "{context}: Reload duration"
        );

        // A submitted frame retires the setup uploads before teardown.
        harness.hide_grid();
        harness.render(READBACK_WARMUP + 1);

        harness.destroy();
        ran += 1;
    }

    if ran == 0 {
        eprintln!("SKIP: no real GPU backend opened");
    }
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
