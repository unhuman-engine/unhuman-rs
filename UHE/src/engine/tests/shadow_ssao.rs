//! The shadow map and SSAO, measured in the frames the CPU receives.
//!
//! Both passes are declared and recorded on every frame; these tests render
//! geometry that makes their effect visible in the composed image instead of
//! asserting that a pass exists in the graph.
//!
//! A backend that cannot open prints one counted `SKIP:` line. A backend that
//! opens but draws nothing fails: an empty frame would make every luminance
//! comparison vacuous.

use bifrost::material::Material;
use bifrost::EditorCamera;
use chassis::backend::BackendChoice;
use glam::Vec3;
use meats::{Light, LightKind, MeshRenderer, Scene, Transform};
use vault::{AssetId, Assets};

mod common;
use common::{darker_than, drawn_pixels, project, region_luminance, Harness, HEIGHT, WIDTH};

/// A key the test owns, so the renderer can resolve the material a test attaches
/// to an entity without loading a material file.
const MATERIAL_ID: AssetId = AssetId::from_parts(4096, 1);

fn ground_camera() -> EditorCamera {
    let mut camera = EditorCamera::new(45.0, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);
    camera.set_focal_point(Vec3::new(0.0, 0.7, -0.2));
    camera.set_distance(9.0);
    camera.set_pitch(0.55);
    camera.set_yaw(std::f32::consts::PI);
    camera
}

fn wall_camera() -> EditorCamera {
    let mut camera = EditorCamera::new(90.0, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);
    camera.set_focal_point(Vec3::new(0.0, 0.2, 0.5));
    camera.set_distance(1.0);
    camera.set_pitch(0.0);
    camera.set_yaw(std::f32::consts::PI);
    camera
}

fn light_angle(scene: &mut Scene) {
    let light = scene.spawn("light");
    assert!(
        scene.insert(
            light,
            Light {
                kind: LightKind::Directional,
                ..Light::default()
            }
        ),
        "light component"
    );
    // Local -Z rotated -60 degrees about X travels down and toward -Z.
    assert!(
        scene.insert(
            light,
            Transform {
                rotation: Vec3::new(-60.0, 0.0, 0.0),
                ..Transform::default()
            }
        ),
        "light transform"
    );
}

fn mesh_entity(
    context: &str,
    scene: &mut Scene,
    name: &str,
    mesh: AssetId,
    material: Option<AssetId>,
    translation: Vec3,
) -> meats::Entity {
    let entity = scene.spawn(name);
    assert!(
        scene.insert(
            entity,
            MeshRenderer {
                mesh: Some(mesh),
                material,
                ..MeshRenderer::default()
            }
        ),
        "{context}: {name} mesh"
    );
    assert!(
        scene.insert(
            entity,
            Transform {
                translation,
                ..Transform::default()
            }
        ),
        "{context}: {name} transform"
    );
    entity
}

/// A plane with a unit cube resting on it, lit so the cube's shadow lands on
/// the ground between it and the camera.
fn ground_and_cube(
    context: &str,
    scratch: &common::Scratch,
    assets: &mut Assets,
) -> (Scene, EditorCamera, meats::Entity) {
    scratch.write_plane("ground", 20.0);
    scratch.write_cube("caster", 1.0);

    let ground = assets
        .load("ground.gltf")
        .unwrap_or_else(|error| panic!("{context}: load ground: {error}"));
    let caster = assets
        .load("caster.gltf")
        .unwrap_or_else(|error| panic!("{context}: load caster: {error}"));

    let mut scene = Scene::new();
    mesh_entity(context, &mut scene, "ground", ground, None, Vec3::ZERO);
    let caster = mesh_entity(
        context,
        &mut scene,
        "caster",
        caster,
        None,
        Vec3::new(0.0, 1.0, 0.0),
    );
    light_angle(&mut scene);

    (scene, ground_camera(), caster)
}

/// A wall with a cube between it and the camera, unlit so the only signal in
/// the frame is the occlusion pass. The 0.5 radius only clears the 0.025 NDC
/// depth bias this close, and the cube touches the wall so the kernel reaches it.
fn wall_and_cube(
    context: &str,
    scratch: &common::Scratch,
    assets: &mut Assets,
) -> (Scene, EditorCamera, meats::Entity) {
    scratch.write_wall("wall", 4.0, 0.5);
    scratch.write_cube("caster", 0.2);

    let wall = assets
        .load("wall.gltf")
        .unwrap_or_else(|error| panic!("{context}: load wall: {error}"));
    let caster = assets
        .load("caster.gltf")
        .unwrap_or_else(|error| panic!("{context}: load caster: {error}"));

    let mut scene = Scene::new();
    mesh_entity(
        context,
        &mut scene,
        "wall",
        wall,
        Some(MATERIAL_ID),
        Vec3::ZERO,
    );
    let caster = mesh_entity(
        context,
        &mut scene,
        "caster",
        caster,
        Some(MATERIAL_ID),
        Vec3::new(0.0, 0.1, 0.3),
    );

    (scene, wall_camera(), caster)
}

fn mean_at(
    harness: &Harness,
    image: &bifrost::FrameImage,
    world: Vec3,
    radius: u32,
) -> (u32, u32, f32) {
    let centre = project(
        harness.camera.view_projection(),
        world,
        image.width,
        image.height,
    )
    .unwrap_or_else(|| panic!("{}: {world:?} is not in the frame", harness.context));
    (centre.0, centre.1, region_luminance(image, centre, radius))
}

fn unlit_material() -> Material {
    Material {
        unlit: true,
        double_sided: true,
        base_color_factor: [1.0, 1.0, 1.0, 1.0],
        ..Material::default()
    }
}

#[test]
fn the_shadow_map_darkens_the_ground() {
    let mut ran = 0;
    let mut skipped_backends = 0;

    for choice in BackendChoice::all() {
        if !choice.is_compiled_in() {
            eprintln!("SKIP: {choice:?}: not compiled in");
            skipped_backends += 1;
            continue;
        }

        let mut caster = None;
        let Some(mut harness) = Harness::open(choice, "shadow", |context, scratch, assets| {
            let (scene, camera, entity) = ground_and_cube(context, scratch, assets);
            caster = Some(entity);
            (scene, camera)
        }) else {
            skipped_backends += 1;
            continue;
        };
        let context = harness.context.clone();
        let caster = caster.expect("caster entity");
        harness.hide_grid();

        harness.render(common::READBACK_WARMUP);
        let with_caster = harness.image();
        harness.check_extent(&with_caster);
        assert!(
            drawn_pixels(&with_caster) > WIDTH as usize * HEIGHT as usize / 20,
            "{context}: the shadow scene drew nothing",
        );

        harness.scene.destroy(caster);
        harness.render(common::READBACK_WARMUP);
        let without_caster = harness.image();
        harness.check_extent(&without_caster);

        // The shadow's location follows the light and the caster, so find where
        // removing the caster made the frame darker instead of assuming a
        // pixel. A large drop is the shadow, not a few texels of ambient
        // occlusion.
        let darker = darker_than(&with_caster, &without_caster, 10.0);
        let (x, y) = darker.max_at;
        let shadowed = region_luminance(&with_caster, (x, y), 2);
        let lit = region_luminance(&without_caster, (x, y), 2);

        let far = Vec3::new(3.0, 0.0, 2.5);
        let (_, _, far_with) = mean_at(&harness, &with_caster, far, 2);
        let (_, _, far_without) = mean_at(&harness, &without_caster, far, 2);

        eprintln!(
            "{context}: shadow at ({x},{y}) is {shadowed:.1} with the caster and \
             {lit:.1} without (max drop {:.1}, {} pixels darker; far ground \
             {far_with:.1} vs {far_without:.1})",
            darker.max_drop, darker.count,
        );

        assert!(
            darker.count >= 16,
            "{context}: only {} pixels are darker with the caster; the shadow map \
             did not darken the ground",
            darker.count,
        );
        assert!(
            darker.max_drop > 40.0,
            "{context}: the largest shadow darkening is {:.1}, too small to be the \
             directional light's shadow",
            darker.max_drop,
        );
        assert!(
            lit - shadowed > 12.0,
            "{context}: the shadow region is {shadowed:.1} with the caster and \
             {lit:.1} without",
        );
        assert!(
            (far_with - far_without).abs() < 2.0,
            "{context}: the far ground moved from {far_with:.1} to {far_without:.1}; \
             the difference is not localized to the shadow",
        );

        harness.destroy();
        ran += 1;
    }

    if ran == 0 {
        eprintln!("SKIP: no real GPU backend opened ({skipped_backends} skipped)");
    }
}

#[test]
fn ssao_darkens_the_contact_region() {
    let mut ran = 0;
    let mut skipped_backends = 0;

    for choice in BackendChoice::all() {
        if !choice.is_compiled_in() {
            eprintln!("SKIP: {choice:?}: not compiled in");
            skipped_backends += 1;
            continue;
        }

        let mut caster = None;
        let Some(mut harness) = Harness::open(choice, "ssao", |context, scratch, assets| {
            let (scene, camera, entity) = wall_and_cube(context, scratch, assets);
            caster = Some(entity);
            (scene, camera)
        }) else {
            skipped_backends += 1;
            continue;
        };
        let context = harness.context.clone();
        let caster = caster.expect("caster entity");
        harness.hide_grid();
        harness.renderer.set_material(MATERIAL_ID, unlit_material());

        harness.render(common::READBACK_WARMUP);
        let with_caster = harness.image();
        harness.check_extent(&with_caster);
        assert!(
            drawn_pixels(&with_caster) > WIDTH as usize * HEIGHT as usize / 20,
            "{context}: the SSAO scene drew nothing",
        );

        harness.scene.destroy(caster);
        harness.render(common::READBACK_WARMUP);
        let without_caster = harness.image();
        harness.check_extent(&without_caster);

        // Unlit and with the grid off, removing the cube changes only the depth
        // the AO kernel reads on the wall behind it.
        let darker = darker_than(&with_caster, &without_caster, 3.0);
        let (x, y) = darker.max_at;
        let contact = region_luminance(&with_caster, (x, y), 2);
        let flat = region_luminance(&without_caster, (x, y), 2);

        let far = Vec3::new(0.6, 0.8, 0.5);
        let (_, _, far_with) = mean_at(&harness, &with_caster, far, 2);
        let (_, _, far_without) = mean_at(&harness, &without_caster, far, 2);

        eprintln!(
            "{context}: SSAO at ({x},{y}) is {contact:.1} with the caster and \
             {flat:.1} without (max drop {:.1}, {} pixels darker; far wall \
             {far_with:.1} vs {far_without:.1})",
            darker.max_drop, darker.count,
        );

        assert!(
            darker.count >= 8,
            "{context}: only {} pixels are darker with the caster; SSAO did not \
             darken the contact region",
            darker.count,
        );
        assert!(
            darker.max_drop > 3.0,
            "{context}: the largest SSAO darkening is {:.1}, too small to read as \
             occlusion",
            darker.max_drop,
        );
        assert!(
            flat - contact > 2.0,
            "{context}: the contact region is {contact:.1} with the caster and \
             {flat:.1} without",
        );
        assert!(
            (far_with - far_without).abs() < 2.0,
            "{context}: the far wall moved from {far_with:.1} to {far_without:.1}; \
             the AO difference is not localized",
        );
        assert!(
            darker.count < WIDTH as usize * HEIGHT as usize / 4,
            "{context}: {} of {} pixels changed, so the AO difference is not \
             localized",
            darker.count,
            WIDTH * HEIGHT,
        );

        harness.destroy();
        ran += 1;
    }

    if ran == 0 {
        eprintln!("SKIP: no real GPU backend opened ({skipped_backends} skipped)");
    }
}
