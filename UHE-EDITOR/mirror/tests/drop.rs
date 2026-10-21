//! Dropped files: the virtual path the registry gets, and the entity each kind
//! spawns.

use std::path::PathBuf;

use meats::Scene;
use mirror::Editor;

/// A registry rooted at the repository's `assets/`, which is two levels above
/// the crate directory.
fn assets() -> vault::Assets {
    vault::Assets::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))
}

fn assets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

#[test]
fn a_model_becomes_a_mesh_entity_that_resolves() {
    let mut scene = Scene::new();
    let mut editor = Editor::new();
    let mut assets = assets();
    let jobs = grind::Jobs::new();

    let entity = editor.add_model(
        &mut scene,
        &mut assets,
        &jobs,
        "models/Sphere.glb",
        "Sphere.glb",
    );

    assert_eq!(scene.entities(), vec![entity]);
    // One draw sample means the mesh handle resolved and the renderer has it.
    assert_eq!(scene.meshes().len(), 1);
    assert_eq!(editor.selected_entity(), Some(entity));
    assert!(editor.history.can_undo());
}

#[test]
fn an_image_becomes_a_sprite_entity_holding_the_path() {
    let mut scene = Scene::new();
    let mut editor = Editor::new();
    let mut assets = assets();
    let jobs = grind::Jobs::new();

    // The repository ships no image, so this asserts the wiring: a sprite
    // entity carrying the path. Resolution into a handle is `resolve_assets`,
    // which warns for a path the registry does not have.
    let entity = editor.add_image(
        &mut scene,
        &mut assets,
        &jobs,
        "textures/missing.png",
        "missing.png",
    );

    let entry = scene.get(entity).expect("the drop spawned an entity");
    assert!(entry.has::<meats::SpriteRenderer>());
    assert_eq!(
        entry.get::<meats::SpriteRenderer>().unwrap().texture_path,
        "textures/missing.png"
    );
    assert_eq!(editor.selected_entity(), Some(entity));
}

#[test]
fn a_path_under_a_root_stays_relative_to_it() {
    let mut assets = assets();
    let path = assets_root().join("models/Sphere.glb");

    assert_eq!(
        mirror::import_path(&mut assets, &path).as_deref(),
        Some("models/Sphere.glb")
    );
}

#[test]
fn a_path_outside_every_root_mounts_its_directory() {
    let mut assets = assets();
    let directory = std::env::temp_dir();
    let path = directory.join("dropped_texture.png");

    assert_eq!(
        mirror::import_path(&mut assets, &path).as_deref(),
        Some("dropped_texture.png")
    );
    assert!(assets.roots().iter().any(|root| root == &directory));
}
