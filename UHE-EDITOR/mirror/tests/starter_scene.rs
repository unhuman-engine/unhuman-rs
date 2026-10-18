//! The scene a fresh editor launches with, and what File > New builds.

use std::path::PathBuf;

use meats::Scene;
use mirror::Editor;

fn assets() -> vault::Assets {
    vault::Assets::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))
}

fn has<T: meats::hecs::Component>(scene: &Scene, entity: meats::Entity) -> bool {
    scene.get(entity).is_some_and(|entry| entry.has::<T>())
}

#[test]
fn the_starter_scene_is_not_empty_and_has_the_pieces() {
    let mut scene = Scene::new();
    mirror::starter_scene(&mut scene);

    let entities = scene.entities();
    assert_eq!(entities.len(), 3, "a camera, a light and a model");

    // The camera is deliberately not primary: a primary scene camera would take
    // the view away from the editor's orbit camera.
    assert!(scene.primary_camera().is_none());
    assert!(entities
        .iter()
        .any(|entity| has::<meats::Camera>(&scene, *entity)));
    assert!(entities
        .iter()
        .any(|entity| has::<meats::Light>(&scene, *entity)));
    let model = entities
        .iter()
        .find(|entity| has::<meats::MeshRenderer>(&scene, **entity))
        .expect("a model entity");
    let entry = scene.get(*model).unwrap();
    assert_eq!(
        entry.get::<meats::MeshRenderer>().unwrap().mesh_path,
        "models/Sphere.glb"
    );
}

#[test]
fn new_starter_scene_replaces_what_was_there() {
    let mut scene = Scene::new();
    scene.spawn("Leftover");

    let mut editor = Editor::new();
    editor.new_starter_scene(&mut scene, &mut assets());

    let names: Vec<String> = scene
        .entities()
        .into_iter()
        .map(|entity| {
            scene
                .get(entity)
                .and_then(|entry| entry.tag())
                .unwrap_or_default()
        })
        .collect();
    assert!(!names.contains(&"Leftover".to_string()));
    assert!(names.contains(&"Camera".to_string()));
    // The mesh path resolved against the repository assets.
    assert_eq!(scene.meshes().len(), 1);
}
