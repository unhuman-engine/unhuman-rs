//! What a save has to survive across builds, edits and hostile hierarchy.

use std::path::Path;

use glam::Vec3;
use meats::{from_yaml, load, to_yaml, Entity, MeshRenderer, Scene, SpriteRenderer, Transform};

fn component<T: Clone + meats::hecs::Component>(scene: &Scene, entity: Entity) -> T {
    (*scene.get(entity).unwrap().get::<T>().unwrap()).clone()
}

fn entity_named(scene: &Scene, tag: &str) -> Entity {
    scene
        .entities()
        .into_iter()
        .find(|entity| scene.get(*entity).unwrap().tag().as_deref() == Some(tag))
        .unwrap_or_else(|| panic!("no entity tagged `{tag}`"))
}

fn round_trip(scene: &Scene) -> Scene {
    from_yaml(&to_yaml(scene).expect("serializes")).expect("deserializes")
}

#[test]
fn an_entity_id_outlives_the_file() {
    let mut scene = Scene::new();
    let root = scene.spawn("Root");
    let child = scene.spawn("Child");
    scene.set_parent(child, Some(root));
    let root_id = scene.id_of(root).expect("spawned with an id");
    let child_id = scene.id_of(child).expect("spawned with an id");

    let reloaded = round_trip(&scene);
    let root = entity_named(&reloaded, "Root");
    let child = entity_named(&reloaded, "Child");

    assert_eq!(reloaded.id_of(root), Some(root_id));
    assert_eq!(reloaded.id_of(child), Some(child_id));
    // The hierarchy is keyed by that id, not by file order.
    assert_eq!(reloaded.parent(child), Some(root));
    assert_eq!(reloaded.children(root), &[child]);
}

#[test]
fn a_component_added_by_a_newer_editor_is_named_on_the_next_save() {
    let yaml = "\
Scene: Newer
Entities:
  - Entity: 7
    TagComponent:
      Tag: Thing
    TransformComponent:
      Translation: [1, 2, 3]
      Rotation: [0, 0, 0]
      Scale: [1, 1, 1]
    TeleporterComponent:
      Speed: 3
";
    let scene = from_yaml(yaml).expect("loads despite the unknown component");
    let entity = scene.entities()[0];
    assert_eq!(
        scene.dropped_components(entity),
        &["TeleporterComponent".to_string()]
    );

    let saved = to_yaml(&scene).expect("serializes");
    assert!(saved.contains("DroppedComponents:"), "{saved}");
    assert!(saved.contains("TeleporterComponent"), "{saved}");

    // The note survives another load/save so it cannot quietly disappear.
    let reloaded = from_yaml(&saved).expect("reloads");
    let entity = reloaded.entities()[0];
    assert_eq!(
        reloaded.dropped_components(entity),
        &["TeleporterComponent".to_string()]
    );
    assert!(to_yaml(&reloaded).unwrap().contains("TeleporterComponent"));
}

#[test]
fn a_component_this_build_no_longer_has_is_named_on_the_next_save() {
    // The shipped scene carries the 2D components this 3D build removed.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/scenes/box2d.unhuman");
    let scene = load(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let entity = scene
        .entities()
        .into_iter()
        .find(|entity| {
            scene
                .dropped_components(*entity)
                .iter()
                .any(|name| name == "RigidBody2DComponent")
        })
        .expect("box2d scene carries RigidBody2DComponent");

    let saved = to_yaml(&scene).expect("serializes");
    assert!(saved.contains("DroppedComponents:"), "{saved}");
    assert!(saved.contains("RigidBody2DComponent"), "{saved}");
    assert!(saved.contains("BoxColliderComponent"), "{saved}");
    assert!(!scene.has::<meats::RigidBody>(entity));
}

#[test]
fn a_component_missing_from_the_file_is_not_invented() {
    let yaml = "\
Scene: Bare
Entities:
  - Entity: 1
    TagComponent:
      Tag: Bare
";
    let scene = from_yaml(yaml).expect("loads");
    let entity = scene.entities()[0];
    assert!(!scene.has::<Transform>(entity));
    assert!(!scene.has::<MeshRenderer>(entity));

    let saved = to_yaml(&scene).expect("serializes");
    assert!(!saved.contains("TransformComponent:"), "{saved}");
    assert!(!saved.contains("CameraComponent:"), "{saved}");
}

#[test]
fn a_component_removed_from_an_entity_does_not_come_back() {
    let mut scene = Scene::new();
    let entity = scene.spawn("Camera");
    scene.insert(entity, meats::Camera::default());
    scene.remove::<meats::Camera>(entity);
    scene.remove::<Transform>(entity);

    let reloaded = round_trip(&scene);
    let entity = entity_named(&reloaded, "Camera");
    assert!(!reloaded.has::<meats::Camera>(entity));
    assert!(!reloaded.has::<Transform>(entity));
}

#[test]
fn a_cycle_in_the_file_loads_as_a_forest_and_saves_once() {
    let yaml = "\
Scene: Cyclic
Entities:
  - Entity: 1
    TagComponent:
      Tag: A
    RelationshipComponent:
      Parent: 2
      Children: [2]
  - Entity: 2
    TagComponent:
      Tag: B
    RelationshipComponent:
      Parent: 1
      Children: [1]
";
    let scene = from_yaml(yaml).expect("a cycle is refused, not fatal");
    assert_eq!(scene.entities().len(), 2);
    let a = entity_named(&scene, "A");
    let b = entity_named(&scene, "B");

    // Exactly one direction can survive; the loader refuses the edge that
    // would close the cycle.
    let edges = [scene.parent(a) == Some(b), scene.parent(b) == Some(a)];
    assert_eq!(edges.iter().filter(|edge| **edge).count(), 1);
    assert!(!scene.is_descendant_of(a, a));
    assert!(!scene.is_descendant_of(b, b));

    let saved = to_yaml(&scene).expect("serializes");
    assert_eq!(saved.matches("Tag: A").count(), 1, "{saved}");
    assert_eq!(saved.matches("Tag: B").count(), 1, "{saved}");
}

#[test]
fn a_reparented_subtree_saves_under_its_new_parent() {
    let mut scene = Scene::new();
    let first = scene.spawn("First");
    let second = scene.spawn("Second");
    let child = scene.spawn("Child");
    scene.set_parent(child, Some(first));
    scene.set_parent(child, Some(second));

    let reloaded = round_trip(&scene);
    let first = entity_named(&reloaded, "First");
    let second = entity_named(&reloaded, "Second");
    let child = entity_named(&reloaded, "Child");

    assert!(reloaded.children(first).is_empty());
    assert_eq!(reloaded.parent(child), Some(second));
    assert_eq!(reloaded.children(second), &[child]);
}

#[test]
fn a_stale_children_list_does_not_override_the_parent_field() {
    let yaml = "\
Scene: Reparented
Entities:
  - Entity: 1
    TagComponent:
      Tag: OldParent
    RelationshipComponent:
      Parent: 0
      Children: [2]
  - Entity: 2
    TagComponent:
      Tag: Child
    RelationshipComponent:
      Parent: 3
      Children: []
  - Entity: 3
    TagComponent:
      Tag: NewParent
    RelationshipComponent:
      Parent: 0
      Children: []
";
    let scene = from_yaml(yaml).expect("loads");
    let old = entity_named(&scene, "OldParent");
    let child = entity_named(&scene, "Child");
    let new = entity_named(&scene, "NewParent");

    assert!(scene.children(old).is_empty());
    assert_eq!(scene.parent(child), Some(new));
    assert_eq!(scene.children(new), &[child]);
}

#[test]
fn a_dangling_parent_leaves_the_entity_a_root() {
    let yaml = "\
Scene: Orphan
Entities:
  - Entity: 1
    TagComponent:
      Tag: Orphan
    RelationshipComponent:
      Parent: 999
      Children: []
";
    let scene = from_yaml(yaml).expect("loads");
    let entity = entity_named(&scene, "Orphan");
    assert_eq!(scene.parent(entity), None);

    let saved = to_yaml(&scene).expect("serializes");
    assert!(saved.contains("Parent: 0"), "{saved}");
}

#[test]
fn floats_survive_text_including_negative_zero() {
    let mut scene = Scene::new();
    let entity = scene.spawn("Numbers");
    scene.insert(
        entity,
        Transform {
            translation: Vec3::new(-0.0, 0.05, 1.0e-7),
            rotation: Vec3::new(f32::MAX, f32::MIN_POSITIVE, 123.456),
            scale: Vec3::new(-1.5, 3.0, -0.0),
        },
    );

    let saved = to_yaml(&scene).expect("serializes");
    assert!(saved.contains("-0.0"), "{saved}");

    let reloaded = from_yaml(&saved).expect("reloads");
    let entity = reloaded.entities()[0];
    let transform: Transform = component(&reloaded, entity);
    assert_eq!(transform.translation.x.to_bits(), (-0.0f32).to_bits());
    assert_eq!(transform.translation.y.to_bits(), 0.05f32.to_bits());
    assert_eq!(transform.translation.z.to_bits(), 1.0e-7f32.to_bits());
    assert_eq!(transform.rotation.x.to_bits(), f32::MAX.to_bits());
    assert_eq!(transform.rotation.y.to_bits(), f32::MIN_POSITIVE.to_bits());
    assert_eq!(transform.rotation.z.to_bits(), 123.456f32.to_bits());
    assert_eq!(transform.scale.x.to_bits(), (-1.5f32).to_bits());
    assert_eq!(transform.scale.z.to_bits(), (-0.0f32).to_bits());
}

#[test]
fn an_unresolved_path_is_reported_per_entity() {
    let mut scene = Scene::new();
    let mesh = scene.spawn("Mesh");
    let sprite = scene.spawn("Sprite");
    scene.insert(
        mesh,
        MeshRenderer {
            mesh_path: "missing/mesh.glb".to_string(),
            ..MeshRenderer::default()
        },
    );
    scene.insert(
        sprite,
        SpriteRenderer {
            texture_path: "missing/sprite.png".to_string(),
            ..SpriteRenderer::default()
        },
    );

    let root = std::env::temp_dir().join(format!("meats-assets-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut assets = vault::Assets::new(&root);
    let errors = scene.resolve_assets(&mut assets);

    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(errors
        .iter()
        .any(|error| error.entity == mesh && error.path == "missing/mesh.glb"));
    assert!(errors
        .iter()
        .any(|error| error.entity == sprite && error.path == "missing/sprite.png"));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_pre_version_file_loads_and_saves_as_the_current_version() {
    let scene = from_yaml("Scene: Legacy\nEntities: []\n").expect("legacy layout loads");
    let saved = to_yaml(&scene).expect("serializes");
    assert!(saved.contains("Format: unhuman"), "{saved}");
    assert!(
        saved.contains(&format!(
            "FormatVersion: {}",
            meats::serialize::FORMAT_VERSION
        )),
        "{saved}"
    );
}

#[test]
fn a_file_from_a_newer_build_is_refused_with_its_version() {
    let yaml = "Scene: Future\nFormatVersion: 999\nEntities: []\n";
    let error = from_yaml(yaml).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("999"), "{message}");
    assert!(message.contains("newer"), "{message}");
}

#[test]
fn a_file_with_a_foreign_format_marker_is_refused() {
    let yaml = "Scene: Other\nFormat: something-else\nEntities: []\n";
    let error = from_yaml(yaml).unwrap_err();
    assert!(error.to_string().contains("Format"), "{error}");
}
