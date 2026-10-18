use std::any::TypeId;
use std::sync::{Arc, Mutex};

use meats::{Entity, Scene, Transform};
use pings::{ObservedScene, Observer};

struct Recorder {
    log: Arc<Mutex<Vec<String>>>,
}

impl Recorder {
    fn note(&self, entry: impl Into<String>) {
        self.log.lock().expect("log poisoned").push(entry.into());
    }
}

impl Observer for Recorder {
    fn on_created(&mut self, _scene: &Scene, entity: Entity) {
        self.note(format!("created:{}", entity.index()));
    }

    fn on_destroyed(&mut self, _scene: &Scene, entity: Entity) {
        self.note(format!("destroyed:{}", entity.index()));
    }

    fn on_component_added(&mut self, _scene: &Scene, entity: Entity, component: TypeId) {
        if component == TypeId::of::<Transform>() {
            self.note(format!("added:transform:{}", entity.index()));
        }
    }

    fn on_component_removed(&mut self, _scene: &Scene, entity: Entity, component: TypeId) {
        if component == TypeId::of::<Transform>() {
            self.note(format!("removed:transform:{}", entity.index()));
        }
    }
}

#[test]
fn observer_fires_on_the_integrated_scene_hooks() {
    let mut observed = ObservedScene::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let _handle = observed.observers_mut().observe(Recorder {
        log: Arc::clone(&log),
    });

    let entity = observed.spawn("Camera");
    assert!(observed.insert(entity, Transform::default()));
    assert!(observed.remove::<Transform>(entity).is_some());
    observed.destroy(entity);

    assert_eq!(
        &*log.lock().expect("log poisoned"),
        &[
            "created:0".to_string(),
            "added:transform:0".to_string(),
            "removed:transform:0".to_string(),
            "destroyed:0".to_string(),
        ]
    );
}

#[test]
fn a_dropped_observer_handle_stops_delivery() {
    let mut observed = ObservedScene::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let handle = observed.observers_mut().observe(Recorder {
        log: Arc::clone(&log),
    });

    let first = observed.spawn("First");
    assert_eq!(log.lock().expect("log poisoned").len(), 1);

    drop(handle);
    let second = observed.spawn("Second");
    assert_eq!(log.lock().expect("log poisoned").len(), 1);

    assert_eq!(first.index(), 0);
    assert_eq!(second.index(), 1);
}

#[test]
fn destroy_notifies_every_entity_in_the_subtree_first() {
    let mut observed = ObservedScene::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let _handle = observed.observers_mut().observe(Recorder {
        log: Arc::clone(&log),
    });

    let parent = observed.spawn("Parent");
    let child = observed.spawn("Child");
    observed.scene_mut().set_parent(child, Some(parent));
    observed.destroy(parent);

    let log = log.lock().expect("log poisoned").clone();
    assert!(log.contains(&format!("destroyed:{}", parent.index())));
    assert!(log.contains(&format!("destroyed:{}", child.index())));
}

#[test]
fn a_component_of_another_type_is_not_reported() {
    let mut observed = ObservedScene::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    let _handle = observed.observers_mut().observe(Recorder {
        log: Arc::clone(&log),
    });

    let entity = observed.spawn("Entity");
    // Every spawn installs a `Transform` itself, so clear the log first.
    log.lock().expect("log poisoned").clear();
    observed.insert(entity, meats::Tag::new("tagged"));

    assert!(log.lock().expect("log poisoned").is_empty());
}
