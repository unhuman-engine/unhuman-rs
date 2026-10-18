//! Hot reload and content-addressed registry behaviour, with no GPU.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use vault::{Assets, Error, Reloader};

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("vault-reload-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("scratch directory");
        Self { root }
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.root.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("scratch parent");
        }
        fs::write(&path, contents).expect("scratch file");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn a_rewritten_file_is_detected_and_the_id_is_bumped() {
    let scratch = Scratch::new("bump");
    scratch.write("notes.txt", "one");

    let mut assets = Assets::new(&scratch.root);
    let before = assets.load("notes.txt").expect("notes.txt loads");
    let before_hash = assets.hash(before).expect("a ready asset has an identity");
    assert_eq!(assets.text(before), Some("one"));

    // Zero interval so the first poll is always due; the default is throttled.
    let mut reloader = Reloader::new(Duration::ZERO);
    reloader.track(&assets, "notes.txt").expect("track");
    assert_eq!(reloader.watcher().len(), 1);

    scratch.write("notes.txt", "two");
    let reloaded = reloader.reload_changed_blocking(&mut assets);
    assert_eq!(reloaded.len(), 1, "one file changed");
    let (path, result) = &reloaded[0];
    assert_eq!(path, "notes.txt");
    let after = *result.as_ref().expect("reload succeeds");

    assert_ne!(after, before);
    assert_eq!(after.generation(), before.generation() + 1);
    assert_eq!(assets.text(after), Some("two"));
    assert_ne!(
        assets.hash(after).expect("identity"),
        before_hash,
        "changed bytes are a changed identity"
    );
    assert_eq!(assets.find("notes.txt"), Some(after));
    assert!(!assets.contains(before), "the stale id stops resolving");
}

#[test]
fn a_removed_file_is_invalidated_and_reported() {
    let scratch = Scratch::new("removed");
    let file = scratch.write("gone.txt", "here");

    let mut assets = Assets::new(&scratch.root);
    let id = assets.load("gone.txt").expect("gone.txt loads");

    let mut reloader = Reloader::new(Duration::ZERO);
    reloader.track(&assets, "gone.txt").expect("track");

    fs::remove_file(&file).expect("remove file");
    let reloaded = reloader.reload_changed_blocking(&mut assets);
    assert_eq!(reloaded.len(), 1);
    assert!(matches!(reloaded[0].1, Err(Error::NotFound(_))));
    assert!(!assets.contains(id));
    assert_eq!(assets.find("gone.txt"), None);
}

#[test]
fn an_unchanged_file_does_not_reload() {
    let scratch = Scratch::new("unchanged");
    scratch.write("stable.txt", "same");

    let mut assets = Assets::new(&scratch.root);
    let id = assets.load("stable.txt").expect("loads");

    let mut reloader = Reloader::new(Duration::ZERO);
    reloader.track(&assets, "stable.txt").expect("track");

    assert!(reloader.reload_changed_blocking(&mut assets).is_empty());
    assert_eq!(assets.find("stable.txt"), Some(id), "the id is untouched");
}

#[test]
fn identical_bytes_share_one_decode_and_evict_to_the_capacity() {
    let scratch = Scratch::new("cache");
    scratch.write("a.txt", "shared");
    scratch.write("b.txt", "shared");
    scratch.write("c.txt", "third");
    scratch.write("d.txt", "fourth");

    let mut assets = Assets::with_cache_capacity(&scratch.root, 2);

    let a = assets.load("a.txt").expect("a loads");
    let b = assets.load("b.txt").expect("b loads");
    // Different paths are different ids, but one content address.
    assert_ne!(a, b);
    assert_eq!(assets.hash(a), assets.hash(b));
    assert_eq!(
        assets.cached(),
        1,
        "the second path reused the first decode"
    );

    let _ = assets.load("c.txt").expect("c loads");
    assert_eq!(assets.cached(), 2);
    let _ = assets.load("d.txt").expect("d loads");
    // Capacity is a bound, not a suggestion: the least recently used is gone.
    assert_eq!(assets.cached(), 2);
    assert_eq!(assets.cache_capacity(), 2);
}

#[test]
fn a_virtual_path_maps_to_its_real_file_and_a_missing_one_errors() {
    let scratch = Scratch::new("vfs");
    scratch.write("textures/wall.txt", "pixels");

    let mut assets = Assets::new(&scratch.root);
    let id = assets
        .load("textures/wall.txt")
        .expect("resolves via the vfs");

    let physical = assets
        .vfs()
        .resolve("textures/wall.txt")
        .expect("real path");
    assert_eq!(physical, scratch.root.join("textures/wall.txt"));
    assert_eq!(assets.path(id), Some("textures/wall.txt"));
    assert_eq!(assets.text(id), Some("pixels"));

    let missing = assets.load("textures/absent.txt").expect_err("absent");
    assert!(matches!(missing, Error::NotFound(path) if path == "textures/absent.txt"));
}
