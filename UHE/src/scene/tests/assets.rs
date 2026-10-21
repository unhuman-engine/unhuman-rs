//! Scene asset resolution: the synchronous path and the worker path.

use std::time::{Duration, Instant};

use meats::{AudioSource, Scene};
use vault::{AssetState, Assets};

fn fixture(bytes: &[u8]) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "meats-async-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("the fixture directory");
    std::fs::write(root.join("shot.wav"), bytes).expect("the fixture file");
    root
}

#[test]
fn an_async_resolve_hands_out_a_pending_handle() {
    let root = fixture(b"not really a wav");
    let mut scene = Scene::new();
    let ear = scene.spawn("Ear");
    scene.insert(
        ear,
        AudioSource {
            clip_path: "shot.wav".to_string(),
            ..Default::default()
        },
    );
    let mut assets = Assets::new(&root);
    let jobs = grind::Jobs::new();

    let errors = scene.resolve_assets_async(&jobs, &mut assets);
    assert!(errors.is_empty(), "{errors:?}");

    let clip = scene
        .get(ear)
        .expect("the ear")
        .get::<AudioSource>()
        .expect("a sound")
        .clip
        .expect("the path resolved");
    assert_eq!(assets.state(clip), Some(AssetState::Pending));
    assert!(
        assets.bytes(clip).is_none(),
        "a pending handle has no payload yet"
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    while assets.state(clip) == Some(AssetState::Pending) {
        assert!(Instant::now() < deadline, "the worker did not finish");
        std::thread::sleep(Duration::from_millis(2));
        assets.poll();
    }
    assert_eq!(assets.state(clip), Some(AssetState::Ready));
    assert_eq!(assets.bytes(clip), Some(&b"not really a wav"[..]));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_synchronous_resolve_is_ready_when_it_returns() {
    let root = fixture(b"not really a wav");
    let mut scene = Scene::new();
    let ear = scene.spawn("Ear");
    scene.insert(
        ear,
        AudioSource {
            clip_path: "shot.wav".to_string(),
            ..Default::default()
        },
    );
    let mut assets = Assets::new(&root);

    let errors = scene.resolve_assets(&mut assets);
    assert!(errors.is_empty(), "{errors:?}");

    let clip = scene
        .get(ear)
        .expect("the ear")
        .get::<AudioSource>()
        .expect("a sound")
        .clip
        .expect("the path resolved");
    assert_eq!(assets.state(clip), Some(AssetState::Ready));

    let _ = std::fs::remove_dir_all(&root);
}
