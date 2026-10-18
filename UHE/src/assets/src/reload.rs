use std::time::Duration;

use grind::Jobs;

use crate::assets::Assets;
use crate::error::{Error, Result};
use crate::id::AssetId;
use crate::watch::{Change, Watcher};

/// A path whose old id was released and reloaded, with its new id or the reason
/// the reload failed.
pub type Reloaded = (String, Result<AssetId>);

/// Drives hot reload: watch paths, and when a file's bytes change, release the
/// asset built from the old bytes and load the new ones.
///
/// [`Reloader::reload_changed`] is the frame-loop entry point. It does no
/// blocking work itself: each changed path is re-submitted on `grind`, and the
/// new ids arrive through [`Assets::poll`] like any other asynchronous load, so
/// a reload cannot stall the thread that draws. A caller with no job system uses
/// [`Reloader::reload_changed_blocking`], which decodes on this thread.
pub struct Reloader {
    watcher: Watcher,
}

impl Reloader {
    /// A reloader that re-checks watched paths at most every `interval`.
    pub fn new(interval: Duration) -> Self {
        Self {
            watcher: Watcher::new(interval),
        }
    }

    pub fn watcher(&self) -> &Watcher {
        &self.watcher
    }

    pub fn watcher_mut(&mut self) -> &mut Watcher {
        &mut self.watcher
    }

    /// Starts watching `path`. A path that does not resolve now cannot be
    /// watched, so nothing is recorded and an [`Error::Watch`] is returned.
    pub fn track(&mut self, assets: &Assets, path: &str) -> Result<()> {
        self.watcher.watch(assets.vfs(), path).map(|_| ())
    }

    /// Polls the watcher and reloads every changed path on the job system. An
    /// empty result means the interval has not elapsed or nothing changed.
    pub fn reload_changed(&mut self, assets: &mut Assets, jobs: &Jobs) -> Vec<Reloaded> {
        let changes = self.watcher.poll(assets.vfs());
        self.apply(assets, changes, |assets, path| {
            assets.reload_async(jobs, path)
        })
    }

    /// The synchronous counterpart of [`Reloader::reload_changed`], for a
    /// registry with no job system.
    pub fn reload_changed_blocking(&mut self, assets: &mut Assets) -> Vec<Reloaded> {
        let changes = self.watcher.poll(assets.vfs());
        self.apply(assets, changes, |assets, path| assets.reload(path))
    }

    fn apply(
        &mut self,
        assets: &mut Assets,
        changes: Vec<(String, Change)>,
        mut reload: impl FnMut(&mut Assets, &str) -> Result<AssetId>,
    ) -> Vec<Reloaded> {
        let mut reloaded = Vec::with_capacity(changes.len());
        for (path, change) in changes {
            let result = if change.after.is_some() {
                reload(assets, &path)
            } else {
                // A removed file has no new content to load: drop the old asset
                // so the stale id stops resolving, and report why.
                if let Some(id) = assets.find(&path) {
                    assets.release(id);
                }
                Err(Error::NotFound(path.clone()))
            };
            reloaded.push((path, result));
        }
        reloaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::thread;
    use std::time::Duration;

    use crate::id::AssetKind;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("vault-reload-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("scratch directory");
        root
    }

    #[test]
    fn an_async_reload_is_submitted_and_delivered_by_poll() {
        let root = scratch("async");
        let file = root.join("a.txt");
        fs::write(&file, "one").expect("write");

        let mut assets = Assets::new(&root);
        let before = assets.load("a.txt").expect("loads");

        let mut reloader = Reloader::new(Duration::ZERO);
        reloader.track(&assets, "a.txt").expect("track");
        fs::write(&file, "two").expect("rewrite");

        let jobs = Jobs::new();
        let reloaded = reloader.reload_changed(&mut assets, &jobs);
        assert_eq!(reloaded.len(), 1);
        let after = *reloaded[0].1.as_ref().expect("submitted to the job system");
        assert_ne!(after, before);

        // The worker decodes off-thread; the frame loop drains it here.
        let mut ready = Vec::new();
        for _ in 0..200 {
            ready = assets.poll();
            if !ready.is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(ready.len(), 1, "poll delivered the reload");
        assert_eq!(ready[0].0, after);
        assert!(matches!(ready[0].1, Ok(AssetKind::Text)));
        assert_eq!(assets.text(after), Some("two"));

        let _ = fs::remove_dir_all(&root);
    }
}
