use std::collections::BTreeMap;
use std::fs;
use std::time::{Duration, Instant};

use crate::cache_hash::AssetHash;
use crate::error::{Error, Result};
use crate::vfs::Vfs;

/// How often [`Watcher::poll`] re-reads watched files.
///
/// Polling rather than inotify/kqueue: a native watcher is a platform backend to
/// port and a dependency to carry, and the interval is the one knob that trades
/// reload latency for syscalls. 250 ms is under a frame at 240 Hz, so a write
/// lands in the next frame or two without a watcher thread parked in a syscall.
pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(250);

/// One watched file whose bytes changed since the last poll.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Change {
    /// The content address the file had when it was last seen.
    pub before: AssetHash,
    /// The content address now, or `None` when the file no longer resolves. A
    /// caller must invalidate rather than reload in that case.
    pub after: Option<AssetHash>,
}

/// Watches virtual paths by content address.
///
/// A poll re-resolves and re-reads each watched file and compares its
/// [`AssetHash`]. Reading the bytes rather than stat-ing the modification time
/// is deliberate: mtime is coarse and does not survive a copy, so a same-tick
/// rewrite or a restored checkout would be missed. The cost is a read of every
/// watched file per interval, which is acceptable for the handful of assets a
/// game watches and is the price of a correct answer.
pub struct Watcher {
    interval: Duration,
    next: Option<Instant>,
    watched: BTreeMap<String, AssetHash>,
}

impl Watcher {
    /// A watcher that re-checks at most once per `interval`.
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            next: None,
            watched: BTreeMap::new(),
        }
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    pub fn len(&self) -> usize {
        self.watched.len()
    }

    pub fn is_empty(&self) -> bool {
        self.watched.is_empty()
    }

    /// The watched paths and the content address last seen for each, in path
    /// order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, AssetHash)> {
        self.watched
            .iter()
            .map(|(path, hash)| (path.as_str(), *hash))
    }

    /// Resolves, reads and hashes `path`, then remembers it. An unwatchable path
    /// is an [`Error::Watch`] and nothing is recorded.
    pub fn watch(&mut self, vfs: &Vfs, path: &str) -> Result<AssetHash> {
        let hash = hash_path(vfs, path)?;
        self.watched.insert(path.to_owned(), hash);
        Ok(hash)
    }

    /// Stops watching `path`. Returns whether it was being watched.
    pub fn forget(&mut self, path: &str) -> bool {
        self.watched.remove(path).is_some()
    }

    /// Re-checks every watched path once the interval has elapsed; an empty
    /// result before then. Changed and removed paths are reported and their
    /// record updated.
    pub fn poll(&mut self, vfs: &Vfs) -> Vec<(String, Change)> {
        if self.next.is_some_and(|deadline| Instant::now() < deadline) {
            return Vec::new();
        }
        self.poll_now(vfs)
    }

    /// Re-checks every watched path now, ignoring the interval. The interval is
    /// restarted, so an explicit check does not also trigger the timed one.
    pub fn poll_now(&mut self, vfs: &Vfs) -> Vec<(String, Change)> {
        self.next = Some(Instant::now() + self.interval);

        let mut changes = Vec::new();
        let snapshot: Vec<(String, AssetHash)> = self
            .watched
            .iter()
            .map(|(path, hash)| (path.clone(), *hash))
            .collect();

        for (path, before) in snapshot {
            let after = hash_path(vfs, &path).ok();
            if after == Some(before) {
                continue;
            }
            match after {
                Some(hash) => {
                    self.watched.insert(path.clone(), hash);
                }
                // XXX: a removed path stops being watched, so a file that
                // reappears is not reported. Re-adding it is the caller's call,
                // not something a poll can decide.
                None => {
                    self.watched.remove(&path);
                }
            }
            changes.push((path, Change { before, after }));
        }

        changes
    }
}

/// Resolves and hashes a virtual path, naming it in the error so a caller knows
/// which watch failed without the underlying resolution error being lost.
fn hash_path(vfs: &Vfs, path: &str) -> Result<AssetHash> {
    let physical = vfs
        .resolve(path)
        .map_err(|_| Error::Watch(path.to_owned()))?;
    let bytes = fs::read(&physical).map_err(|_| Error::Watch(path.to_owned()))?;
    Ok(AssetHash::of(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("vault-watch-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("scratch directory");
        root
    }

    #[test]
    fn the_interval_gates_a_second_poll_but_poll_now_does_not() {
        let root = scratch("gate");
        let file = root.join("a.txt");
        fs::write(&file, "one").expect("write");

        let mut vfs = Vfs::new();
        vfs.mount(&root);
        let mut watcher = Watcher::new(Duration::from_secs(60));
        watcher.watch(&vfs, "a.txt").expect("watch");

        fs::write(&file, "two").expect("rewrite");
        let first = watcher.poll(&vfs);
        assert_eq!(first.len(), 1, "the first poll is due and sees the change");

        fs::write(&file, "three").expect("rewrite again");
        assert!(
            watcher.poll(&vfs).is_empty(),
            "the second poll is inside the interval and must not look"
        );
        let forced = watcher.poll_now(&vfs);
        assert_eq!(forced.len(), 1, "poll_now ignores the interval");
        assert_ne!(
            forced[0].1.before,
            forced[0].1.after.expect("still present")
        );

        let _ = fs::remove_dir_all(&root);
    }
}
