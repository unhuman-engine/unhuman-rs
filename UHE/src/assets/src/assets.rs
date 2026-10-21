use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use grind::{CancelToken, Jobs, Priority};

use crate::cache::{Cache, DEFAULT_CAPACITY};
use crate::cache_hash::AssetHash;
use crate::decode::{
    classify, decode_bytes, decode_key, read_and_decode, Decoded, Finished, LoadResult,
};
use crate::error::Result;
use crate::id::{AssetData, AssetId, AssetKind, AssetState, Handle};
use crate::image::{Image, MipLevel};
use crate::mesh::{MeshData, ModelData};
use crate::report::LoadReport;
use crate::vfs::Vfs;

#[derive(Clone, Copy, PartialEq, Eq)]
enum LoadState {
    Loading,
    Ready,
}

/// One asset: what it is, where it came from, and what decoding it produced.
struct Slot {
    /// Bumped on every release, so an id from before the release stops
    /// resolving even when the slot is handed out again.
    generation: u32,
    path: String,
    kind: AssetKind,
    hash: AssetHash,
    state: LoadState,
    /// The decoded payload. Shared with the cache, so two assets with identical
    /// bytes point at one decode without the registry duplicating it.
    decoded: Option<Arc<Decoded>>,
    /// Set while an asynchronous decode is in flight, so `release` can tell the
    /// worker to drop a load whose reason is gone.
    cancel: Option<CancelToken>,
}

impl Slot {
    fn new() -> Self {
        Self {
            // Generations start at 1, so a zeroed or default id is never valid.
            generation: 1,
            path: String::new(),
            kind: AssetKind::Binary,
            hash: AssetHash::of(&[]),
            state: LoadState::Loading,
            decoded: None,
            cancel: None,
        }
    }

    fn data(&self) -> Option<&Decoded> {
        self.decoded.as_deref()
    }
}

/// The asset registry.
///
/// Deliberately simpler than a refcounted manager: no `shared_ptr` per asset
/// (the registry owns every payload and a handle is an index), no `finalized`
/// flag (there is no GPU upload step in this crate, so "decoded" is the whole
/// of "ready"), and no mutex (a registry is reached through `&mut Assets`,
/// which is the one owner the spine asks for).
///
/// A path-keyed registry exists so a call does not re-walk a `path -> ID` map
/// and every handle does not re-check a generation counter on a shared record.
pub struct Assets {
    vfs: Vfs,
    slots: Vec<Slot>,
    /// Decoded payloads keyed by content address, so a release can drop a slot
    /// without dropping the decode and a reload or a second path with the same
    /// bytes is a hit.
    cache: Cache<Arc<Decoded>>,
    /// Indices of released slots, for reuse.
    free: Vec<u32>,
    /// Virtual path -> the live id for it, so a second request shares the first
    /// decode.
    paths: HashMap<String, AssetId>,
    sender: mpsc::Sender<Finished>,
    receiver: mpsc::Receiver<Finished>,
    /// Loads that were already finished when they were asked for, reported by
    /// the next `poll` like any other so a caller waiting on it is not left
    /// waiting forever.
    ready: Vec<(AssetId, LoadResult)>,
    live: usize,
}

impl Assets {
    /// A registry with `root` as its first (and initially only) search root.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::with_cache_capacity(root, DEFAULT_CAPACITY)
    }

    /// A registry whose payload cache holds at most `capacity` decodes. A small
    /// capacity is how a caller bounds memory, and how eviction is exercised
    /// without loading hundreds of assets.
    pub fn with_cache_capacity(root: impl Into<PathBuf>, capacity: usize) -> Self {
        Self::with_cache_limits(root, capacity, None)
    }

    /// A registry bounded by both entry count and, when `byte_capacity` is
    /// positive, the estimated heap of the decoded payloads.
    pub fn with_cache_limits(
        root: impl Into<PathBuf>,
        capacity: usize,
        byte_capacity: Option<usize>,
    ) -> Self {
        let mut vfs = Vfs::new();
        vfs.mount(root);
        let (sender, receiver) = mpsc::channel();
        Self {
            vfs,
            slots: Vec::new(),
            cache: Cache::with_byte_capacity(capacity, byte_capacity),
            free: Vec::new(),
            paths: HashMap::new(),
            sender,
            receiver,
            ready: Vec::new(),
            live: 0,
        }
    }

    /// Appends a search root. Earlier roots win, so a mod directory mounted
    /// ahead of the game directory overrides it.
    pub fn mount(&mut self, dir: impl Into<PathBuf>) {
        self.vfs.mount(dir);
    }

    pub fn roots(&self) -> &[PathBuf] {
        self.vfs.roots()
    }

    /// The virtual file system this registry resolves against.
    pub fn vfs(&self) -> &Vfs {
        &self.vfs
    }

    /// Resolves, reads and decodes one asset, returning a ready id.
    ///
    /// A second load of the same path returns the first id rather than decoding
    /// twice; an id whose slot a `load_async` already owns is returned in its
    /// current (possibly not yet decoded) state.
    pub fn load(&mut self, path: &str) -> Result<AssetId> {
        if let Some(id) = self.live_id(path) {
            return Ok(id);
        }
        let physical = self.vfs.resolve(path)?;
        let bytes: Arc<[u8]> = std::fs::read(&physical)?.into();
        let hash = AssetHash::of(&bytes);
        // Identical bytes already decoded the same way are shared, so a copy of
        // a texture is a hit rather than a second decode.
        let key = decode_key(hash, classify(path), &physical, Arc::clone(&bytes));
        if let Some(decoded) = self.cache.get(&key).cloned() {
            return Ok(self.store(path, decoded));
        }
        let decoded = decode_bytes(&physical, path, bytes, hash)?;
        let decoded = self.intern(decoded);
        Ok(self.store(path, decoded))
    }

    /// Releases the live id for `path`, then decodes the file again.
    ///
    /// The new id differs from the old one even though the slot index is reused,
    /// because `release` bumps the generation: a caller still holding the old id
    /// stops resolving it rather than reading new bytes under the old name.
    pub fn reload(&mut self, path: &str) -> Result<AssetId> {
        if let Some(id) = self.find(path) {
            self.release(id);
        }
        self.load(path)
    }

    /// The asynchronous counterpart of [`Assets::reload`].
    pub fn reload_async(&mut self, jobs: &Jobs, path: &str) -> Result<AssetId> {
        if let Some(id) = self.find(path) {
            self.release(id);
        }
        self.load_async(jobs, path)
    }

    /// Submits a load to the job system and returns its id immediately.
    ///
    /// The worker reads the file and decodes it (CPU work only) and hands the
    /// result back through a channel that [`Assets::poll`] drains. Until then
    /// the id is [`AssetState::Pending`]: a consumer draws without the asset and
    /// it appears once `poll` reports it. The decode runs at
    /// [`Priority::Low`], because a frame that is recording is waiting and a
    /// decode can be a frame late. A load that fails is reported once, and its
    /// slot is released, so nothing is re-delivered forever.
    pub fn load_async(&mut self, jobs: &Jobs, path: &str) -> Result<AssetId> {
        if let Some(id) = self.live_id(path) {
            let kind = self.slot(id).map(|slot| slot.kind);
            if let Some(kind) = kind {
                self.ready.push((id, Ok(kind)));
            }
            return Ok(id);
        }

        let physical = self.vfs.resolve(path)?;
        let id = self.allocate(path, classify(path));
        let cancel = CancelToken::new();
        if let Some(slot) = self.slots.get_mut(id.index() as usize) {
            slot.cancel = Some(cancel.clone());
        }
        let sender = self.sender.clone();
        let virtual_path = path.to_owned();
        jobs.execute_cancellable(
            Box::new(move || {
                let decoded = read_and_decode(&physical, &virtual_path);
                // A closed receiver means the registry is gone; the result has
                // nowhere to go and nothing to clean up.
                let _ = sender.send(Finished { id, decoded });
            }),
            None,
            Priority::Low,
            cancel,
        );
        Ok(id)
    }

    /// Every load that has finished since the last call, in completion order.
    pub fn poll(&mut self) -> Vec<(AssetId, LoadResult)> {
        let mut out: Vec<(AssetId, LoadResult)> = std::mem::take(&mut self.ready);

        while let Ok(finished) = self.receiver.try_recv() {
            let id = finished.id;
            // Looked up without the `Ready` requirement the accessors apply: a
            // loading slot is exactly what a finished job exists to resolve.
            let Some(slot) = self.slots.get(id.index() as usize) else {
                continue;
            };
            if slot.generation != id.generation() {
                // Released since the job was submitted.
                continue;
            }
            if slot.state != LoadState::Loading {
                // A synchronous load already resolved this slot; the worker's
                // answer is stale.
                continue;
            }

            match finished.decoded {
                Ok(decoded) => {
                    let kind = decoded.kind;
                    let decoded = self.intern(decoded);
                    self.store_at(id, decoded);
                    out.push((id, Ok(kind)));
                }
                Err(error) => {
                    self.release(id);
                    out.push((id, Err(error)));
                }
            }
        }

        out
    }

    /// The encoded file bytes, for every kind. Valid UTF-8 for a text asset is
    /// not a promise this makes; [`Assets::text`] is the accessor for that.
    pub fn bytes(&self, id: AssetId) -> Option<&[u8]> {
        self.slot(id)?.data().map(|data| &data.bytes[..])
    }

    /// The decoded image, which is level 0 of a KTX2 chain.
    pub fn image(&self, id: AssetId) -> Option<&Image> {
        self.slot(id)?.data()?.image.as_ref()
    }

    /// The mip levels the file shipped, empty when it shipped none or the
    /// asset is not an image.
    pub fn mips(&self, id: AssetId) -> Option<&[MipLevel]> {
        self.slot(id)
            .and_then(|slot| slot.data())
            .map(|data| data.mips.as_slice())
    }

    /// The first mesh of a model, which is also the one carrying the model's
    /// animation clips and skeleton.
    pub fn mesh(&self, id: AssetId) -> Option<&MeshData> {
        self.slot(id)?.data()?.meshes.first()
    }

    /// Every mesh primitive of a model, in file order.
    pub fn meshes(&self, id: AssetId) -> Option<&[MeshData]> {
        self.slot(id)
            .and_then(|slot| slot.data())
            .map(|data| data.meshes.as_slice())
    }

    /// A model's node graph, skins and unimplemented-extension list.
    pub fn model(&self, id: AssetId) -> Option<&ModelData> {
        self.slot(id)?.data()?.model.as_ref()
    }

    /// What the load read, skipped and substituted. Reachable from the decoded
    /// payload, so inspecting it does not re-parse the file.
    pub fn report(&self, id: AssetId) -> Option<&LoadReport> {
        let decoded = self.slot(id)?.data()?;
        decoded
            .model
            .as_ref()
            .map(|model| &model.report)
            .or(decoded.report.as_ref())
    }

    /// The asset as text, or `None` when it is not a text asset or not UTF-8.
    pub fn text(&self, id: AssetId) -> Option<&str> {
        let slot = self.slot(id)?;
        if slot.kind != AssetKind::Text {
            return None;
        }
        std::str::from_utf8(&slot.data()?.bytes).ok()
    }

    pub fn kind(&self, id: AssetId) -> Option<AssetKind> {
        self.slot(id).map(|slot| slot.kind)
    }

    /// The content address of the bytes this asset was decoded from. Same bytes
    /// always give the same value; a changed file gives a different one and so a
    /// different id.
    pub fn hash(&self, id: AssetId) -> Option<AssetHash> {
        self.slot(id).map(|slot| slot.hash)
    }

    /// Decoded payloads currently held by the cache, which is at most its
    /// capacity. A release drops a slot but not its cache entry.
    pub fn cached(&self) -> usize {
        self.cache.len()
    }

    pub fn cache_capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Estimated bytes charged by the cache, and its optional byte limit.
    pub fn cache_bytes(&self) -> usize {
        self.cache.bytes()
    }

    pub fn cache_byte_capacity(&self) -> Option<usize> {
        self.cache.byte_capacity()
    }

    /// The virtual path this asset was loaded from.
    pub fn path(&self, id: AssetId) -> Option<&str> {
        self.slot(id).map(|slot| slot.path.as_str())
    }

    /// Whether the id names a pending or a resolved asset.
    ///
    /// `None` for an id that names nothing at all, which is the same answer a
    /// released asset gives. A `Pending` id is valid: it will resolve (or the
    /// load will fail and be reported by `poll`), so a consumer holds it and
    /// draws without the asset until then.
    pub fn state(&self, id: AssetId) -> Option<AssetState> {
        let slot = self.any_slot(id)?;
        Some(match slot.state {
            LoadState::Loading => AssetState::Pending,
            LoadState::Ready => AssetState::Ready,
        })
    }

    /// A typed handle, or `None` when the id names nothing or names a different
    /// payload kind than `T`. A pending id still yields a handle: the payload
    /// resolves later, and `get` returns `None` until it does.
    pub fn handle<T: ?Sized + AssetData>(&self, id: AssetId) -> Option<Handle<T>> {
        let slot = self.any_slot(id)?;
        T::accepts(slot.kind).then(|| Handle::new(id))
    }

    /// The live id for a virtual path, which is the registry's `name -> handle`
    /// half. `None` for a path that was never loaded or has been released.
    pub fn find(&self, path: &str) -> Option<AssetId> {
        let id = self.paths.get(path).copied()?;
        self.slot(id).map(|_| id)
    }

    /// Frees a slot: the payload is dropped, the path mapping is removed, and
    /// the generation is bumped so every outstanding id for this asset stops
    /// resolving instead of aliasing whatever takes the slot next.
    ///
    /// Returns whether the id was live. A stale id is refused, not ignored.
    pub fn release(&mut self, id: AssetId) -> bool {
        if id.is_null() {
            return false;
        }
        let Some(slot) = self.slots.get_mut(id.index() as usize) else {
            return false;
        };
        if slot.generation != id.generation() {
            return false;
        }

        if self
            .paths
            .get(slot.path.as_str())
            .is_some_and(|current| *current == id)
        {
            let path = slot.path.clone();
            self.paths.remove(&path);
        }

        // A pending decode whose reason is gone is told to stop before it
        // starts; one already running is not interrupted, but its result is
        // discarded because the generation below no longer matches.
        if let Some(cancel) = slot.cancel.take() {
            cancel.cancel();
        }

        // Wraps after 2^32 releases of one slot, which no session reaches.
        slot.generation = slot.generation.wrapping_add(1);
        slot.state = LoadState::Loading;
        slot.path.clear();
        slot.kind = AssetKind::Binary;
        slot.hash = AssetHash::of(&[]);
        // The cache still holds the decode until it is evicted, so a reload of
        // the same content is a hit even though this slot no longer points at
        // it.
        slot.decoded = None;

        self.free.push(id.index());
        self.live -= 1;
        true
    }

    /// Whether the id names a decoded asset right now.
    pub fn contains(&self, id: AssetId) -> bool {
        self.slot(id).is_some()
    }

    /// Live assets, not slot count: released slots are kept for reuse.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// The slot if the id's generation matches, regardless of whether the
    /// decode has finished: a loading slot is a valid handle target.
    fn any_slot(&self, id: AssetId) -> Option<&Slot> {
        if id.is_null() {
            return None;
        }
        let slot = self.slots.get(id.index() as usize)?;
        (slot.generation == id.generation()).then_some(slot)
    }

    fn slot(&self, id: AssetId) -> Option<&Slot> {
        self.any_slot(id)
            .filter(|slot| slot.state == LoadState::Ready)
    }

    fn live_id(&self, path: &str) -> Option<AssetId> {
        let id = self.paths.get(path).copied()?;
        self.slots
            .get(id.index() as usize)
            .filter(|slot| slot.generation == id.generation())
            .map(|_| id)
    }

    fn allocate(&mut self, path: &str, kind: AssetKind) -> AssetId {
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                self.slots.push(Slot::new());
                (self.slots.len() - 1) as u32
            }
        };

        // The free list only ever holds indices this table created, so the
        // index is in range.
        let slot = &mut self.slots[index as usize];
        let id = AssetId::from_parts(index, slot.generation);
        slot.path = path.to_owned();
        slot.kind = kind;
        slot.state = LoadState::Loading;
        slot.cancel = None;

        self.paths.insert(path.to_owned(), id);
        self.live += 1;
        id
    }

    /// Files a freshly decoded payload in the cache and returns the shared
    /// handle, reusing an existing entry with the same content rather than
    /// keeping a second copy of the decode.
    fn intern(&mut self, decoded: Decoded) -> Arc<Decoded> {
        let key = decoded.key.clone();
        if let Some(cached) = self.cache.get(&key).cloned() {
            return cached;
        }
        let size = decoded.size_bytes();
        let decoded = Arc::new(decoded);
        let _ = self.cache.insert_sized(key, size, Arc::clone(&decoded));
        decoded
    }

    fn store(&mut self, path: &str, decoded: Arc<Decoded>) -> AssetId {
        match self.live_id(path) {
            Some(id) => {
                self.store_at(id, decoded);
                id
            }
            None => {
                let id = self.allocate(path, decoded.kind);
                self.store_at(id, decoded);
                id
            }
        }
    }

    fn store_at(&mut self, id: AssetId, decoded: Arc<Decoded>) {
        let Some(slot) = self.slots.get_mut(id.index() as usize) else {
            return;
        };
        if slot.generation != id.generation() {
            return;
        }

        slot.kind = decoded.kind;
        slot.hash = decoded.hash;
        slot.state = LoadState::Ready;
        slot.decoded = Some(decoded);
        // The decode is done; there is nothing left to cancel.
        slot.cancel = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn a_missing_path_is_an_error_and_allocates_nothing() {
        let mut assets = Assets::new(".");
        assert!(matches!(
            assets.load("nothing/here.png"),
            Err(Error::NotFound(_))
        ));
        assert!(assets.is_empty());
    }
}
