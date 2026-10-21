use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::assets::Assets;
use crate::image::Image;
use crate::mesh::{MeshData, ModelData};

/// A stable reference to a registered asset: slot index plus the generation the
/// slot was handed out at.
///
/// A UUID used as a map key would say nothing about which slot it meant, so a
/// stale id and a live id would be indistinguishable and a freed asset's id
/// could be reused with no way to tell. Here the generation is compared on
/// every dereference, so an id from before a release resolves to nothing
/// instead of aliasing the asset that took its slot.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetId {
    index: u32,
    generation: u32,
}

impl AssetId {
    /// The id no asset has. `Assets::release` refuses it and every accessor
    /// returns `None`, which is what a default-constructed handle should do in
    /// a struct that is `Default` without owning anything.
    pub const NULL: AssetId = AssetId {
        index: u32::MAX,
        generation: 0,
    };

    /// Packs a slot index and a generation into an id.
    ///
    /// Public because a table keyed by `AssetId` (the audio backend's sound
    /// table is one) has to be tested without an `Assets` to create ids from.
    /// A fabricated id resolves to nothing unless it happens to match a live
    /// slot's generation, which is the same answer a released asset gives.
    pub const fn from_parts(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn index(self) -> u32 {
        self.index
    }

    pub fn generation(self) -> u32 {
        self.generation
    }

    pub fn is_null(self) -> bool {
        self.index == u32::MAX
    }
}

impl Default for AssetId {
    fn default() -> Self {
        AssetId::NULL
    }
}

impl fmt::Debug for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "AssetId(null)")
        } else {
            write!(f, "AssetId({}#{})", self.index, self.generation)
        }
    }
}

/// What a registered asset holds, which decides the accessor that resolves it.
///
/// The kinds are the ones a caller can actually ask for; there is no `None` (a
/// slot with no kind does not exist) and no separate kind per image encoding,
/// because the crate decodes to one [`Image`] either way.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AssetKind {
    /// UTF-8 text: `.json`, `.slang`, `.unhuman`, ... Resolved by [`Assets::text`].
    Text,
    /// Opaque bytes with no decoder. Resolved by [`Assets::bytes`].
    Binary,
    /// A decoded RGBA8 image, from any encoding. Resolved by [`Assets::image`].
    Image,
    /// A glTF/GLB model: geometry, materials, skins, animation. Resolved by
    /// [`Assets::mesh`] / [`Assets::meshes`] / [`Assets::model`].
    Model,
}

/// Where a handle stands: not decoded yet, or holding a payload.
///
/// A load submitted to a worker is [`AssetState::Pending`] until the registry
/// polls its result, which is what lets a frame draw without the asset instead
/// of blocking on the decode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssetState {
    Pending,
    Ready,
}

/// A typed, generation-checked reference to an asset.
///
/// `T` is the payload, so a `Handle<Image>` cannot be passed where a
/// `Handle<MeshData>` is expected and `get` cannot return the wrong payload.
/// The handle does not keep the asset alive: a refcount would, at the cost of a
/// slot that could never be reclaimed while any stale handle existed. Here the
/// registry owns every payload and a released asset is gone for every handle at
/// once.
#[derive(Debug)]
pub struct Handle<T: ?Sized> {
    id: AssetId,
    // `PhantomData<T>` and not `fn() -> T`: `T` may be `str` or `[u8]`, which a
    // function returning it cannot express.
    _tag: PhantomData<T>,
}

impl<T: ?Sized> Handle<T> {
    pub(crate) fn new(id: AssetId) -> Self {
        Self {
            id,
            _tag: PhantomData,
        }
    }

    pub fn id(&self) -> AssetId {
        self.id
    }

    /// Whether the asset is still [`AssetState::Pending`] or has resolved.
    /// `None` when it was released or the slot was reused.
    pub fn state(&self, assets: &Assets) -> Option<AssetState> {
        assets.state(self.id)
    }

    /// The payload, or `None` when the decode has not finished, the asset was
    /// released, or the slot was reused since this handle was made.
    pub fn get<'a>(&self, assets: &'a Assets) -> Option<&'a T>
    where
        T: AssetData,
    {
        T::lookup(assets, self.id)
    }
}

impl<T: ?Sized> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Handle<T> {}

impl<T: ?Sized> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: ?Sized> Eq for Handle<T> {}

impl<T: ?Sized> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// The payload a [`Handle`] names.
///
/// Sealed in effect: the lookup is implemented in terms of the public
/// accessors, so a type outside this crate cannot claim a kind it has no
/// accessor for.
pub trait AssetData: 'static {
    const KIND: AssetKind;

    /// Whether a slot of `kind` resolves through this payload. Every payload
    /// matches its own [`KIND`](Self::KIND) except the raw bytes, which every
    /// asset holds and which therefore accept any kind.
    fn accepts(kind: AssetKind) -> bool {
        kind == Self::KIND
    }

    fn lookup(assets: &Assets, id: AssetId) -> Option<&Self>;
}

impl AssetData for Image {
    const KIND: AssetKind = AssetKind::Image;

    fn lookup(assets: &Assets, id: AssetId) -> Option<&Self> {
        assets.image(id)
    }
}

impl AssetData for MeshData {
    const KIND: AssetKind = AssetKind::Model;

    fn lookup(assets: &Assets, id: AssetId) -> Option<&Self> {
        assets.mesh(id)
    }
}

impl AssetData for ModelData {
    const KIND: AssetKind = AssetKind::Model;

    fn lookup(assets: &Assets, id: AssetId) -> Option<&Self> {
        assets.model(id)
    }
}

impl AssetData for str {
    const KIND: AssetKind = AssetKind::Text;

    fn lookup(assets: &Assets, id: AssetId) -> Option<&Self> {
        assets.text(id)
    }
}

impl AssetData for [u8] {
    const KIND: AssetKind = AssetKind::Binary;

    fn accepts(_kind: AssetKind) -> bool {
        true
    }

    fn lookup(assets: &Assets, id: AssetId) -> Option<&Self> {
        assets.bytes(id)
    }
}
