//! Asset registry, virtual file system, loaders.
//!
//! Nothing here names a GPU type. `chassis` is not a dependency of this crate,
//! so the headless binary links the whole loader set and the editor can decode
//! a texture without a device.
//!
//! # The questions the design answers
//!
//! **Identity.** The *virtual path* is the lookup: it is what a caller has and
//! what [`Assets::find`] resolves. The *content hash* ([`AssetHash`]) is the
//! cache key and the reload's change detector: same bytes, same decode. The
//! [`AssetId`] is an internal slot index plus a generation, stable for the life
//! of one load and refused once released. Three names, three jobs: the path
//! names, the hash compares, the id refers.
//!
//! **Lifetime.** The registry owns every payload; a handle does not keep an
//! asset alive. A caller that is done with one calls [`Assets::release`], which
//! bumps the slot's generation so a stale id resolves to nothing instead of
//! aliasing whatever takes the slot next. There is no frame-numbered release
//! and no session-long pin: the cache, not the slot, is what a reclaimable
//! decode stays in.
//!
//! **Residency.** The decode cache ([`Cache`]) is bounded and evicts
//! least-recently-used. A released slot drops its reference but the decode
//! stays cacheable until eviction, so a reload or a second path with the same
//! bytes is a hit. VRAM residency is not this crate's: it hands the renderer
//! CPU-side pixels, and the renderer owns eviction on the device.
//!
//! **Failure.** A missing or undecodable file is an [`Error`] from the load,
//! never a null handle: a substituted 1x1 belongs at the consumer that chose to
//! substitute, not here. A model records what it ignored
//! ([`ModelData::unsupported_extensions`]) and every load keeps a
//! [`LoadReport`] of what it read, skipped and substituted, reachable through
//! [`Assets::report`] without a second parse. [`Assets::poll`] reports a failed
//! asynchronous load exactly once.
//!
//! **Hot reload.** The watcher detects a changed file; [`Reloader`] releases
//! the old id and loads the new bytes. A reload runs off-thread by default, so
//! a frame holding the old handle keeps drawing the old asset until the new one
//! is polled and the caller re-reads the path. The old handle is not mutated
//! under the caller: it stops resolving, and the new asset has a new id.
//!
//! **Threading.** `vault` may load on a worker ([`Assets::load_async`]): read
//! and decode are CPU-only. Upload is not: the device paths are not
//! thread-safe, so the GPU handoff happens on the main thread, from the pixels
//! `poll` made ready. A load that blocks the frame is the one thing this shape
//! exists to avoid.
//!
//! # Drawing without an asset
//!
//! A load submitted with [`Assets::load_async`] returns an id that is
//! [`AssetState::Pending`] until [`Assets::poll`] reports it. The contract for
//! every consumer is: draw without the asset this frame, and it appears when it
//! resolves. The editor draws the node and highlights nothing; a scene mesh
//! skips its draw rather than stalling; a material falls back to its factors
//! with no map bound. A [`Handle`] for a pending id is valid, its `get` returns
//! `None`, and its [`Handle::state`] says `Pending`.

mod anim;
mod assets;
mod cache;
mod cache_hash;
mod decode;
mod error;
mod extensions;
mod id;
mod image;
mod json;
mod ktx2;
mod load_lights;
mod load_material;
mod load_model;
mod load_model_animation;
mod load_model_geometry;
mod load_texture;
mod material_extensions;
mod mesh;
mod reload;
mod report;
mod uv;
mod vfs;
mod watch;

pub use anim::{compress, measure, CompressSettings, CompressionStats};
pub use assets::Assets;
pub use cache::{Cache, DEFAULT_CAPACITY};
pub use cache_hash::{AssetHash, ContentKey};
pub use decode::LoadResult;
pub use error::{Error, Result};
pub use id::{AssetData, AssetId, AssetKind, AssetState, Handle};
pub use image::{
    decode_image, downsample_rgba8, generate_mipmap_chain_rgba8, rgba8_size, Image, MipLevel,
};
pub use ktx2::{is_ktx2, load_ktx2, Ktx2Image};
pub use mesh::{
    texture_slot, AlphaMode, AnimationData, BlendApproach, BoneData, Filter, Interpolation,
    LightType, MaterialData, MaterialExtensions, MeshData, ModelData, ModelNode, PunctualLight,
    QuaternionTrack, SamplerData, SkeletonData, SkinData, TextureData, UVTransform, VectorTrack,
    Wrap, MATERIAL_TEXTURE_SLOT_COUNT,
};
pub use reload::{Reloaded, Reloader};
pub use report::{ExtensionClass, ExtensionUse, LoadCounts, LoadReport, Substitution};
pub use vfs::Vfs;
pub use watch::{Change, Watcher, DEFAULT_INTERVAL};
