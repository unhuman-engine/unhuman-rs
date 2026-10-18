//! Asset registry, virtual file system, loaders.
//!
//! A path-based registry keyed by payload type, with a handle that held a
//! `shared_ptr` and compared a generation counter on every dereference, is
//! replaced here: the registry is the only way to reach a payload. Every asset
//! is an [`AssetId`] (slot index + generation), a caller asks for the id once
//! with [`Assets::load`], and a stale id is refused rather than silently
//! aliasing whatever took the slot next.
//!
//! Nothing here names a GPU type. `chassis` is not a dependency of this crate,
//! so the headless binary links the whole loader set and the editor can decode
//! a texture without a device.

mod anim;
mod assets;
mod cache;
mod cache_hash;
mod error;
mod id;
mod image;
mod ktx2;
mod load_lights;
mod load_material;
mod load_model;
mod load_model_animation;
mod load_model_geometry;
mod load_texture;
mod mesh;
mod reload;
mod vfs;
mod watch;

pub use anim::{compress, measure, CompressSettings, CompressionStats};
pub use assets::{Assets, LoadResult};
pub use cache::{Cache, DEFAULT_CAPACITY};
pub use cache_hash::AssetHash;
pub use error::{Error, Result};
pub use id::{AssetData, AssetId, AssetKind, Handle};
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
pub use vfs::Vfs;
pub use watch::{Change, Watcher, DEFAULT_INTERVAL};
