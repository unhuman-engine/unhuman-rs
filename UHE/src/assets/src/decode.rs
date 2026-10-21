//! The one place a file becomes an asset payload.
//!
//! The synchronous path and the worker path both call [`decode_bytes`], so they
//! cannot diverge on what a slot holds. The extension decides the kind, and a
//! file that does not match its extension is an error rather than a silent
//! binary asset: the caller asked for it as the other kind.

use std::path::Path;
use std::sync::Arc;

use crate::cache_hash::{AssetHash, ContentKey};
use crate::error::{Error, Result};
use crate::id::{AssetId, AssetKind};
use crate::image::{Image, MipLevel};
use crate::load_model;
use crate::load_texture::decode_texture;
use crate::mesh::{MeshData, ModelData};
use crate::report::LoadReport;

/// What a finished load reports: the kind that landed, or why it did not.
///
/// A plain `Result` rather than a bespoke enum, because that is how a caller
/// wants to read it: `for (id, result) in assets.poll() { if let Err(error) =
/// result { ... } }`.
pub type LoadResult = std::result::Result<AssetKind, Error>;

/// Everything one load produced.
///
/// One type for the synchronous and the worker path, so both reach the registry
/// through the same code and cannot diverge on what a slot holds.
pub(crate) struct Decoded {
    /// The content address of `bytes`: the asset's stable identity.
    pub(crate) hash: AssetHash,
    /// The address of everything this decode depended on. The hash is the
    /// bucket; the full source bytes and context make a collision a miss.
    pub(crate) key: ContentKey,
    pub(crate) kind: AssetKind,
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) image: Option<Image>,
    pub(crate) mips: Vec<MipLevel>,
    pub(crate) meshes: Vec<MeshData>,
    pub(crate) model: Option<ModelData>,
    /// What this load read, skipped and substituted. A model carries the same
    /// report on [`ModelData`]; other kinds keep it here.
    pub(crate) report: Option<LoadReport>,
}

impl Decoded {
    /// An estimate of the heap this decode holds, for a byte-bounded cache.
    pub(crate) fn size_bytes(&self) -> usize {
        let mut size = self.bytes.len()
            + self.mips.iter().map(|mip| mip.rgba8.len()).sum::<usize>()
            + self.image.as_ref().map_or(0, |image| image.rgba8.len());
        for mesh in &self.meshes {
            size += mesh.positions.len() * size_of::<[f32; 3]>()
                + mesh.normals.len() * size_of::<[f32; 3]>()
                + mesh.uvs.len() * size_of::<[f32; 2]>()
                + mesh.tangents.len() * size_of::<[f32; 4]>()
                + mesh.colors.len() * size_of::<[f32; 4]>()
                + mesh.weights.len() * size_of::<[f32; 4]>()
                + mesh.joints.len() * size_of::<[u16; 4]>()
                + mesh.indices.len() * size_of::<u32>();
        }
        size
    }
}

/// A finished asynchronous load, sent from a worker thread.
pub(crate) struct Finished {
    pub(crate) id: AssetId,
    pub(crate) decoded: Result<Decoded>,
}

/// The kind a virtual path's extension names.
///
/// Choosing a loader from the payload type the caller asked for cannot be right
/// for a registry keyed by path, because the path is the only thing a caller
/// has. Unknown extensions are [`AssetKind::Binary`] rather than an error,
/// because a `.bin` nobody can decode is still worth having the bytes of.
pub(crate) fn classify(path: &str) -> AssetKind {
    let extension = path
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        "png" | "jpg" | "jpeg" | "bmp" | "tga" | "ktx2" => AssetKind::Image,
        "gltf" | "glb" => AssetKind::Model,
        "txt" | "json" | "md" | "toml" | "xml" | "csv" | "yaml" | "yml" | "unhuman" | "slang"
        | "wgsl" | "glsl" | "vert" | "frag" | "comp" => AssetKind::Text,
        _ => AssetKind::Binary,
    }
}

/// The cache key for a decode: a content address over everything the decode
/// reads, not just the bytes.
///
/// The loader-kind is folded in because an extension decides how the same bytes
/// are decoded (text versus binary), so keying on bytes alone would hand a text
/// asset's payload to a `.bin` of the same content. A model folds in its
/// directory because its external buffers and images resolve relative to the
/// file, so identical `.gltf` bytes in two trees can decode differently.
pub(crate) fn decode_key(
    hash: AssetHash,
    kind: AssetKind,
    physical: &Path,
    content: Arc<[u8]>,
) -> ContentKey {
    let mut context = vec![kind as u8];
    if kind == AssetKind::Model {
        context.extend_from_slice(physical.as_os_str().as_encoded_bytes());
    }
    let mut bucket = hash.to_bytes().to_vec();
    bucket.extend_from_slice(&context);
    ContentKey::new(AssetHash::of(&bucket), content, context)
}

/// Reads one file and decodes it, the job-system entry point.
///
/// The synchronous path reads the bytes first so it can look them up in the
/// cache; this is the same work for the worker, which has no cache of its own.
pub(crate) fn read_and_decode(physical: &Path, virtual_path: &str) -> Result<Decoded> {
    let bytes: Arc<[u8]> = std::fs::read(physical)?.into();
    let hash = AssetHash::of(&bytes);
    decode_bytes(physical, virtual_path, bytes, hash)
}

/// Decodes bytes that were already read and hashed.
///
/// The single place a load happens, shared by the synchronous path and the job
/// system. A file that is read but does not match its extension is an error,
/// not a silent binary asset: the caller asked for it as the other kind.
pub(crate) fn decode_bytes(
    physical: &Path,
    virtual_path: &str,
    bytes: Arc<[u8]>,
    hash: AssetHash,
) -> Result<Decoded> {
    let kind = classify(virtual_path);
    let key = decode_key(hash, kind, physical, Arc::clone(&bytes));
    let mut report = LoadReport::new(virtual_path, format_of(virtual_path));
    let mut decoded = Decoded {
        hash,
        key,
        kind,
        bytes,
        image: None,
        mips: Vec::new(),
        meshes: Vec::new(),
        model: None,
        report: None,
    };

    match kind {
        AssetKind::Image => {
            // A standalone image is colour data unless a caller says otherwise;
            // the linear maps only exist inside a material, where the slot
            // decides. One decoder handles KTX2 and PNG/JPEG alike, so the two
            // cannot diverge.
            let (image, mips) = decode_texture(&decoded.bytes, true)?;
            decoded.image = Some(image);
            decoded.mips = mips;
            report.loaded.textures = 1;
        }
        AssetKind::Model => {
            let loaded = load_model::load(physical, &decoded.bytes, &mut report)?;
            decoded.meshes = loaded.meshes;
            decoded.model = Some(loaded.model);
        }
        AssetKind::Text | AssetKind::Binary => {}
    }

    report.log();
    // A model already carries the report on `ModelData`; storing a second copy
    // here would make the two diverge on the next edit.
    if decoded.model.is_none() {
        decoded.report = Some(report);
    }
    Ok(decoded)
}

fn format_of(path: &str) -> String {
    path.rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_by_extension_and_case_insensitive() {
        assert_eq!(classify("textures/wall.PNG"), AssetKind::Image);
        assert_eq!(classify("models/scene.glb"), AssetKind::Model);
        assert_eq!(classify("shaders/pbr.wgsl"), AssetKind::Text);
        assert_eq!(classify("data/blob"), AssetKind::Binary);
        assert_eq!(classify("data/thing.unknownext"), AssetKind::Binary);
    }
}
