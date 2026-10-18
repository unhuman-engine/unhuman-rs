//! Texture decoding for a glTF material slot.
//!
//! One cache for the whole asset: a glTF image is routinely bound by several
//! materials and slots, and without the cache the identical pixels are decoded
//! and resident once per reference. The key includes the colour space and the
//! sampler, because the same image legitimately becomes both an sRGB and a
//! linear texture, with different wrap state.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use gltf::buffer::Data as BufferData;

use crate::image::{decode_image, Image, MipLevel};
use crate::ktx2::{is_ktx2, load_ktx2};
use crate::mesh::{Filter, SamplerData, TextureData, Wrap};

/// Decoded material textures, keyed by (image index, colour space, sampler).
pub(crate) type TextureCache = HashMap<(usize, bool, SamplerData), Option<Arc<TextureData>>>;

/// Decodes one glTF texture, honouring `KHR_texture_basisu` first.
///
/// The extension's KTX2 image is the primary representation when present; the
/// texture's own `source` is the fallback for loaders without basis support.
/// `Texture::source` is deliberately not called, because it panics on a
/// basisu-only texture whose `source` field is absent.
pub(crate) fn load_texture(
    document: &gltf::Document,
    texture: gltf::Texture<'_>,
    cache: &mut TextureCache,
    base: &Path,
    buffers: &[BufferData],
    srgb: bool,
) -> Option<Arc<TextureData>> {
    let sampler = sampler_data(texture.sampler());
    let key = (texture.index(), srgb, sampler);
    if let Some(cached) = cache.get(&key) {
        return cached.clone();
    }

    let image = image_index(document, &texture)
        .and_then(|index| document.images().nth(index))
        .and_then(|image| image_bytes(image.source(), base, buffers))
        .and_then(|bytes| decode_texture(&bytes, srgb))
        .map(|(image, mips)| {
            Arc::new(TextureData {
                image: Arc::new(image),
                mips: mips.into(),
                sampler,
            })
        });

    cache.insert(key, image.clone());
    image
}

/// The KTX2 image `KHR_texture_basisu` names, else the texture's own `source`.
fn image_index(document: &gltf::Document, texture: &gltf::Texture<'_>) -> Option<usize> {
    if let Some(source) = texture
        .extension_value("KHR_texture_basisu")
        .and_then(|value| value.get("source"))
        .and_then(serde_json::Value::as_u64)
    {
        return Some(source as usize);
    }
    let source = document
        .as_json()
        .textures
        .get(texture.index())?
        .source
        .value();
    (source != u32::MAX as usize).then_some(source)
}

/// Decodes an image payload, KTX2 or otherwise, keeping any authored mip chain.
pub(crate) fn decode_texture(bytes: &[u8], srgb: bool) -> Option<(Image, Vec<MipLevel>)> {
    if is_ktx2(bytes) {
        match load_ktx2(bytes) {
            Ok(ktx) => {
                let image = Image {
                    width: ktx.width,
                    height: ktx.height,
                    rgba8: ktx.rgba8,
                    srgb: ktx.srgb,
                };
                Some((image, ktx.mips))
            }
            Err(error) => {
                log::warn!("model texture: {error}");
                None
            }
        }
    } else {
        match decode_image(bytes, srgb) {
            Ok(image) => Some((image, Vec::new())),
            Err(error) => {
                log::warn!("model texture: {error}");
                None
            }
        }
    }
}

/// The encoded bytes of one glTF image, however the file stores it.
///
/// An external URI, a `data:` URI, or a slice of a buffer view (which in a GLB
/// is the common case for embedded textures).
pub(crate) fn image_bytes(
    source: gltf::image::Source<'_>,
    base: &Path,
    buffers: &[BufferData],
) -> Option<Vec<u8>> {
    match source {
        gltf::image::Source::Uri { uri, .. } => {
            if let Some(rest) = uri.strip_prefix("data:") {
                let (_, payload) = rest.split_once(";base64,")?;
                base64_decode(payload)
            } else {
                std::fs::read(base.join(uri)).ok()
            }
        }
        gltf::image::Source::View { view, .. } => {
            let buffer = buffers.get(view.buffer().index())?;
            let end = view.offset().checked_add(view.length())?;
            buffer.0.get(view.offset()..end).map(|bytes| bytes.to_vec())
        }
    }
}

/// The sampler a texture declares, with the spec defaults for what it omits.
///
/// Every mip-map variant of a filter folds into the filter itself, because mip
/// selection is the sampler's other axis; wrap S and wrap T stay independent,
/// since applying one mode to both axes is wrong for any atlas that clamps U
/// and repeats V.
pub(crate) fn sampler_data(sampler: gltf::texture::Sampler<'_>) -> SamplerData {
    let default = SamplerData::default();
    SamplerData {
        mag_filter: sampler.mag_filter().map_or(default.mag_filter, filter),
        min_filter: sampler.min_filter().map_or(default.min_filter, min_filter),
        wrap_s: wrap(sampler.wrap_s()),
        wrap_t: wrap(sampler.wrap_t()),
    }
}

pub(crate) fn filter(mag: gltf::texture::MagFilter) -> Filter {
    match mag {
        gltf::texture::MagFilter::Nearest => Filter::Nearest,
        gltf::texture::MagFilter::Linear => Filter::Linear,
    }
}

pub(crate) fn min_filter(min: gltf::texture::MinFilter) -> Filter {
    match min {
        gltf::texture::MinFilter::Nearest
        | gltf::texture::MinFilter::NearestMipmapNearest
        | gltf::texture::MinFilter::NearestMipmapLinear => Filter::Nearest,
        gltf::texture::MinFilter::Linear
        | gltf::texture::MinFilter::LinearMipmapNearest
        | gltf::texture::MinFilter::LinearMipmapLinear => Filter::Linear,
    }
}

pub(crate) fn wrap(mode: gltf::texture::WrappingMode) -> Wrap {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => Wrap::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => Wrap::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => Wrap::Repeat,
    }
}

/// Standard base64, tolerating whitespace and stopping at the padding.
///
/// The glTF crate decodes `data:` URIs for buffers itself, but an *image*
/// embedded that way is handed to the caller as a URI, so one of the two has to
/// own the decoder. Kept local and small rather than adding a crate for it.
pub(crate) fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut accumulator: u32 = 0;
    let mut bits: u32 = 0;

    for byte in input.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return None,
        } as u32;
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((accumulator >> bits) as u8);
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_padding_and_ignores_whitespace() {
        assert_eq!(base64_decode("AQIDBA=="), Some(vec![1, 2, 3, 4]));
        assert_eq!(base64_decode("AQID\nBA=="), Some(vec![1, 2, 3, 4]));
        assert_eq!(base64_decode(""), Some(Vec::new()));
        assert_eq!(base64_decode("AQID*"), None);
    }

    #[test]
    fn min_filter_folds_the_mip_variants() {
        assert_eq!(
            min_filter(gltf::texture::MinFilter::LinearMipmapLinear),
            Filter::Linear
        );
        assert_eq!(
            min_filter(gltf::texture::MinFilter::NearestMipmapNearest),
            Filter::Nearest
        );
    }
}
