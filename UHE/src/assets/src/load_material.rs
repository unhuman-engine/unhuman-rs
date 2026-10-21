//! Material assembly from glTF materials: the five core texture slots and the
//! `KHR_materials_*` blocks (the latter in [`crate::material_extensions`]).
//!
//! The `gltf` crate exposes extension blocks as raw JSON, so every value read
//! here goes through [`crate::json`], which reports a present-but-malformed
//! value instead of defaulting it. A declared extension is either shaded with
//! or reported by [`crate::load_model`], never parsed and dropped.

use std::path::Path;
use std::sync::Arc;

use gltf::buffer::Data as BufferData;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::json;
use crate::load_texture::{load_texture, TextureCache};
use crate::material_extensions;
use crate::mesh::{texture_slot, AlphaMode, MaterialData, TextureData};
use crate::report::LoadReport;
use crate::uv;

/// Every material in the file, with its textures decoded.
///
/// Colour space is a property of the *slot*: `baseColor` and `emissive` are
/// sRGB, `metallicRoughness`, `normal` and `occlusion` are linear data, and
/// using one format for all of them gamma-decodes the linear maps on every
/// sample.
pub(crate) fn load_materials(
    document: &gltf::Document,
    base: &Path,
    buffers: &[BufferData],
    report: &mut LoadReport,
) -> Result<Vec<MaterialData>> {
    let mut cache = TextureCache::new();
    let mut materials = Vec::new();

    for material in document.materials() {
        let context = material.name().unwrap_or("material").to_owned();
        let mut out = MaterialData {
            name: context.clone(),
            ..Default::default()
        };

        let pbr = material.pbr_metallic_roughness();
        out.base_color = pbr.base_color_factor();
        out.metallic = pbr.metallic_factor();
        out.roughness = pbr.roughness_factor();
        out.emissive = material.emissive_factor();
        out.alpha = match material.alpha_mode() {
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        };
        out.alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);
        out.double_sided = material.double_sided();

        if let Some(info) = pbr.base_color_texture() {
            out.albedo = assign_texture(
                &mut out,
                texture_slot::ALBEDO,
                document,
                info.texture(),
                info.extension_value("KHR_texture_transform"),
                &mut cache,
                base,
                buffers,
                true,
                &context,
                report,
            )?;
        }
        if let Some(info) = pbr.metallic_roughness_texture() {
            // Metallic (B) and roughness (G) are scalars, not colour: gamma
            // decoding them shifts both.
            out.metallic_roughness = assign_texture(
                &mut out,
                texture_slot::METALLIC_ROUGHNESS,
                document,
                info.texture(),
                info.extension_value("KHR_texture_transform"),
                &mut cache,
                base,
                buffers,
                false,
                &context,
                report,
            )?;
        }

        if let Some(normal) = material.normal_texture() {
            out.normal = assign_texture(
                &mut out,
                texture_slot::NORMAL,
                document,
                normal.texture(),
                normal.extension_value("KHR_texture_transform"),
                &mut cache,
                base,
                buffers,
                false,
                &context,
                report,
            )?;
            out.normal_scale = normal.scale();
        }
        if let Some(occlusion) = material.occlusion_texture() {
            out.occlusion = assign_texture(
                &mut out,
                texture_slot::OCCLUSION,
                document,
                occlusion.texture(),
                occlusion.extension_value("KHR_texture_transform"),
                &mut cache,
                base,
                buffers,
                false,
                &context,
                report,
            )?;
            out.occlusion_strength = occlusion.strength();
        }
        if let Some(info) = material.emissive_texture() {
            out.emissive_texture = assign_texture(
                &mut out,
                texture_slot::EMISSIVE,
                document,
                info.texture(),
                info.extension_value("KHR_texture_transform"),
                &mut cache,
                base,
                buffers,
                true,
                &context,
                report,
            )?;
        }

        material_extensions::apply(
            document, &material, &mut out, &mut cache, base, buffers, &context, report,
        )?;

        materials.push(out);
    }

    Ok(materials)
}

/// Loads one texture into `slot` and records the slot's UV transform.
#[allow(clippy::too_many_arguments)]
fn assign_texture(
    out: &mut MaterialData,
    slot: usize,
    document: &gltf::Document,
    texture: gltf::Texture<'_>,
    transform: Option<&Value>,
    cache: &mut TextureCache,
    base: &Path,
    buffers: &[BufferData],
    srgb: bool,
    context: &str,
    report: &mut LoadReport,
) -> Result<Option<Arc<TextureData>>> {
    if let Some(transform) = transform {
        out.uv_transforms[slot] = uv::transform(transform, context)?;
    }
    Ok(load_texture(
        document, texture, cache, base, buffers, srgb, context, report,
    ))
}

/// Loads a texture-info object from an extension block into `slot`.
///
/// The object is raw JSON because the per-extension typed accessors are not
/// built in this configuration. A missing `index`, or one past the end of the
/// texture list, is an error: the map was declared and cannot be dropped
/// silently.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assign_extension_slot(
    out: &mut MaterialData,
    slot: usize,
    document: &gltf::Document,
    info: Option<&Value>,
    cache: &mut TextureCache,
    base: &Path,
    buffers: &[BufferData],
    srgb: bool,
    context: &str,
    report: &mut LoadReport,
) -> Result<Option<Arc<TextureData>>> {
    let Some(info) = info else {
        return Ok(None);
    };
    let index = json::index(info, "index", context)?;
    let texture = document.textures().nth(index).ok_or_else(|| {
        Error::Decode(format!("{context}: texture index {index} is out of range"))
    })?;
    let transform = info
        .get("extensions")
        .and_then(|ext| ext.get("KHR_texture_transform"));
    assign_texture(
        out, slot, document, texture, transform, cache, base, buffers, srgb, context, report,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_misspelled_extension_index_is_an_error() {
        let value = serde_json::json!({ "index": "not a number" });
        assert!(json::index(&value, "index", "Mat").is_err());
    }
}
