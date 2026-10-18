//! Material assembly from glTF materials and their extension values.
//!
//! The `gltf` crate exposes extension blocks through `extension_value`, which
//! returns raw JSON, because the per-extension typed accessors are feature-gated
//! and this workspace enables only `utils`/`names`/`extensions`. Every value
//! read here lands on [`MaterialData`], so a declared extension is either shaded
//! with or reported by [`crate::load_model`], never parsed and dropped.

use std::path::Path;
use std::sync::Arc;

use gltf::buffer::Data as BufferData;
use serde_json::Value;

use crate::load_texture::{load_texture, TextureCache};
use crate::mesh::{texture_slot, AlphaMode, BlendApproach, MaterialData, TextureData, UVTransform};

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
) -> Vec<MaterialData> {
    let mut cache = TextureCache::new();
    let mut materials = Vec::new();

    for material in document.materials() {
        let name = material.name().map(str::to_owned).unwrap_or_default();
        let mut out = MaterialData {
            name,
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
            let transform = info.extension_value("KHR_texture_transform");
            out.albedo = assign_texture(
                &mut out,
                texture_slot::ALBEDO,
                document,
                info.texture(),
                transform,
                &mut cache,
                base,
                buffers,
                true,
            );
        }
        if let Some(info) = pbr.metallic_roughness_texture() {
            // Metallic (B) and roughness (G) are scalars, not colour: gamma
            // decoding them shifts both.
            let transform = info.extension_value("KHR_texture_transform");
            out.metallic_roughness = assign_texture(
                &mut out,
                texture_slot::METALLIC_ROUGHNESS,
                document,
                info.texture(),
                transform,
                &mut cache,
                base,
                buffers,
                false,
            );
        }

        if let Some(normal) = material.normal_texture() {
            let transform = normal.extension_value("KHR_texture_transform");
            let texture = assign_texture(
                &mut out,
                texture_slot::NORMAL,
                document,
                normal.texture(),
                transform,
                &mut cache,
                base,
                buffers,
                false,
            );
            out.normal = texture;
            out.normal_scale = normal.scale();
        }
        if let Some(occlusion) = material.occlusion_texture() {
            let transform = occlusion.extension_value("KHR_texture_transform");
            let texture = assign_texture(
                &mut out,
                texture_slot::OCCLUSION,
                document,
                occlusion.texture(),
                transform,
                &mut cache,
                base,
                buffers,
                false,
            );
            out.occlusion = texture;
            out.occlusion_strength = occlusion.strength();
        }
        if let Some(info) = material.emissive_texture() {
            let transform = info.extension_value("KHR_texture_transform");
            out.emissive_texture = assign_texture(
                &mut out,
                texture_slot::EMISSIVE,
                document,
                info.texture(),
                transform,
                &mut cache,
                base,
                buffers,
                true,
            );
        }

        apply_extensions(document, &material, &mut out, &mut cache, base, buffers);

        materials.push(out);
    }

    materials
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
) -> Option<Arc<TextureData>> {
    if let Some(transform) = transform {
        set_transform(&mut out.uv_transforms[slot], transform, &out.name);
    }
    load_texture(document, texture, cache, base, buffers, srgb)
}

/// Loads an extension texture-info object into `slot`, recording its transform.
///
/// The info is raw JSON because the per-extension typed accessors are not built
/// in this configuration.
#[allow(clippy::too_many_arguments)]
fn assign_extension_slot(
    out: &mut MaterialData,
    slot: usize,
    document: &gltf::Document,
    info: Option<&Value>,
    cache: &mut TextureCache,
    base: &Path,
    buffers: &[BufferData],
    srgb: bool,
) -> Option<Arc<TextureData>> {
    let info = info?;
    let index = info.get("index").and_then(Value::as_u64)? as usize;
    let texture = document.textures().nth(index)?;
    let transform = info
        .get("extensions")
        .and_then(|ext| ext.get("KHR_texture_transform"));
    assign_texture(
        out, slot, document, texture, transform, cache, base, buffers, srgb,
    )
}

/// Converts a `KHR_texture_transform` into engine-space parameters.
///
/// The imported TEXCOORD_0 has V flipped, so the file's raw transform does not
/// apply to it. Working `uv' = R(theta) * (uv * scale) + offset` through the
/// flip gives the engine-space offset below, with `cos`/`sin` of the original
/// angle.
fn set_transform(target: &mut UVTransform, value: &Value, material_name: &str) {
    let rotation = number(value, "rotation").unwrap_or(0.0);
    let (sin, cos) = rotation.sin_cos();
    let scale = vec2(value, "scale", [1.0, 1.0]);
    let offset = vec2(value, "offset", [0.0, 0.0]);
    let tex_coord_set = value.get("texCoord").and_then(Value::as_u64).unwrap_or(0) as u32;

    *target = UVTransform {
        cos_rotation: cos,
        sin_rotation: sin,
        scale,
        offset: [offset[0] - sin * scale[1], 1.0 - cos * scale[1] - offset[1]],
        tex_coord_set,
        has_transform: true,
    };

    if tex_coord_set != 0 {
        log::warn!(
            "material {material_name}: texture transform targets TEXCOORD_{tex_coord_set}; only \
             TEXCOORD_0 exists, applying it to TEXCOORD_0"
        );
    }
}

/// Reads every optional PBR extension block off the material.
fn apply_extensions(
    document: &gltf::Document,
    material: &gltf::material::Material<'_>,
    out: &mut MaterialData,
    cache: &mut TextureCache,
    base: &Path,
    buffers: &[BufferData],
) {
    let ext = |key: &str| material.extension_value(key);

    // Scalars the parser always defaults correctly: strength 1.0 is a no-op,
    // 1.5 is the dielectric IOR, and unlit is a bare flag.
    out.extensions.emissive_strength = ext("KHR_materials_emissive_strength")
        .and_then(|value| number(value, "emissiveStrength"))
        .unwrap_or(1.0);
    out.extensions.ior = ext("KHR_materials_ior")
        .and_then(|value| number(value, "ior"))
        .unwrap_or(1.5);
    out.extensions.unlit = ext("KHR_materials_unlit").is_some();

    if let Some(value) = ext("KHR_materials_clearcoat") {
        out.extensions.has_clearcoat = true;
        out.extensions.clearcoat_factor = number(value, "clearcoatFactor").unwrap_or(0.0);
        out.extensions.clearcoat_roughness_factor =
            number(value, "clearcoatRoughnessFactor").unwrap_or(0.0);
        out.extensions.clearcoat_normal_scale = value
            .get("clearcoatNormalTexture")
            .and_then(|t| number(t, "scale"))
            .unwrap_or(1.0);
        let clearcoat = assign_extension_slot(
            out,
            texture_slot::CLEARCOAT,
            document,
            value.get("clearcoatTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.clearcoat_texture = clearcoat;
        let roughness = assign_extension_slot(
            out,
            texture_slot::CLEARCOAT_ROUGHNESS,
            document,
            value.get("clearcoatRoughnessTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.clearcoat_roughness_texture = roughness;
        let normal = assign_extension_slot(
            out,
            texture_slot::CLEARCOAT_NORMAL,
            document,
            value.get("clearcoatNormalTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.clearcoat_normal_texture = normal;
    }

    if let Some(value) = ext("KHR_materials_specular") {
        out.extensions.has_specular = true;
        out.extensions.specular_factor = number(value, "specularFactor").unwrap_or(1.0);
        out.extensions.specular_color_factor = vec3(value, "specularColorFactor", [1.0, 1.0, 1.0]);
        let specular = assign_extension_slot(
            out,
            texture_slot::SPECULAR,
            document,
            value.get("specularTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.specular_texture = specular;
        let color = assign_extension_slot(
            out,
            texture_slot::SPECULAR_COLOR,
            document,
            value.get("specularColorTexture"),
            cache,
            base,
            buffers,
            true,
        );
        out.extensions.specular_color_texture = color;
    }

    if let Some(value) = ext("KHR_materials_sheen") {
        out.extensions.has_sheen = true;
        out.extensions.sheen_color_factor = vec3(value, "sheenColorFactor", [0.0, 0.0, 0.0]);
        out.extensions.sheen_roughness_factor =
            number(value, "sheenRoughnessFactor").unwrap_or(0.0);
        let color = assign_extension_slot(
            out,
            texture_slot::SHEEN_COLOR,
            document,
            value.get("sheenColorTexture"),
            cache,
            base,
            buffers,
            true,
        );
        out.extensions.sheen_color_texture = color;
        let roughness = assign_extension_slot(
            out,
            texture_slot::SHEEN_ROUGHNESS,
            document,
            value.get("sheenRoughnessTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.sheen_roughness_texture = roughness;
    }

    // Transmission and volume are separate extensions but only meaningful
    // together: thickness and attenuation describe the medium a transmitted ray
    // travels through.
    if let Some(value) = ext("KHR_materials_transmission") {
        out.extensions.has_transmission = true;
        out.extensions.transmission_factor = number(value, "transmissionFactor").unwrap_or(0.0);
        let texture = assign_extension_slot(
            out,
            texture_slot::TRANSMISSION,
            document,
            value.get("transmissionTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.transmission_texture = texture;
    }
    if let Some(value) = ext("KHR_materials_volume") {
        out.extensions.has_transmission = true;
        out.extensions.thickness_factor = number(value, "thicknessFactor").unwrap_or(0.0);
        out.extensions.attenuation_distance =
            number(value, "attenuationDistance").unwrap_or(f32::INFINITY);
        out.extensions.attenuation_color = vec3(value, "attenuationColor", [1.0, 1.0, 1.0]);
        let texture = assign_extension_slot(
            out,
            texture_slot::THICKNESS,
            document,
            value.get("thicknessTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.thickness_texture = texture;
    }
    if out.extensions.has_transmission {
        out.extensions.transmission_blend = if out.extensions.transmission_factor > 0.0 {
            BlendApproach::Blend
        } else {
            BlendApproach::OpaqueWithTransmission
        };
    }

    if let Some(value) = ext("KHR_materials_iridescence") {
        out.extensions.has_iridescence = true;
        out.extensions.iridescence_factor = number(value, "iridescenceFactor").unwrap_or(0.0);
        out.extensions.iridescence_ior = number(value, "iridescenceIor").unwrap_or(1.3);
        out.extensions.iridescence_thickness_minimum =
            number(value, "iridescenceThicknessMinimum").unwrap_or(100.0);
        out.extensions.iridescence_thickness_maximum =
            number(value, "iridescenceThicknessMaximum").unwrap_or(400.0);
        let texture = assign_extension_slot(
            out,
            texture_slot::IRIDESCENCE,
            document,
            value.get("iridescenceTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.iridescence_texture = texture;
        let thickness = assign_extension_slot(
            out,
            texture_slot::IRIDESCENCE_THICKNESS,
            document,
            value.get("iridescenceThicknessTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.iridescence_thickness_texture = thickness;
    }

    if let Some(value) = ext("KHR_materials_anisotropy") {
        out.extensions.has_anisotropy = true;
        out.extensions.anisotropy_strength = number(value, "anisotropyStrength").unwrap_or(0.0);
        out.extensions.anisotropy_rotation = number(value, "anisotropyRotation").unwrap_or(0.0);
        let texture = assign_extension_slot(
            out,
            texture_slot::ANISOTROPY,
            document,
            value.get("anisotropyTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.anisotropy_texture = texture;
    }

    if let Some(value) = ext("KHR_materials_diffuse_transmission") {
        out.extensions.has_diffuse_transmission = true;
        out.extensions.diffuse_transmission_factor =
            number(value, "diffuseTransmissionFactor").unwrap_or(0.0);
        out.extensions.diffuse_transmission_color =
            vec3(value, "diffuseTransmissionColorFactor", [1.0, 1.0, 1.0]);
        let texture = assign_extension_slot(
            out,
            texture_slot::DIFFUSE_TRANSMISSION,
            document,
            value.get("diffuseTransmissionTexture"),
            cache,
            base,
            buffers,
            false,
        );
        out.extensions.diffuse_transmission_texture = texture;
        let color = assign_extension_slot(
            out,
            texture_slot::DIFFUSE_TRANSMISSION_COLOR,
            document,
            value.get("diffuseTransmissionColorTexture"),
            cache,
            base,
            buffers,
            true,
        );
        out.extensions.diffuse_transmission_color_texture = color;
    }

    // KHR_materials_pbrSpecularGlossiness. Its diffuse map is the base colour
    // when the metallic-roughness path declares none, and the specular and
    // glossiness factors reach shading in place of the metallic-roughness ones.
    if let Some(value) = ext("KHR_materials_pbrSpecularGlossiness") {
        out.extensions.has_specular_glossiness = true;
        out.extensions.sg_diffuse_factor = vec4(value, "diffuseFactor", [1.0, 1.0, 1.0, 1.0]);
        out.extensions.sg_specular_factor = vec3(value, "specularFactor", [1.0, 1.0, 1.0]);
        out.extensions.sg_glossiness_factor = number(value, "glossinessFactor").unwrap_or(1.0);

        let diffuse_slot = if out.albedo.is_none() {
            texture_slot::ALBEDO
        } else {
            crate::mesh::texture_slot::SPECULAR
        };
        let diffuse = assign_extension_slot(
            out,
            diffuse_slot,
            document,
            value.get("diffuseTexture"),
            cache,
            base,
            buffers,
            true,
        );
        out.extensions.sg_diffuse_texture = diffuse.clone();
        if out.albedo.is_none() {
            out.albedo = diffuse;
        }

        let spec_gloss = assign_extension_slot(
            out,
            crate::mesh::texture_slot::SPECULAR,
            document,
            value.get("specularGlossinessTexture"),
            cache,
            base,
            buffers,
            true,
        );
        out.extensions.sg_specular_glossiness_texture = spec_gloss;
    }
}

fn number(value: &Value, key: &str) -> Option<f32> {
    value.get(key)?.as_f64().map(|v| v as f32)
}

fn vec2(value: &Value, key: &str, default: [f32; 2]) -> [f32; 2] {
    let Some(array) = value.get(key).and_then(Value::as_array) else {
        return default;
    };
    let mut out = default;
    for (slot, item) in out.iter_mut().zip(array) {
        if let Some(v) = item.as_f64() {
            *slot = v as f32;
        }
    }
    out
}

fn vec3(value: &Value, key: &str, default: [f32; 3]) -> [f32; 3] {
    let Some(array) = value.get(key).and_then(Value::as_array) else {
        return default;
    };
    let mut out = default;
    for (slot, item) in out.iter_mut().zip(array) {
        if let Some(v) = item.as_f64() {
            *slot = v as f32;
        }
    }
    out
}

fn vec4(value: &Value, key: &str, default: [f32; 4]) -> [f32; 4] {
    let Some(array) = value.get(key).and_then(Value::as_array) else {
        return default;
    };
    let mut out = default;
    for (slot, item) in out.iter_mut().zip(array) {
        if let Some(v) = item.as_f64() {
            *slot = v as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_offsets_the_flipped_axis() {
        let mut target = UVTransform::default();
        let value = serde_json::json!({
            "offset": [0.25, 0.5],
            "scale": [2.0, 3.0],
            "rotation": 0.0,
        });
        set_transform(&mut target, &value, "Mat");
        assert!(target.has_transform);
        assert_eq!(target.scale, [2.0, 3.0]);
        // cos(0)=1, sin(0)=0: offset.x unchanged, offset.y becomes 1 - scale.y - offset.y.
        assert_eq!(target.offset, [0.25, 1.0 - 3.0 - 0.5]);
    }

    #[test]
    fn vector_readers_tolerate_short_arrays() {
        let value = serde_json::json!({ "v": [0.5] });
        assert_eq!(vec3(&value, "v", [1.0, 1.0, 1.0]), [0.5, 1.0, 1.0]);
        assert_eq!(vec4(&value, "missing", [2.0; 4]), [2.0; 4]);
    }
}
