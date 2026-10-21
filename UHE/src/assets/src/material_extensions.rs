//! The optional `KHR_materials_*` blocks, read into [`MaterialData`].
//!
//! Every value read here lands on a field the renderer shades with, so an
//! extension is never parsed and dropped. A field present with the wrong type
//! is a reported error ([`crate::json`]), not a silent default: a misspelled
//! `emissiveStrength` must not render as 1.0 with nothing said.

use std::path::Path;

use gltf::buffer::Data as BufferData;

use crate::error::Result;
use crate::json;
use crate::load_material::assign_extension_slot;
use crate::load_texture::TextureCache;
use crate::mesh::{texture_slot, BlendApproach, MaterialData};
use crate::report::LoadReport;

/// Reads every optional PBR extension block off the material.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply(
    document: &gltf::Document,
    material: &gltf::material::Material<'_>,
    out: &mut MaterialData,
    cache: &mut TextureCache,
    base: &Path,
    buffers: &[BufferData],
    context: &str,
    report: &mut LoadReport,
) -> Result<()> {
    let ext = |key: &str| material.extension_value(key);

    // Scalars the parser always defaults correctly: strength 1.0 is a no-op,
    // 1.5 is the dielectric IOR, and unlit is a bare flag.
    out.extensions.emissive_strength = match ext("KHR_materials_emissive_strength") {
        Some(value) => json::scalar(value, "emissiveStrength", 1.0, context)?,
        None => 1.0,
    };
    out.extensions.ior = match ext("KHR_materials_ior") {
        Some(value) => json::scalar(value, "ior", 1.5, context)?,
        None => 1.5,
    };
    out.extensions.unlit = ext("KHR_materials_unlit").is_some();

    if let Some(value) = ext("KHR_materials_clearcoat") {
        out.extensions.has_clearcoat = true;
        out.extensions.clearcoat_factor = json::scalar(value, "clearcoatFactor", 0.0, context)?;
        out.extensions.clearcoat_roughness_factor =
            json::scalar(value, "clearcoatRoughnessFactor", 0.0, context)?;
        out.extensions.clearcoat_normal_scale = match value.get("clearcoatNormalTexture") {
            Some(normal) => json::scalar(normal, "scale", 1.0, context)?,
            None => 1.0,
        };
        out.extensions.clearcoat_texture = assign_extension_slot(
            out,
            texture_slot::CLEARCOAT,
            document,
            value.get("clearcoatTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
        out.extensions.clearcoat_roughness_texture = assign_extension_slot(
            out,
            texture_slot::CLEARCOAT_ROUGHNESS,
            document,
            value.get("clearcoatRoughnessTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
        out.extensions.clearcoat_normal_texture = assign_extension_slot(
            out,
            texture_slot::CLEARCOAT_NORMAL,
            document,
            value.get("clearcoatNormalTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
    }

    if let Some(value) = ext("KHR_materials_specular") {
        out.extensions.has_specular = true;
        out.extensions.specular_factor = json::scalar(value, "specularFactor", 1.0, context)?;
        out.extensions.specular_color_factor =
            json::vec3(value, "specularColorFactor", [1.0, 1.0, 1.0], context)?;
        out.extensions.specular_texture = assign_extension_slot(
            out,
            texture_slot::SPECULAR,
            document,
            value.get("specularTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
        out.extensions.specular_color_texture = assign_extension_slot(
            out,
            texture_slot::SPECULAR_COLOR,
            document,
            value.get("specularColorTexture"),
            cache,
            base,
            buffers,
            true,
            context,
            report,
        )?;
    }

    if let Some(value) = ext("KHR_materials_sheen") {
        out.extensions.has_sheen = true;
        out.extensions.sheen_color_factor =
            json::vec3(value, "sheenColorFactor", [0.0, 0.0, 0.0], context)?;
        out.extensions.sheen_roughness_factor =
            json::scalar(value, "sheenRoughnessFactor", 0.0, context)?;
        out.extensions.sheen_color_texture = assign_extension_slot(
            out,
            texture_slot::SHEEN_COLOR,
            document,
            value.get("sheenColorTexture"),
            cache,
            base,
            buffers,
            true,
            context,
            report,
        )?;
        out.extensions.sheen_roughness_texture = assign_extension_slot(
            out,
            texture_slot::SHEEN_ROUGHNESS,
            document,
            value.get("sheenRoughnessTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
    }

    // Transmission and volume are separate extensions but only meaningful
    // together: thickness and attenuation describe the medium a transmitted ray
    // travels through.
    if let Some(value) = ext("KHR_materials_transmission") {
        out.extensions.has_transmission = true;
        out.extensions.transmission_factor =
            json::scalar(value, "transmissionFactor", 0.0, context)?;
        out.extensions.transmission_texture = assign_extension_slot(
            out,
            texture_slot::TRANSMISSION,
            document,
            value.get("transmissionTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
    }
    if let Some(value) = ext("KHR_materials_volume") {
        out.extensions.has_transmission = true;
        out.extensions.thickness_factor = json::scalar(value, "thicknessFactor", 0.0, context)?;
        out.extensions.attenuation_distance =
            json::scalar(value, "attenuationDistance", f32::INFINITY, context)?;
        out.extensions.attenuation_color =
            json::vec3(value, "attenuationColor", [1.0, 1.0, 1.0], context)?;
        out.extensions.thickness_texture = assign_extension_slot(
            out,
            texture_slot::THICKNESS,
            document,
            value.get("thicknessTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
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
        out.extensions.iridescence_factor = json::scalar(value, "iridescenceFactor", 0.0, context)?;
        out.extensions.iridescence_ior = json::scalar(value, "iridescenceIor", 1.3, context)?;
        out.extensions.iridescence_thickness_minimum =
            json::scalar(value, "iridescenceThicknessMinimum", 100.0, context)?;
        out.extensions.iridescence_thickness_maximum =
            json::scalar(value, "iridescenceThicknessMaximum", 400.0, context)?;
        out.extensions.iridescence_texture = assign_extension_slot(
            out,
            texture_slot::IRIDESCENCE,
            document,
            value.get("iridescenceTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
        out.extensions.iridescence_thickness_texture = assign_extension_slot(
            out,
            texture_slot::IRIDESCENCE_THICKNESS,
            document,
            value.get("iridescenceThicknessTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
    }

    if let Some(value) = ext("KHR_materials_anisotropy") {
        out.extensions.has_anisotropy = true;
        out.extensions.anisotropy_strength =
            json::scalar(value, "anisotropyStrength", 0.0, context)?;
        out.extensions.anisotropy_rotation =
            json::scalar(value, "anisotropyRotation", 0.0, context)?;
        out.extensions.anisotropy_texture = assign_extension_slot(
            out,
            texture_slot::ANISOTROPY,
            document,
            value.get("anisotropyTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
    }

    if let Some(value) = ext("KHR_materials_diffuse_transmission") {
        out.extensions.has_diffuse_transmission = true;
        out.extensions.diffuse_transmission_factor =
            json::scalar(value, "diffuseTransmissionFactor", 0.0, context)?;
        out.extensions.diffuse_transmission_color = json::vec3(
            value,
            "diffuseTransmissionColorFactor",
            [1.0, 1.0, 1.0],
            context,
        )?;
        out.extensions.diffuse_transmission_texture = assign_extension_slot(
            out,
            texture_slot::DIFFUSE_TRANSMISSION,
            document,
            value.get("diffuseTransmissionTexture"),
            cache,
            base,
            buffers,
            false,
            context,
            report,
        )?;
        out.extensions.diffuse_transmission_color_texture = assign_extension_slot(
            out,
            texture_slot::DIFFUSE_TRANSMISSION_COLOR,
            document,
            value.get("diffuseTransmissionColorTexture"),
            cache,
            base,
            buffers,
            true,
            context,
            report,
        )?;
    }

    // KHR_materials_pbrSpecularGlossiness. Its diffuse map is the base colour
    // when the metallic-roughness path declares none, and the specular and
    // glossiness factors reach shading in place of the metallic-roughness ones.
    if let Some(value) = ext("KHR_materials_pbrSpecularGlossiness") {
        out.extensions.has_specular_glossiness = true;
        out.extensions.sg_diffuse_factor =
            json::vec4(value, "diffuseFactor", [1.0, 1.0, 1.0, 1.0], context)?;
        out.extensions.sg_specular_factor =
            json::vec3(value, "specularFactor", [1.0, 1.0, 1.0], context)?;
        out.extensions.sg_glossiness_factor =
            json::scalar(value, "glossinessFactor", 1.0, context)?;

        let diffuse_slot = if out.albedo.is_none() {
            texture_slot::ALBEDO
        } else {
            texture_slot::SPECULAR
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
            context,
            report,
        )?;
        out.extensions.sg_diffuse_texture = diffuse.clone();
        if out.albedo.is_none() {
            out.albedo = diffuse;
        }

        out.extensions.sg_specular_glossiness_texture = assign_extension_slot(
            out,
            texture_slot::SPECULAR,
            document,
            value.get("specularGlossinessTexture"),
            cache,
            base,
            buffers,
            true,
            context,
            report,
        )?;
    }

    Ok(())
}
