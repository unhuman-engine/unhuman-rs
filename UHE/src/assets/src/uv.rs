//! The engine's texture-coordinate convention, in one place.
//!
//! glTF's V points down the image; this engine's points up, so importing
//! TEXCOORD_0 flips V ([`import_uv`], applied by the geometry extractor).
//! Because of that flip, a `KHR_texture_transform` cannot be copied straight
//! onto the imported coordinates: it is defined in glTF space, and the shader
//! applies `uv' = R * (uv * scale) + offset` in engine space. [`transform`]
//! converts the one into the other, and it is the only place that does.
//!
//! The conversion is the V-flip applied to the whole affine map: `F * T * F`,
//! with `F` the flip. The reflection appears in two places -- the offset's
//! second row and the sign of the rotation -- which is why the rotation is
//! stored negated.

use serde_json::Value;

use crate::error::Result;
use crate::json;
use crate::mesh::UVTransform;

/// glTF UV to engine UV: V is flipped so the texture origin is top-left.
pub(crate) fn import_uv(uv: [f32; 2]) -> [f32; 2] {
    [uv[0], 1.0 - uv[1]]
}

/// `KHR_texture_transform` JSON as an engine-space [`UVTransform`].
///
/// Errors when a present field is the wrong type, rather than defaulting a
/// rotation the file meant to set.
pub(crate) fn transform(value: &Value, context: &str) -> Result<UVTransform> {
    let rotation = json::scalar(value, "rotation", 0.0, context)?;
    let (sin, cos) = rotation.sin_cos();
    let scale = json::vec2(value, "scale", [1.0, 1.0], context)?;
    let offset = json::vec2(value, "offset", [0.0, 0.0], context)?;
    let tex_coord_set = match value.get("texCoord") {
        None => 0,
        Some(_) => json::index(value, "texCoord", context)? as u32,
    };

    if tex_coord_set != 0 {
        log::warn!(
            "{context}: texture transform targets TEXCOORD_{tex_coord_set}; only \
             TEXCOORD_0 exists, applying it to TEXCOORD_0"
        );
    }

    Ok(UVTransform {
        cos_rotation: cos,
        // Negated: the V-flip reflects the rotation's sense, and the shader
        // rebuilds R from these two values.
        sin_rotation: -sin,
        scale,
        offset: [offset[0] - sin * scale[1], 1.0 - cos * scale[1] - offset[1]],
        tex_coord_set,
        has_transform: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's engine-space formula, from `model.wgsl`.
    fn engine_apply(t: &UVTransform, uv: [f32; 2]) -> [f32; 2] {
        let scaled = [uv[0] * t.scale[0], uv[1] * t.scale[1]];
        let rotated = [
            scaled[0] * t.cos_rotation - scaled[1] * t.sin_rotation,
            scaled[0] * t.sin_rotation + scaled[1] * t.cos_rotation,
        ];
        [rotated[0] + t.offset[0], rotated[1] + t.offset[1]]
    }

    /// The glTF spec's raw formula: offset + R(theta) * (scale * uv).
    fn gltf_apply(value: &Value, uv: [f32; 2]) -> [f32; 2] {
        let rotation = value["rotation"].as_f64().unwrap_or(0.0) as f32;
        let (sin, cos) = rotation.sin_cos();
        let scale = [
            value["scale"][0].as_f64().unwrap_or(1.0) as f32,
            value["scale"][1].as_f64().unwrap_or(1.0) as f32,
        ];
        let offset = [
            value["offset"][0].as_f64().unwrap_or(0.0) as f32,
            value["offset"][1].as_f64().unwrap_or(0.0) as f32,
        ];
        let scaled = [uv[0] * scale[0], uv[1] * scale[1]];
        [
            scaled[0] * cos - scaled[1] * sin + offset[0],
            scaled[0] * sin + scaled[1] * cos + offset[1],
        ]
    }

    #[test]
    fn the_conversion_matches_flipping_the_gltf_result() {
        // A rotation and a non-square scale, so every term is exercised.
        let value = serde_json::json!({
            "offset": [0.25, 0.5],
            "scale": [2.0, 3.0],
            "rotation": 0.7,
        });
        let t = transform(&value, "Mat").expect("valid transform");

        for uv in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.3, 0.8]] {
            let reference = import_uv(gltf_apply(&value, uv));
            let engine = engine_apply(&t, import_uv(uv));
            assert!(
                (reference[0] - engine[0]).abs() < 1e-5 && (reference[1] - engine[1]).abs() < 1e-5,
                "uv {uv:?}: reference {reference:?} != engine {engine:?}"
            );
        }
    }

    #[test]
    fn a_rotation_with_no_flip_would_have_the_wrong_sign() {
        let value = serde_json::json!({ "rotation": std::f32::consts::FRAC_PI_2 });
        let t = transform(&value, "Mat").expect("valid transform");
        // 90 degrees in glTF is clockwise once V is flipped.
        assert!((t.sin_rotation - -1.0).abs() < 1e-5);
        assert!(t.cos_rotation.abs() < 1e-5);
    }

    #[test]
    fn a_malformed_field_is_an_error() {
        let value = serde_json::json!({ "scale": [1.0, "wide"] });
        assert!(transform(&value, "Mat").is_err());
    }
}
