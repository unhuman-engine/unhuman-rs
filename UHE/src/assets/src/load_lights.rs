//! `KHR_lights_punctual` parsing.
//!
//! The glTF crate's typed `Document::lights` accessor is feature-gated and this
//! workspace does not enable it, so the light definitions are read from the
//! document's raw extension JSON through [`crate::json`], which reports a
//! present-but-malformed field instead of defaulting it.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::json;
use crate::mesh::{LightType, PunctualLight};

/// Every light the document declares, in file order.
pub(crate) fn parse_lights(document: &gltf::Document) -> Result<Vec<PunctualLight>> {
    let Some(lights) = document
        .extension_value("KHR_lights_punctual")
        .and_then(|value| value.get("lights"))
        .and_then(Value::as_array)
    else {
        return Ok(Vec::new());
    };

    lights
        .iter()
        .enumerate()
        .map(|(index, value)| parse_light(value, index))
        .collect()
}

fn parse_light(value: &Value, index: usize) -> Result<PunctualLight> {
    let context = format!("KHR_lights_punctual.lights[{index}]");
    let light_type = match value.get("type") {
        None => LightType::Point,
        Some(Value::String(kind)) => match kind.as_str() {
            "directional" => LightType::Directional,
            "spot" => LightType::Spot,
            "point" => LightType::Point,
            other => {
                return Err(Error::Decode(format!(
                    "{context}: unknown light type '{other}'"
                )))
            }
        },
        Some(_) => return Err(Error::Decode(format!("{context}: 'type' is not a string"))),
    };

    let spot = value.get("spot");
    Ok(PunctualLight {
        light_type,
        color: json::vec3(value, "color", [1.0, 1.0, 1.0], &context)?,
        intensity: json::scalar(value, "intensity", 1.0, &context)?,
        range: json::scalar(value, "range", f32::INFINITY, &context)?,
        inner_cone_angle: match spot {
            Some(spot) => json::scalar(spot, "innerConeAngle", 0.0, &context)?,
            None => 0.0,
        },
        outer_cone_angle: match spot {
            Some(spot) => json::scalar(
                spot,
                "outerConeAngle",
                std::f32::consts::FRAC_PI_4,
                &context,
            )?,
            None => std::f32::consts::FRAC_PI_4,
        },
        name: value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    })
}

/// The light a node places, validated against the number of declared lights.
///
/// A malformed file may name a light past the end of the list; the clamp here
/// keeps the scene code that instantiates it from reading out of bounds.
pub(crate) fn node_light(node: &gltf::Node<'_>, light_count: usize) -> i32 {
    let Some(index) = node
        .extension_value("KHR_lights_punctual")
        .and_then(|value| value.get("light"))
        .and_then(Value::as_u64)
    else {
        return -1;
    };

    if (index as usize) < light_count {
        index as i32
    } else {
        log::warn!("node references light {index} but the file declares {light_count}; ignoring");
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_malformed_light_field_is_an_error() {
        let value = serde_json::json!({ "type": "point", "intensity": "bright" });
        assert!(parse_light(&value, 0).is_err());
        let wrong_type = serde_json::json!({ "type": 7 });
        assert!(parse_light(&wrong_type, 0).is_err());
    }
}
