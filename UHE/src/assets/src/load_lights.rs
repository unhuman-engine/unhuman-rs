//! `KHR_lights_punctual` parsing.
//!
//! The glTF crate's typed `Document::lights` accessor is feature-gated and this
//! workspace does not enable it, so the light definitions are read from the
//! document's raw extension JSON.

use serde_json::Value;

use crate::mesh::{LightType, PunctualLight};

/// Every light the document declares, in file order.
pub(crate) fn parse_lights(document: &gltf::Document) -> Vec<PunctualLight> {
    let Some(lights) = document
        .extension_value("KHR_lights_punctual")
        .and_then(|value| value.get("lights"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };

    lights.iter().map(parse_light).collect()
}

fn parse_light(value: &Value) -> PunctualLight {
    let light_type = match value.get("type").and_then(Value::as_str) {
        Some("directional") => LightType::Directional,
        Some("spot") => LightType::Spot,
        _ => LightType::Point,
    };

    let spot = value.get("spot");
    PunctualLight {
        light_type,
        color: vec3(value.get("color"), [1.0, 1.0, 1.0]),
        intensity: number(value, "intensity").unwrap_or(1.0),
        range: number(value, "range").unwrap_or(f32::INFINITY),
        inner_cone_angle: spot
            .and_then(|value| number(value, "innerConeAngle"))
            .unwrap_or(0.0),
        outer_cone_angle: spot
            .and_then(|value| number(value, "outerConeAngle"))
            .unwrap_or(std::f32::consts::FRAC_PI_4),
        name: value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    }
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

fn number(value: &Value, key: &str) -> Option<f32> {
    value.get(key)?.as_f64().map(|v| v as f32)
}

fn vec3(value: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    let Some(array) = value.and_then(Value::as_array) else {
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
