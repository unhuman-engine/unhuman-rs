//! Typed reads of glTF extension JSON, with malformed values reported.
//!
//! The `gltf` crate hands extension blocks back as raw [`Value`] because the
//! per-extension typed accessors are feature-gated. Reading them with
//! `value.get(key).and_then(as_f64).unwrap_or(default)` cannot tell an absent
//! key from a present one of the wrong type: both become the default, so a
//! file that misspells `emissiveStrength` as a string renders with strength 1.0
//! and no complaint. These readers separate the two.

use serde_json::Value;

use crate::error::{Error, Result};

/// An optional scalar: absent or null is `default`, anything non-numeric is an
/// error naming the key and its owner.
pub(crate) fn scalar(value: &Value, key: &str, default: f32, context: &str) -> Result<f32> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(found) => found
            .as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| malformed(context, key, "a number")),
    }
}

/// An optional 2-vector. Present elements must be numeric; missing ones keep
/// the default, so a short array is not silently zero-filled.
pub(crate) fn vec2(value: &Value, key: &str, default: [f32; 2], context: &str) -> Result<[f32; 2]> {
    let mut out = default;
    fill(value, key, &mut out, context)?;
    Ok(out)
}

pub(crate) fn vec3(value: &Value, key: &str, default: [f32; 3], context: &str) -> Result<[f32; 3]> {
    let mut out = default;
    fill(value, key, &mut out, context)?;
    Ok(out)
}

pub(crate) fn vec4(value: &Value, key: &str, default: [f32; 4], context: &str) -> Result<[f32; 4]> {
    let mut out = default;
    fill(value, key, &mut out, context)?;
    Ok(out)
}

/// An optional `u64` index: absent or null is an error only when the caller
/// requires it.
pub(crate) fn index(value: &Value, key: &str, context: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| malformed(context, key, "an integer index"))
}

fn fill<const N: usize>(value: &Value, key: &str, out: &mut [f32; N], context: &str) -> Result<()> {
    let Some(array) = value.get(key) else {
        return Ok(());
    };
    if array.is_null() {
        return Ok(());
    }
    let array = array
        .as_array()
        .ok_or_else(|| malformed(context, key, "an array"))?;
    for (slot, item) in out.iter_mut().zip(array) {
        if item.is_null() {
            continue;
        }
        *slot = item
            .as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| malformed(context, key, "an array of numbers"))?;
    }
    Ok(())
}

fn malformed(context: &str, key: &str, expected: &str) -> Error {
    Error::Decode(format!("{context}: '{key}' is not {expected}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_key_is_the_default_and_a_wrong_type_is_an_error() {
        let value = serde_json::json!({ "good": 2.5, "bad": "high" });
        assert_eq!(scalar(&value, "absent", 1.0, "M").unwrap(), 1.0);
        assert_eq!(scalar(&value, "good", 0.0, "M").unwrap(), 2.5);
        assert!(scalar(&value, "bad", 0.0, "M").is_err());
    }

    #[test]
    fn a_short_vector_keeps_defaults_but_a_wrong_element_errors() {
        let short = serde_json::json!({ "v": [0.5] });
        assert_eq!(vec3(&short, "v", [1.0; 3], "M").unwrap(), [0.5, 1.0, 1.0]);

        let wrong = serde_json::json!({ "v": [0.5, "x", 0.0] });
        assert!(vec3(&wrong, "v", [1.0; 3], "M").is_err());
    }
}
