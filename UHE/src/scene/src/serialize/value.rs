//! YAML scalar and vector helpers shared by the reader and the writer.

use glam::{Vec2, Vec3, Vec4};
use yaml_serde::Value;

pub(super) fn key(name: &str) -> Value {
    Value::String(name.to_string())
}

pub(super) fn f32_value(value: f32) -> Value {
    // Use the shortest f32 spelling, not `From<f32>`, which prints `0.05` as
    // `0.05000000074505806`; the fallback covers `inf` and `NaN`.
    if value == 0.0 {
        // "-0" parses as an integer and loses the sign; only the float spelling
        // round-trips negative zero.
        return Value::Number(f64::from(value).into());
    }
    match value.to_string().parse::<yaml_serde::Number>() {
        Ok(number) => Value::Number(number),
        Err(_) => Value::Number(f64::from(value).into()),
    }
}

pub(super) fn int(value: i32) -> Value {
    Value::Number(value.into())
}

pub(super) fn v2(value: Vec2) -> Value {
    Value::Sequence(vec![f32_value(value.x), f32_value(value.y)])
}

pub(super) fn v3(value: Vec3) -> Value {
    Value::Sequence(vec![
        f32_value(value.x),
        f32_value(value.y),
        f32_value(value.z),
    ])
}

pub(super) fn v4(value: Vec4) -> Value {
    Value::Sequence(vec![
        f32_value(value.x),
        f32_value(value.y),
        f32_value(value.z),
        f32_value(value.w),
    ])
}

pub(super) fn as_f32(value: &Value) -> Option<f32> {
    value.as_f64().map(|value| value as f32)
}

pub(super) fn read_string(value: &Value) -> Option<String> {
    value.as_str().map(str::to_string)
}

pub(super) fn read_v2(value: &Value) -> Option<Vec2> {
    let values = value.as_sequence()?;
    Some(Vec2::new(as_f32(values.first()?)?, as_f32(values.get(1)?)?))
}

pub(super) fn read_v3(value: &Value) -> Option<Vec3> {
    let values = value.as_sequence()?;
    Some(Vec3::new(
        as_f32(values.first()?)?,
        as_f32(values.get(1)?)?,
        as_f32(values.get(2)?)?,
    ))
}

pub(super) fn read_v4(value: &Value) -> Option<Vec4> {
    let values = value.as_sequence()?;
    Some(Vec4::new(
        as_f32(values.first()?)?,
        as_f32(values.get(1)?)?,
        as_f32(values.get(2)?)?,
        as_f32(values.get(3)?)?,
    ))
}
