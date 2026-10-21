//! Light, physics, audio and animation components.

use yaml_serde::Value;

use crate::components::{
    Animator, AudioSource, Collider, ColliderShape, Light, LightKind, RigidBody, RigidBodyKind,
    SpriteAnimation,
};
use crate::entity::Entity;
use crate::scene::Scene;

use super::read::read_vec2;
use super::value::{as_f32, read_string, read_v3, read_v4};

pub(super) fn components(scene: &mut Scene, entity: Entity, node: &Value) {
    for (name, kind) in [
        ("DirectionalLightComponent", LightKind::Directional),
        ("PointLightComponent", LightKind::Point),
        ("SpotLightComponent", LightKind::Spot),
    ] {
        let Some(node) = node.get(name) else {
            continue;
        };
        let default = Light::default();
        scene.insert(
            entity,
            Light {
                kind,
                color: node.get("Color").and_then(read_v3).unwrap_or(default.color),
                intensity: node
                    .get("Intensity")
                    .and_then(as_f32)
                    .unwrap_or(default.intensity),
                radius: node
                    .get("Radius")
                    .and_then(as_f32)
                    .unwrap_or(default.radius),
                inner_cone_angle: node
                    .get("InnerConeAngle")
                    .and_then(as_f32)
                    .map(f32::to_degrees)
                    .unwrap_or(default.inner_cone_angle),
                outer_cone_angle: node
                    .get("OuterConeAngle")
                    .and_then(as_f32)
                    .map(f32::to_degrees)
                    .unwrap_or(default.outer_cone_angle),
                enabled: node
                    .get("Enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.enabled),
            },
        );
    }

    if let Some(node) = node.get("RigidBody3DComponent") {
        let default = RigidBody::default();
        let kind = match node.get("BodyType").and_then(Value::as_i64) {
            Some(1) => RigidBodyKind::Kinematic,
            Some(2) => RigidBodyKind::Dynamic,
            Some(_) => RigidBodyKind::Static,
            None => default.kind,
        };
        scene.insert(
            entity,
            RigidBody {
                kind,
                mass: node.get("Mass").and_then(as_f32).unwrap_or(default.mass),
                velocity: node
                    .get("Velocity")
                    .and_then(read_v3)
                    .unwrap_or(default.velocity),
                damping: node
                    .get("LinearDamping")
                    .and_then(as_f32)
                    .unwrap_or(default.damping),
                gravity_scale: node
                    .get("GravityScale")
                    .and_then(as_f32)
                    .unwrap_or(default.gravity_scale),
            },
        );
    }

    if let Some(node) = node.get("BoxCollider3DComponent") {
        let default = Collider::default();
        let shape = match node.get("HalfExtent").and_then(read_v3) {
            Some(half_extent) => ColliderShape::Box { half_extent },
            None => default.shape,
        };
        scene.insert(
            entity,
            Collider {
                shape,
                offset: node
                    .get("Offset")
                    .and_then(read_v3)
                    .unwrap_or(default.offset),
                friction: node
                    .get("Friction")
                    .and_then(as_f32)
                    .unwrap_or(default.friction),
                restitution: node
                    .get("Restitution")
                    .and_then(as_f32)
                    .unwrap_or(default.restitution),
            },
        );
    } else if let Some(node) = node.get("SphereCollider3DComponent") {
        let default = Collider::default();
        scene.insert(
            entity,
            Collider {
                shape: ColliderShape::Sphere {
                    // The `SphereCollider3DComponent` default radius.
                    radius: node.get("Radius").and_then(as_f32).unwrap_or(0.5),
                },
                offset: node
                    .get("Offset")
                    .and_then(read_v3)
                    .unwrap_or(default.offset),
                friction: node
                    .get("Friction")
                    .and_then(as_f32)
                    .unwrap_or(default.friction),
                restitution: node
                    .get("Restitution")
                    .and_then(as_f32)
                    .unwrap_or(default.restitution),
            },
        );
    } else if let Some(node) = node.get("CapsuleCollider3DComponent") {
        let default = Collider::default();
        scene.insert(
            entity,
            Collider {
                shape: ColliderShape::Capsule {
                    // The `CapsuleCollider3DComponent` defaults.
                    radius: node.get("Radius").and_then(as_f32).unwrap_or(0.5),
                    half_height: node.get("HalfHeight").and_then(as_f32).unwrap_or(1.0),
                },
                offset: node
                    .get("Offset")
                    .and_then(read_v3)
                    .unwrap_or(default.offset),
                friction: node
                    .get("Friction")
                    .and_then(as_f32)
                    .unwrap_or(default.friction),
                restitution: node
                    .get("Restitution")
                    .and_then(as_f32)
                    .unwrap_or(default.restitution),
            },
        );
    }

    if let Some(node) = node.get("AudioSourceComponent") {
        let default = AudioSource::default();
        scene.insert(
            entity,
            AudioSource {
                clip: None,
                clip_path: node
                    .get("ClipPath")
                    .and_then(read_string)
                    .unwrap_or(default.clip_path),
                playing: node
                    .get("Playing")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.playing),
                looping: node
                    .get("Looping")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.looping),
                gain: node.get("Gain").and_then(as_f32).unwrap_or(default.gain),
                pitch: node.get("Pitch").and_then(as_f32).unwrap_or(default.pitch),
                spatial: node
                    .get("Spatial")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.spatial),
                range: node.get("Range").and_then(as_f32).unwrap_or(default.range),
            },
        );
    }

    if let Some(node) = node.get("AnimatorComponent") {
        let default = Animator::default();
        scene.insert(
            entity,
            Animator {
                clip: None,
                clip_path: node
                    .get("CurrentAnimationName")
                    .and_then(read_string)
                    .unwrap_or(default.clip_path),
                time: node.get("Time").and_then(as_f32).unwrap_or(default.time),
                speed: node
                    .get("PlaybackSpeed")
                    .and_then(as_f32)
                    .unwrap_or(default.speed),
                // LoopMode: None = 0, Loop = 1, PingPong = 2.
                looping: match node.get("LoopMode").and_then(Value::as_i64) {
                    Some(mode) => mode == 1,
                    None => default.looping,
                },
                playing: node
                    .get("IsPlaying")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.playing),
            },
        );
    }

    if let Some(node) = node.get("SpriteAnimationComponent") {
        let default = SpriteAnimation::default();
        scene.insert(
            entity,
            SpriteAnimation {
                sprite_sheet_path: node
                    .get("SpriteSheetPath")
                    .and_then(read_string)
                    .unwrap_or(default.sprite_sheet_path),
                sheet: None,
                frame_size: read_vec2(node, "FrameSizeX", "FrameSizeY", default.frame_size),
                frame_count: node
                    .get("FrameCount")
                    .and_then(Value::as_u64)
                    .map(|count| count as u32)
                    .unwrap_or(default.frame_count),
                frame_duration: node
                    .get("FrameDuration")
                    .and_then(as_f32)
                    .unwrap_or(default.frame_duration),
                looping: node
                    .get("Loop")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.looping),
                color: node.get("Color").and_then(read_v4).unwrap_or(default.color),
                time: node.get("Time").and_then(as_f32).unwrap_or(default.time),
            },
        );
    }
}
