//! YAML entity mapping -> components.

use std::collections::HashSet;

use glam::Vec2;
use yaml_serde::Value;

use crate::camera::{Camera, ProjectionType, SceneCamera};
use crate::components::{MeshRenderer, SpriteRenderer, Tag, Text, Transform};
use crate::entity::Entity;
use crate::scene::Scene;

use super::value::{as_f32, read_string, read_v2, read_v3, read_v4};

/// Every component name this build can read. Anything else ending in
/// `Component` is reported as dropped rather than silently lost on save.
pub(super) const KNOWN_COMPONENTS: &[&str] = &[
    "TagComponent",
    "TransformComponent",
    "CameraComponent",
    "MeshRendererComponent",
    "SpriteRendererComponent",
    "TextComponent",
    "SpriteAnimationComponent",
    "DirectionalLightComponent",
    "PointLightComponent",
    "SpotLightComponent",
    "RigidBody3DComponent",
    "BoxCollider3DComponent",
    "SphereCollider3DComponent",
    "CapsuleCollider3DComponent",
    "AudioSourceComponent",
    "AnimatorComponent",
    "RelationshipComponent",
];

pub(super) fn components(scene: &mut Scene, entity: Entity, node: &Value) {
    if let Some(node) = node.get("TransformComponent") {
        if let Some(mut view) = scene.get_mut(entity) {
            if let Some(transform) = view.get_mut::<Transform>() {
                if let Some(value) = node.get("Translation").and_then(read_v3) {
                    transform.translation = value;
                }
                if let Some(value) = node.get("Rotation").and_then(read_v3) {
                    transform.rotation = value;
                }
                if let Some(value) = node.get("Scale").and_then(read_v3) {
                    transform.scale = value;
                }
            }
        }
    } else {
        // `spawn` installs a Transform; a file that does not name one had the
        // component removed, so keeping the default would resurrect it.
        scene.remove::<Transform>(entity);
    }

    if node.get("TagComponent").is_none() {
        // `spawn` installs a Tag for the hierarchy's name; a file without one
        // must not gain a component it did not contain.
        scene.remove::<Tag>(entity);
    }

    if let Some(node) = node.get("CameraComponent") {
        let mut camera = SceneCamera::default();
        if let Some(value) = node.get("ProjectionType").and_then(Value::as_i64) {
            camera.set_projection_type(if value == 1 {
                ProjectionType::Orthographic
            } else {
                ProjectionType::Perspective
            });
        }
        if let Some(value) = node.get("PerspectiveFOV").and_then(as_f32) {
            camera.set_fov(value);
        }
        if let Some(value) = node.get("PerspectiveNear").and_then(as_f32) {
            camera.set_perspective_near(value);
        }
        if let Some(value) = node.get("PerspectiveFar").and_then(as_f32) {
            camera.set_perspective_far(value);
        }
        if let Some(value) = node.get("OrthographicSize").and_then(as_f32) {
            camera.set_orthographic_size(value);
        }
        if let Some(value) = node.get("OrthographicNear").and_then(as_f32) {
            camera.set_orthographic_near(value);
        }
        if let Some(value) = node.get("OrthographicFar").and_then(as_f32) {
            camera.set_orthographic_far(value);
        }
        let default = Camera::default();
        scene.insert(
            entity,
            Camera {
                camera,
                primary: node
                    .get("Primary")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.primary),
                fixed_aspect_ratio: node
                    .get("FixedAspectRatio")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.fixed_aspect_ratio),
            },
        );
    }

    if let Some(node) = node.get("MeshRendererComponent") {
        let default = MeshRenderer::default();
        scene.insert(
            entity,
            MeshRenderer {
                mesh: None,
                mesh_path: node
                    .get("MeshPath")
                    .and_then(read_string)
                    .unwrap_or(default.mesh_path),
                material: None,
                material_path: node
                    .get("MaterialPath")
                    .and_then(read_string)
                    .unwrap_or(default.material_path),
                tint: node.get("Tint").and_then(read_v4).unwrap_or(default.tint),
                visible: node
                    .get("Visible")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.visible),
                cast_shadows: node
                    .get("CastShadows")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.cast_shadows),
            },
        );
    }

    if let Some(node) = node.get("SpriteRendererComponent") {
        let default = SpriteRenderer::default();
        scene.insert(
            entity,
            SpriteRenderer {
                color: node.get("Color").and_then(read_v4).unwrap_or(default.color),
                texture: None,
                texture_path: node
                    .get("TexturePath")
                    .and_then(read_string)
                    .unwrap_or(default.texture_path),
                tiling_factor: node
                    .get("TilingFactor")
                    .and_then(as_f32)
                    .unwrap_or(default.tiling_factor),
                use_sub_texture: node
                    .get("UseSubTexture")
                    .and_then(Value::as_bool)
                    .unwrap_or(default.use_sub_texture),
                sub_texture_coords: node
                    .get("SubTextureCoords")
                    .and_then(read_v2)
                    .unwrap_or(default.sub_texture_coords),
                sub_texture_cell_size: node
                    .get("SubTextureCellSize")
                    .and_then(read_v2)
                    .unwrap_or(default.sub_texture_cell_size),
                sub_texture_sprite_size: node
                    .get("SubTextureSpriteSize")
                    .and_then(read_v2)
                    .unwrap_or(default.sub_texture_sprite_size),
            },
        );
    }

    if let Some(node) = node.get("TextComponent") {
        // `FontAssetPath` is the canonical key; `FontAsset` is accepted too
        // because the renderer contract named it that way.
        let font_path = node
            .get("FontAssetPath")
            .or_else(|| node.get("FontAsset"))
            .and_then(read_string);
        let default = Text::default();
        scene.insert(
            entity,
            Text {
                text: node
                    .get("TextString")
                    .and_then(read_string)
                    .unwrap_or(default.text),
                font: None,
                font_path: font_path.unwrap_or(default.font_path),
                color: node.get("Color").and_then(read_v4).unwrap_or(default.color),
                kerning: node
                    .get("Kerning")
                    .and_then(as_f32)
                    .unwrap_or(default.kerning),
                line_spacing: node
                    .get("LineSpacing")
                    .and_then(as_f32)
                    .unwrap_or(default.line_spacing),
            },
        );
    }

    super::read_physics::components(scene, entity, node);
}

/// Records the components a file carried that this build cannot read, including
/// a previous save's `DroppedComponents` note, so the loss is never silent.
pub(super) fn note_dropped(scene: &mut Scene, entity: Entity, node: &Value) {
    let Some(map) = node.as_mapping() else {
        return;
    };
    let present: HashSet<&str> = map
        .keys()
        .filter_map(Value::as_str)
        .filter(|name| name.ends_with("Component"))
        .collect();

    if let Some(previous) = node.get("DroppedComponents").and_then(Value::as_sequence) {
        for name in previous.iter().filter_map(Value::as_str) {
            // A component that is back on the entity is no longer dropped.
            if !present.contains(name) {
                scene.record_dropped_component(entity, name);
            }
        }
    }

    for &name in &present {
        if !KNOWN_COMPONENTS.contains(&name) {
            log::warn!("meats: skipping unknown component `{name}`; a save will record it");
            scene.record_dropped_component(entity, name);
        }
    }
}

pub(super) fn read_vec2(node: &Value, x: &str, y: &str, default: Vec2) -> Vec2 {
    Vec2::new(
        node.get(x).and_then(as_f32).unwrap_or(default.x),
        node.get(y).and_then(as_f32).unwrap_or(default.y),
    )
}
