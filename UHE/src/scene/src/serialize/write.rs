//! Scene -> YAML entity mapping.

use std::collections::HashSet;

use yaml_serde::{Mapping, Value};

use crate::camera::{Camera, ProjectionType};
use crate::components::{
    Animator, AudioSource, Collider, ColliderShape, Light, LightKind, MeshRenderer, RigidBody,
    RigidBodyKind, SpriteAnimation, SpriteRenderer, Tag, Text, Transform,
};
use crate::entity::Entity;
use crate::scene::Scene;

use super::value::{f32_value, int, key, v2, v3, v4};

pub(super) fn tree(scene: &Scene) -> Vec<Value> {
    // Roots and their subtrees first, then a sweep for anything the hierarchy
    // did not reach, so no live entity can silently vanish from a save.
    let mut entities = Vec::new();
    let mut written = HashSet::new();
    for entity in scene.entities() {
        if scene.parent(entity).is_none() {
            write_subtree(scene, entity, &mut entities, &mut written);
        }
    }
    for entity in scene.entities() {
        if !written.contains(&entity) {
            write_subtree(scene, entity, &mut entities, &mut written);
        }
    }
    entities
}

fn write_subtree(
    scene: &Scene,
    entity: Entity,
    out: &mut Vec<Value>,
    written: &mut HashSet<Entity>,
) {
    // Exactly-once guard: an id reachable through two paths must serialize once.
    if !written.insert(entity) {
        return;
    }
    out.push(entity_value(scene, entity));
    for child in scene.children(entity).to_vec() {
        // A child entry is followed only when the child points back, so a
        // stale entry cannot serialize twice.
        if scene.parent(child) == Some(entity) {
            write_subtree(scene, child, out, written);
        }
    }
}

fn entity_value(scene: &Scene, entity: Entity) -> Value {
    let mut map = Mapping::new();
    map.insert(
        key("Entity"),
        Value::Number(scene.id_of(entity).unwrap_or(0).into()),
    );

    if let Some(tag) = component::<Tag>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("Tag"), Value::String(tag.tag.clone()));
        map.insert(key("TagComponent"), Value::Mapping(c));
    }

    if let Some(transform) = component::<Transform>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("Translation"), v3(transform.translation));
        c.insert(key("Rotation"), v3(transform.rotation));
        c.insert(key("Scale"), v3(transform.scale));
        map.insert(key("TransformComponent"), Value::Mapping(c));
    }

    if let Some(camera) = component::<Camera>(scene, entity) {
        let mut c = Mapping::new();
        let projection = match camera.camera.projection_type() {
            ProjectionType::Perspective => 0,
            ProjectionType::Orthographic => 1,
        };
        c.insert(key("ProjectionType"), int(projection));
        c.insert(key("PerspectiveFOV"), f32_value(camera.camera.fov()));
        c.insert(
            key("PerspectiveNear"),
            f32_value(camera.camera.perspective_near()),
        );
        c.insert(
            key("PerspectiveFar"),
            f32_value(camera.camera.perspective_far()),
        );
        c.insert(
            key("OrthographicSize"),
            f32_value(camera.camera.orthographic_size()),
        );
        c.insert(
            key("OrthographicNear"),
            f32_value(camera.camera.orthographic_near()),
        );
        c.insert(
            key("OrthographicFar"),
            f32_value(camera.camera.orthographic_far()),
        );
        c.insert(key("Primary"), Value::Bool(camera.primary));
        c.insert(
            key("FixedAspectRatio"),
            Value::Bool(camera.fixed_aspect_ratio),
        );
        map.insert(key("CameraComponent"), Value::Mapping(c));
    }

    if let Some(renderer) = component::<MeshRenderer>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("MeshPath"), Value::String(renderer.mesh_path.clone()));
        c.insert(
            key("MaterialPath"),
            Value::String(renderer.material_path.clone()),
        );
        c.insert(key("Tint"), v4(renderer.tint));
        c.insert(key("Visible"), Value::Bool(renderer.visible));
        c.insert(key("CastShadows"), Value::Bool(renderer.cast_shadows));
        map.insert(key("MeshRendererComponent"), Value::Mapping(c));
    }

    if let Some(sprite) = component::<SpriteRenderer>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("Color"), v4(sprite.color));
        c.insert(
            key("TexturePath"),
            Value::String(sprite.texture_path.clone()),
        );
        c.insert(key("TilingFactor"), f32_value(sprite.tiling_factor));
        c.insert(key("UseSubTexture"), Value::Bool(sprite.use_sub_texture));
        c.insert(key("SubTextureCoords"), v2(sprite.sub_texture_coords));
        c.insert(key("SubTextureCellSize"), v2(sprite.sub_texture_cell_size));
        c.insert(
            key("SubTextureSpriteSize"),
            v2(sprite.sub_texture_sprite_size),
        );
        map.insert(key("SpriteRendererComponent"), Value::Mapping(c));
    }

    if let Some(text) = component::<Text>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("TextString"), Value::String(text.text.clone()));
        c.insert(key("FontAssetPath"), Value::String(text.font_path.clone()));
        c.insert(key("Color"), v4(text.color));
        c.insert(key("Kerning"), f32_value(text.kerning));
        c.insert(key("LineSpacing"), f32_value(text.line_spacing));
        map.insert(key("TextComponent"), Value::Mapping(c));
    }

    if let Some(light) = component::<Light>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("Color"), v3(light.color));
        c.insert(key("Intensity"), f32_value(light.intensity));
        c.insert(key("Radius"), f32_value(light.radius));
        // Radians in the file, degrees in the component, for KHR_lights_punctual
        // parity.
        c.insert(
            key("InnerConeAngle"),
            f32_value(light.inner_cone_angle.to_radians()),
        );
        c.insert(
            key("OuterConeAngle"),
            f32_value(light.outer_cone_angle.to_radians()),
        );
        c.insert(key("Enabled"), Value::Bool(light.enabled));
        let name = match light.kind {
            LightKind::Directional => "DirectionalLightComponent",
            LightKind::Point => "PointLightComponent",
            LightKind::Spot => "SpotLightComponent",
        };
        map.insert(key(name), Value::Mapping(c));
    }

    if let Some(body) = component::<RigidBody>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(
            key("BodyType"),
            int(match body.kind {
                RigidBodyKind::Static => 0,
                RigidBodyKind::Kinematic => 1,
                RigidBodyKind::Dynamic => 2,
            }),
        );
        c.insert(key("Mass"), f32_value(body.mass));
        c.insert(key("LinearDamping"), f32_value(body.damping));
        c.insert(key("AngularDamping"), f32_value(body.damping));
        c.insert(key("IsSensor"), Value::Bool(false));
        c.insert(
            key("AllowedDOFs"),
            Value::Sequence(vec![Value::Bool(true); 6]),
        );
        c.insert(key("Velocity"), v3(body.velocity));
        c.insert(key("GravityScale"), f32_value(body.gravity_scale));
        map.insert(key("RigidBody3DComponent"), Value::Mapping(c));
    }

    if let Some(collider) = component::<Collider>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("Offset"), v3(collider.offset));
        c.insert(key("Friction"), f32_value(collider.friction));
        c.insert(key("Restitution"), f32_value(collider.restitution));
        let name = match collider.shape {
            ColliderShape::Box { half_extent } => {
                c.insert(key("HalfExtent"), v3(half_extent));
                "BoxCollider3DComponent"
            }
            ColliderShape::Sphere { radius } => {
                c.insert(key("Radius"), f32_value(radius));
                "SphereCollider3DComponent"
            }
            ColliderShape::Capsule {
                radius,
                half_height,
            } => {
                c.insert(key("Radius"), f32_value(radius));
                c.insert(key("HalfHeight"), f32_value(half_height));
                "CapsuleCollider3DComponent"
            }
        };
        map.insert(key(name), Value::Mapping(c));
    }

    if let Some(audio) = component::<AudioSource>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(key("ClipPath"), Value::String(audio.clip_path.clone()));
        c.insert(key("Playing"), Value::Bool(audio.playing));
        c.insert(key("Looping"), Value::Bool(audio.looping));
        c.insert(key("Gain"), f32_value(audio.gain));
        c.insert(key("Pitch"), f32_value(audio.pitch));
        c.insert(key("Spatial"), Value::Bool(audio.spatial));
        c.insert(key("Range"), f32_value(audio.range));
        map.insert(key("AudioSourceComponent"), Value::Mapping(c));
    }

    if let Some(animator) = component::<Animator>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(
            key("CurrentAnimationName"),
            Value::String(animator.clip_path.clone()),
        );
        c.insert(key("PlaybackSpeed"), f32_value(animator.speed));
        c.insert(key("IsPlaying"), Value::Bool(animator.playing));
        // LoopMode: None = 0, Loop = 1, PingPong = 2.
        c.insert(key("LoopMode"), int(if animator.looping { 1 } else { 0 }));
        c.insert(key("Time"), f32_value(animator.time));
        map.insert(key("AnimatorComponent"), Value::Mapping(c));
    }

    if let Some(animation) = component::<SpriteAnimation>(scene, entity) {
        let mut c = Mapping::new();
        c.insert(
            key("SpriteSheetPath"),
            Value::String(animation.sprite_sheet_path.clone()),
        );
        c.insert(key("FrameSizeX"), f32_value(animation.frame_size.x));
        c.insert(key("FrameSizeY"), f32_value(animation.frame_size.y));
        c.insert(
            key("FrameCount"),
            Value::Number(u64::from(animation.frame_count).into()),
        );
        c.insert(key("FrameDuration"), f32_value(animation.frame_duration));
        c.insert(key("Loop"), Value::Bool(animation.looping));
        c.insert(key("Color"), v4(animation.color));
        c.insert(key("Time"), f32_value(animation.time));
        map.insert(key("SpriteAnimationComponent"), Value::Mapping(c));
    }

    let dropped = scene.dropped_components(entity);
    if !dropped.is_empty() {
        map.insert(
            key("DroppedComponents"),
            Value::Sequence(
                dropped
                    .iter()
                    .map(|name| Value::String(name.clone()))
                    .collect(),
            ),
        );
    }

    let mut relationship = Mapping::new();
    relationship.insert(
        key("Parent"),
        Value::Number(
            scene
                .parent(entity)
                .and_then(|p| scene.id_of(p))
                .unwrap_or(0)
                .into(),
        ),
    );
    relationship.insert(
        key("Children"),
        Value::Sequence(
            scene
                .children(entity)
                .iter()
                .map(|child| Value::Number(scene.id_of(*child).unwrap_or(0).into()))
                .collect(),
        ),
    );
    map.insert(key("RelationshipComponent"), Value::Mapping(relationship));

    Value::Mapping(map)
}

fn component<'a, T: hecs::Component>(scene: &'a Scene, entity: Entity) -> Option<hecs::Ref<'a, T>> {
    scene.get(entity)?.get::<T>()
}
