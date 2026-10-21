//! The one file that names `meats` types and builds components.
//!
//! Kept apart from the rules so that the ECS shape (component names, field names, the
//! insert call) is in one place instead of spread through the game logic.

use glam::{Mat4, Vec3};
use meats::{
    Animator, AudioSource, Camera, Entity, Light, LightKind, MeshRenderer, Scene, SceneCamera,
    Transform,
};

/// Spawns an entity with a translation and a scale.
pub(crate) fn spawn(scene: &mut Scene, name: &str, translation: Vec3, scale: Vec3) -> Entity {
    let entity = scene.spawn(name);
    set_translation(scene, entity, translation);
    set_scale(scene, entity, scale);
    entity
}

pub(crate) fn set_translation(scene: &mut Scene, entity: Entity, translation: Vec3) {
    if let Some(mut entry) = scene.get_mut(entity) {
        if let Some(transform) = entry.get_mut::<Transform>() {
            transform.translation = translation;
        }
    }
}

fn set_scale(scene: &mut Scene, entity: Entity, scale: Vec3) {
    if let Some(mut entry) = scene.get_mut(entity) {
        if let Some(transform) = entry.get_mut::<Transform>() {
            transform.scale = scale;
        }
    }
}

pub(crate) fn set_rotation(scene: &mut Scene, entity: Entity, rotation: Vec3) {
    if let Some(mut entry) = scene.get_mut(entity) {
        if let Some(transform) = entry.get_mut::<Transform>() {
            transform.rotation = rotation;
        }
    }
}

/// Yaw then pitch, in degrees, onto the primary camera's rotation.
pub(crate) fn aim(scene: &mut Scene, yaw_degrees: f32, pitch_degrees: f32) {
    let Some(camera) = scene.primary_camera() else {
        return;
    };
    if let Some(mut entry) = scene.get_mut(camera) {
        if let Some(transform) = entry.get_mut::<Transform>() {
            transform.rotation = Vec3::new(pitch_degrees, yaw_degrees, 0.0);
        }
    }
}

/// Builds a perspective camera, with `fov_degrees` converted to radians.
pub(crate) fn add_primary_camera(
    scene: &mut Scene,
    entity: Entity,
    fov_degrees: f32,
    near: f32,
    far: f32,
) {
    let camera = Camera {
        camera: SceneCamera::new_perspective(fov_degrees.to_radians(), near, far),
        primary: true,
        ..Default::default()
    };
    let _ = scene.insert(entity, camera);
}

/// Adds a white directional light at full intensity.
pub(crate) fn add_directional_light(scene: &mut Scene, entity: Entity) {
    let light = Light {
        kind: LightKind::Directional,
        ..Default::default()
    };
    let _ = scene.insert(entity, light);
}

/// A mesh by virtual path.
pub(crate) fn add_mesh(scene: &mut Scene, entity: Entity, path: &str) {
    let mesh = MeshRenderer {
        mesh_path: path.to_owned(),
        ..Default::default()
    };
    let _ = scene.insert(entity, mesh);
}

/// Marks a skinned mesh entity as playing. `bifrost` attaches the model's
/// skeleton and chooses Idle, or the first clip, when this component appears.
pub(crate) fn add_animator(scene: &mut Scene, entity: Entity) {
    let animator = Animator {
        playing: true,
        looping: true,
        ..Default::default()
    };
    let _ = scene.insert(entity, animator);
}

/// Selects a clip by name and its loop mode; a missing animator is a no-op.
pub(crate) fn play_clip(scene: &mut Scene, entity: Entity, clip: &str, looping: bool) {
    if let Some(mut entry) = scene.get_mut(entity) {
        if let Some(animator) = entry.get_mut::<Animator>() {
            if animator.clip_path != clip {
                animator.clip_path.clear();
                animator.clip_path.push_str(clip);
                animator.time = 0.0;
            }
            animator.looping = looping;
            animator.playing = true;
        }
    }
}

/// A sound on an entity by virtual path. The handle beside it is
/// `Scene::resolve_assets`' to fill.
pub(crate) fn add_audio_source(scene: &mut Scene, entity: Entity, path: &str) {
    let source = AudioSource {
        clip_path: path.to_owned(),
        ..Default::default()
    };
    let _ = scene.insert(entity, source);
}

/// The clip `resolve_assets` filled on this entity's sound component, and the
/// gain beside it. `None` before resolution, or when the file is missing.
pub(crate) fn audio(scene: &Scene, entity: Entity) -> Option<(vault::AssetId, f32)> {
    let entry = scene.get(entity)?;
    let source = entry.get::<AudioSource>()?;
    Some((source.clip?, source.gain))
}

/// World position of an entity.
pub(crate) fn translation(scene: &Scene, entity: Entity) -> Vec3 {
    scene.world_transform(entity).w_axis.truncate()
}

/// Where the primary camera is and which way it looks.
///
/// The camera looks down its own -Z, the convention `meats` uses.
pub(crate) fn camera_pose(scene: &Scene) -> Option<(Vec3, Vec3)> {
    let camera = scene.primary_camera()?;
    let transform: Mat4 = scene.world_transform(camera);
    let position = transform.w_axis.truncate();
    let forward = -transform.z_axis.truncate().normalize_or_zero();
    Some((position, forward))
}
