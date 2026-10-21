//! Converts a loaded model's skeleton, skins and clips into runtime animation data.
//!
//! `vault` stores the asset as parsed, with parallel arrays and glTF node ids;
//! this is the one place that shape becomes the hierarchy and keyframe tables
//! `Animator` plays.

use glam::{Mat4, Quat, Vec3};

use super::clip::{
    Bone, Clip, Interpolation, LoopMode, QuaternionTrack, Skeleton, Skin, VectorTrack,
};
use super::Animator;

fn interpolation(value: vault::Interpolation) -> Interpolation {
    match value {
        vault::Interpolation::Step => Interpolation::Step,
        vault::Interpolation::Linear => Interpolation::Linear,
        vault::Interpolation::CubicSpline => Interpolation::CubicSpline,
    }
}

fn skeleton_from(data: &vault::SkeletonData) -> Skeleton {
    Skeleton {
        bones: data
            .bones
            .iter()
            .map(|bone| Bone {
                name: bone.name.clone(),
                id: bone.id,
                parent_id: bone.parent,
                inverse_bind_matrix: Mat4::from_cols_array_2d(&bone.inverse_bind),
                local_transform: Mat4::from_cols_array_2d(&bone.local),
            })
            .collect(),
        joint_nodes: data.joints.clone(),
        root_bone_id: data.root,
    }
}

fn skin_from(data: &vault::SkinData) -> Skin {
    Skin {
        name: data.name.clone(),
        joint_nodes: data.joints.clone(),
        inverse_bind_matrices: data
            .inverse_bind
            .iter()
            .map(Mat4::from_cols_array_2d)
            .collect(),
    }
}

fn clip_from(data: &vault::AnimationData) -> Clip {
    let mut clip = Clip::new(data.name.clone());
    clip.duration = data.duration;
    clip.loop_mode = LoopMode::Loop;

    for track in &data.position_tracks {
        let mut out = VectorTrack::new(track.target as i32);
        out.interpolation = interpolation(track.interpolation);
        out.set_keys(
            &track.times,
            &track
                .values
                .iter()
                .map(|value| Vec3::from_array(*value))
                .collect::<Vec<_>>(),
        );
        if track.interpolation == vault::Interpolation::CubicSpline {
            out.in_tangents = track
                .in_tangents
                .iter()
                .map(|value| Vec3::from_array(*value))
                .collect();
            out.out_tangents = track
                .out_tangents
                .iter()
                .map(|value| Vec3::from_array(*value))
                .collect();
        }
        clip.position_tracks.push(out);
    }

    for track in &data.rotation_tracks {
        let mut out = QuaternionTrack::new(track.target as i32);
        out.interpolation = interpolation(track.interpolation);
        out.set_keys(
            &track.times,
            &track
                .values
                .iter()
                .map(|value| Quat::from_array(*value))
                .collect::<Vec<_>>(),
        );
        if track.interpolation == vault::Interpolation::CubicSpline {
            out.in_tangents = track
                .in_tangents
                .iter()
                .map(|value| Quat::from_array(*value))
                .collect();
            out.out_tangents = track
                .out_tangents
                .iter()
                .map(|value| Quat::from_array(*value))
                .collect();
        }
        clip.rotation_tracks.push(out);
    }

    for track in &data.scale_tracks {
        let mut out = VectorTrack::new(track.target as i32);
        out.interpolation = interpolation(track.interpolation);
        out.set_keys(
            &track.times,
            &track
                .values
                .iter()
                .map(|value| Vec3::from_array(*value))
                .collect::<Vec<_>>(),
        );
        if track.interpolation == vault::Interpolation::CubicSpline {
            out.in_tangents = track
                .in_tangents
                .iter()
                .map(|value| Vec3::from_array(*value))
                .collect();
            out.out_tangents = track
                .out_tangents
                .iter()
                .map(|value| Vec3::from_array(*value))
                .collect();
        }
        clip.scale_tracks.push(out);
    }

    clip
}

/// Builds an animator from the first mesh that carries a skeleton.
///
/// A model with no skeleton is not animated: `None`, not an empty animator that
/// would then draw every mesh through a one-bone identity palette.
pub fn animator_from_model(
    meshes: &[vault::MeshData],
    model: &vault::ModelData,
) -> Option<Animator> {
    let data = meshes.iter().find_map(|mesh| mesh.skeleton.as_ref())?;
    let mut animator = Animator::new(skeleton_from(data));
    let clips: Vec<Clip> = meshes
        .iter()
        .find(|mesh| !mesh.animations.is_empty())
        .map(|mesh| mesh.animations.iter().map(clip_from).collect())
        .unwrap_or_default();
    let skins: Vec<Skin> = model.skins.iter().map(skin_from).collect();
    animator.set_clips(clips);
    animator.set_skins(skins);
    Some(animator)
}

/// The clip a name asks for, or Idle, or the first one.
///
/// An exact name wins so a caller can select Fire or Reload; the fallback keeps
/// a model with differently named clips animating instead of standing still.
pub fn select_clip(clips: &[Clip], hint: Option<&str>) -> Option<usize> {
    if clips.is_empty() {
        return None;
    }
    if let Some(hint) = hint {
        if let Some(index) = clips.iter().position(|clip| clip.name == hint) {
            return Some(index);
        }
    }
    clips
        .iter()
        .position(|clip| clip.name.eq_ignore_ascii_case("idle"))
        .or(Some(0))
}
