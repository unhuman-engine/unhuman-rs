//! Skin and animation parsing.
//!
//! Separate from the scene walk because it answers a different question: the
//! walk places geometry, this builds the deformation data a runtime applies on
//! top.

use glam::Mat4;
use gltf::animation::util::{ReadOutputs, Rotations, Translations};
use gltf::animation::Property;
use gltf::buffer::Data as BufferData;

use crate::anim::normalize;
use crate::load_model::{node_transform, view_bytes};
use crate::mesh::{
    AnimationData, BoneData, Interpolation, QuaternionTrack, SkeletonData, SkinData, VectorTrack,
};

/// Every skin in the file, plus the one joint hierarchy that spans all nodes.
///
/// The hierarchy is built from *every* node, not only the ones a skin lists,
/// because a bone chain's parents are ordinary nodes too; the first skin's
/// joints and inverse bind matrices are also copied onto the skeleton, so both
/// remain reachable without a skin lookup.
pub(crate) fn parse_skins(
    document: &gltf::Document,
    buffers: &[BufferData],
) -> (Vec<SkinData>, SkeletonData) {
    let mut skins = Vec::new();
    let mut skeleton = SkeletonData {
        root: -1,
        ..Default::default()
    };

    let skin_list: Vec<gltf::Skin> = document.skins().collect();
    if skin_list.is_empty() {
        return (skins, skeleton);
    }
    let nodes: Vec<gltf::Node> = document.nodes().collect();

    for (index, node) in nodes.iter().enumerate() {
        skeleton.bones.push(BoneData {
            name: node
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Bone_{index}")),
            id: index as i32,
            parent: -1,
            inverse_bind: Mat4::IDENTITY.to_cols_array_2d(),
            local: node_transform(node).to_cols_array_2d(),
        });
    }
    for (index, node) in nodes.iter().enumerate() {
        for child in node.children() {
            if let Some(bone) = skeleton.bones.get_mut(child.index()) {
                bone.parent = index as i32;
            }
        }
    }

    let first = &skin_list[0];
    skeleton.root = first
        .skeleton()
        .map(|node| node.index() as i32)
        .or_else(|| first.joints().next().map(|node| node.index() as i32))
        .unwrap_or(-1);

    for (index, skin) in skin_list.iter().enumerate() {
        let mut out = SkinData {
            name: skin
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Skin_{index}")),
            ..Default::default()
        };
        for joint in skin.joints() {
            out.joints.push(joint.index() as i32);
        }
        // A skin without inverse bind matrices poses as if every joint were at
        // the origin of the mesh; identity is the default.
        out.inverse_bind = vec![Mat4::IDENTITY.to_cols_array_2d(); out.joints.len()];

        if let Some(matrices) = skin
            .reader(|buffer| view_bytes(buffer, buffers))
            .read_inverse_bind_matrices()
        {
            for (slot, matrix) in matrices.enumerate() {
                if let Some(target) = out.inverse_bind.get_mut(slot) {
                    *target = matrix;
                }
            }
        }

        if index == 0 {
            skeleton.joints = out.joints.clone();
            for (slot, joint) in out.joints.iter().enumerate() {
                if *joint < 0 {
                    continue;
                }
                if let Some(bone) = skeleton.bones.get_mut(*joint as usize) {
                    if let Some(matrix) = out.inverse_bind.get(slot) {
                        bone.inverse_bind = *matrix;
                    }
                }
            }
        }

        skins.push(out);
    }

    (skins, skeleton)
}

/// Every animation clip in the file.
///
/// Morph-target weight channels are skipped, because nothing in this engine
/// blends them.
pub(crate) fn parse_animations(
    document: &gltf::Document,
    buffers: &[BufferData],
) -> Vec<AnimationData> {
    let mut animations = Vec::new();

    for (index, animation) in document.animations().enumerate() {
        let mut clip = AnimationData {
            name: animation
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Anim_{index}")),
            ..Default::default()
        };

        for channel in animation.channels() {
            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
            };
            let target = channel.target().node().index() as u32;
            let reader = channel.reader(|buffer| view_bytes(buffer, buffers));

            let Some(times) = reader
                .read_inputs()
                .map(|inputs| inputs.collect::<Vec<f32>>())
            else {
                continue;
            };
            if times.is_empty() {
                continue;
            }
            for time in &times {
                clip.duration = clip.duration.max(*time);
            }

            let Some(outputs) = reader.read_outputs() else {
                continue;
            };
            match channel.target().property() {
                Property::Translation => {
                    let ReadOutputs::Translations(values) = outputs else {
                        continue;
                    };
                    let track = find_or_add_vector(&mut clip.position_tracks, target);
                    track.interpolation = interpolation;
                    fill_vector(track, &times, values, interpolation);
                }
                Property::Rotation => {
                    let ReadOutputs::Rotations(values) = outputs else {
                        continue;
                    };
                    let track = find_or_add_quaternion(&mut clip.rotation_tracks, target);
                    track.interpolation = interpolation;
                    fill_quaternion(track, &times, values, interpolation);
                }
                Property::Scale => {
                    let ReadOutputs::Scales(values) = outputs else {
                        continue;
                    };
                    let track = find_or_add_vector(&mut clip.scale_tracks, target);
                    track.interpolation = interpolation;
                    fill_vector(track, &times, values, interpolation);
                }
                Property::MorphTargetWeights => continue,
            }
        }

        animations.push(clip);
    }

    animations
}

/// The track already targeting this node, or a new one.
///
/// glTF may emit several channels for one (node, path) pair, and merging them
/// is what keeps the sampled pose well-defined.
fn find_or_add_vector(tracks: &mut Vec<VectorTrack>, target: u32) -> &mut VectorTrack {
    let index = match tracks.iter().position(|track| track.target == target) {
        Some(index) => index,
        None => {
            tracks.push(VectorTrack {
                target,
                interpolation: Interpolation::Linear,
                ..Default::default()
            });
            tracks.len() - 1
        }
    };
    &mut tracks[index]
}

fn find_or_add_quaternion(tracks: &mut Vec<QuaternionTrack>, target: u32) -> &mut QuaternionTrack {
    let index = match tracks.iter().position(|track| track.target == target) {
        Some(index) => index,
        None => {
            tracks.push(QuaternionTrack {
                target,
                interpolation: Interpolation::Linear,
                ..Default::default()
            });
            tracks.len() - 1
        }
    };
    &mut tracks[index]
}

/// `CUBICSPLINE` output is `[inTangent_k, value_k, outTangent_k]` per key, so
/// one accessor element in three belongs in each array.
fn fill_vector(
    track: &mut VectorTrack,
    times: &[f32],
    values: Translations<'_>,
    interpolation: Interpolation,
) {
    track.times = times.to_vec();
    if interpolation == Interpolation::CubicSpline {
        let keys = times.len();
        track.values = vec![[0.0; 3]; keys];
        track.in_tangents = vec![[0.0; 3]; keys];
        track.out_tangents = vec![[0.0; 3]; keys];
        for (slot, value) in values.enumerate() {
            let key = slot / 3;
            if key >= keys {
                break;
            }
            match slot % 3 {
                0 => track.in_tangents[key] = value,
                1 => track.values[key] = value,
                _ => track.out_tangents[key] = value,
            }
        }
    } else {
        track.values = values.take(times.len()).collect();
        track.in_tangents.clear();
        track.out_tangents.clear();
    }
}

fn fill_quaternion(
    track: &mut QuaternionTrack,
    times: &[f32],
    values: Rotations<'_>,
    interpolation: Interpolation,
) {
    track.times = times.to_vec();
    let values = values.into_f32();
    if interpolation == Interpolation::CubicSpline {
        let keys = times.len();
        track.values = vec![[0.0, 0.0, 0.0, 1.0]; keys];
        track.in_tangents = vec![[0.0; 4]; keys];
        track.out_tangents = vec![[0.0; 4]; keys];
        for (slot, value) in values.enumerate() {
            let key = slot / 3;
            if key >= keys {
                break;
            }
            match slot % 3 {
                0 => track.in_tangents[key] = value,
                // Orientation keys must be unit quaternions, since everything
                // downstream multiplies them. glTF 2.0 spline *tangents* are
                // derivatives and must NOT be normalised.
                1 => track.values[key] = normalize(value),
                _ => track.out_tangents[key] = value,
            }
        }
    } else {
        track.values = values.take(times.len()).map(normalize).collect();
        track.in_tangents.clear();
        track.out_tangents.clear();
    }
}
