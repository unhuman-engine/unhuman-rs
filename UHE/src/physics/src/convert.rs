//! The glam to Rapier boundary, and back.
//!
//! This Rapier re-exports glam through `glamx`: its `Vector` is `glam::Vec3` and
//! its `Rotation` is `glam::Quat`, so vectors, axes and collision groups cross
//! with no conversion. The one type that is not glam-native is `Pose`, a
//! translation-and-rotation pair; this is the only module that names it, so no
//! other file in the crate names a Rapier math type.
//!
//! Both directions go through `meats`' own conversion. `Transform::rotation_quat`
//! is the `Rz*Ry*Rx` order the scene stores, not glam's `EulerRot::XYZ` which is
//! `Rx*Ry*Rz`, and `meats::euler_degrees` is its inverse.

use glam::Vec3;
use meats::Transform;
use rapier3d::prelude::Pose;

/// A scene pose (translation, Euler degrees) as Rapier's `Pose`.
pub(crate) fn rapier_pose(translation: Vec3, rotation_degrees: Vec3) -> Pose {
    let transform = Transform {
        translation: Vec3::ZERO,
        rotation: rotation_degrees,
        scale: Vec3::ONE,
    };
    Pose::from_parts(translation, transform.rotation_quat())
}

/// Rapier's pose in the scene's own representation.
pub(crate) fn scene_pose(pose: &Pose) -> (Vec3, Vec3) {
    (pose.translation, meats::euler_degrees(pose.rotation))
}
