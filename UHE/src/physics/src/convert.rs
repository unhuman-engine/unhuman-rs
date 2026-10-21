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
use rapier3d::prelude::{Collider, Pose};

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

/// A collider's world translation. It is already a glam `Vec3`, so this exists
/// only to keep the Rapier math type name inside this file.
pub(crate) fn collider_translation(collider: &Collider) -> Vec3 {
    collider.translation()
}

/// A collider's world-space bounding box, for the debug view's broad phase.
pub(crate) fn collider_aabb(collider: &Collider) -> (Vec3, Vec3) {
    let aabb = collider.compute_aabb();
    (aabb.mins, aabb.maxs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_translation_crosses_unchanged() {
        let translation = Vec3::new(1.0, -2.5, 3.25);
        let pose = rapier_pose(translation, Vec3::ZERO);
        let (back, rotation) = scene_pose(&pose);
        assert_eq!(back, translation);
        assert!(
            rotation.abs_diff_eq(Vec3::ZERO, 1.0e-5),
            "rotation {rotation:?}"
        );
    }

    /// The scene stores `Rz*Ry*Rx`; a conversion that reached for glam's
    /// `EulerRot::XYZ` (which is `Rx*Ry*Rz`) would transpose the order. A single
    /// axis isolates the sign and the component, which an order mix-up changes.
    #[test]
    fn a_single_axis_rotation_lands_on_its_own_axis() {
        let pose = rapier_pose(Vec3::ZERO, Vec3::new(0.0, 90.0, 0.0));
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!((pose.rotation.x).abs() < 1.0e-6);
        assert!((pose.rotation.y - half).abs() < 1.0e-6);
        assert!((pose.rotation.z).abs() < 1.0e-6);
        assert!((pose.rotation.w - half).abs() < 1.0e-6);
    }

    #[test]
    fn an_authored_rotation_round_trips_through_both_directions() {
        let authored = Vec3::new(30.0, 20.0, 10.0);
        let (_, back) = scene_pose(&rapier_pose(Vec3::new(4.0, 5.0, 6.0), authored));
        assert!(
            (back - authored).length() < 0.1,
            "{authored:?} came back as {back:?}"
        );
    }
}
