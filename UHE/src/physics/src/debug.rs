//! The debug view: wireframes for collider shapes, contacts and broad-phase
//! boxes, as line segments `bifrost`'s line pass draws.
//!
//! The data, not the draw. `bifrost` already has a line pass (the grid and the
//! gizmo) and owns its own vertex format; this crate must not depend on it, so
//! the view is a flat list of world-space segments a drawing user turns into
//! that format.
//!
//! Colours are RGB and the caller owns strategy (depth, blend, thickness): this
//! only says where the lines are.

use glam::{Quat, Vec3};

/// One world-space line segment with an RGB colour.
///
/// `bifrost`'s line vertex is per-end (`position`, `color`); a drawing owner
/// emits two of them and an index pair per `DebugLine`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DebugLine {
    pub from: Vec3,
    pub to: Vec3,
    pub color: Vec3,
}

/// The three debug layers, each a list of segments so a caller can toggle one
/// without rebuilding the others.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct PhysicsDebug {
    /// A wireframe per simulated collider shape.
    pub shapes: Vec<DebugLine>,
    /// A cross at each touching contact.
    pub contacts: Vec<DebugLine>,
    /// A wireframe per collider's world bounding box, the broad phase's cells.
    pub broad_phase: Vec<DebugLine>,
}

const SHAPE_COLOR: Vec3 = Vec3::new(0.2, 0.8, 1.0);
const CONTACT_COLOR: Vec3 = Vec3::new(1.0, 0.8, 0.1);
const BROAD_PHASE_COLOR: Vec3 = Vec3::new(0.2, 1.0, 0.3);
const CONTACT_CROSS: f32 = 0.08;

/// Appends the wireframe for a shape centered at `center`, rotated by `rotation`.
pub(crate) fn shape_lines(
    shape: crate::shape::ScaledShape,
    center: Vec3,
    rotation: Quat,
    out: &mut Vec<DebugLine>,
) {
    use crate::shape::ScaledShape;
    match shape {
        ScaledShape::Box(half) => box_lines(half, center, rotation, SHAPE_COLOR, out),
        ScaledShape::Sphere(radius) => circle_lines(radius, center, rotation, SHAPE_COLOR, out),
        ScaledShape::Capsule {
            radius,
            half_height,
        } => capsule_lines(radius, half_height, center, rotation, out),
    }
}

/// A cross at a contact point, so a contact that is not a corner of a shape is
/// still visible.
pub(crate) fn contact_lines(point: Vec3, out: &mut Vec<DebugLine>) {
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        out.push(DebugLine {
            from: point - axis * CONTACT_CROSS,
            to: point + axis * CONTACT_CROSS,
            color: CONTACT_COLOR,
        });
    }
}

/// The broad-phase cell for a collider: its axis-aligned world bounding box.
pub(crate) fn aabb_lines(min: Vec3, max: Vec3, out: &mut Vec<DebugLine>) {
    let center = (min + max) * 0.5;
    let half = (max - min) * 0.5;
    box_lines(half, center, Quat::IDENTITY, BROAD_PHASE_COLOR, out);
}

/// Twelve edges of an axis-aligned box, then rotated and moved into place.
fn box_lines(half: Vec3, center: Vec3, rotation: Quat, color: Vec3, out: &mut Vec<DebugLine>) {
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            let sign = |bit: u32| if i & (1 << bit) == 0 { -1.0 } else { 1.0 };
            center + rotation * (half * Vec3::new(sign(0), sign(1), sign(2)))
        })
        .collect();
    for i in 0..8u32 {
        // An edge joins corners that differ in exactly one axis bit.
        for bit in 0..3u32 {
            if i & (1 << bit) != 0 {
                continue;
            }
            out.push(DebugLine {
                from: corners[i as usize],
                to: corners[(i | (1 << bit)) as usize],
                color,
            });
        }
    }
}

/// Three great circles, the usual sphere wireframe.
fn circle_lines(radius: f32, center: Vec3, rotation: Quat, color: Vec3, out: &mut Vec<DebugLine>) {
    const SEGMENTS: u32 = 24;
    for axis in 0..3 {
        for i in 0..SEGMENTS {
            let a = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
            let b = std::f32::consts::TAU * (i + 1) as f32 / SEGMENTS as f32;
            out.push(DebugLine {
                from: center + rotation * ring(radius, axis, a),
                to: center + rotation * ring(radius, axis, b),
                color,
            });
        }
    }
}

fn ring(radius: f32, axis: u32, angle: f32) -> Vec3 {
    let (sin, cos) = angle.sin_cos();
    match axis {
        0 => Vec3::new(0.0, radius * cos, radius * sin),
        1 => Vec3::new(radius * cos, 0.0, radius * sin),
        _ => Vec3::new(radius * cos, radius * sin, 0.0),
    }
}

/// Two rings and the four side lines, which reads as a capsule from any angle.
fn capsule_lines(
    radius: f32,
    half_height: f32,
    center: Vec3,
    rotation: Quat,
    out: &mut Vec<DebugLine>,
) {
    const SEGMENTS: u32 = 24;
    for cap in [-1.0f32, 1.0] {
        let cap_center = center + rotation * (Vec3::Y * half_height * cap);
        for i in 0..SEGMENTS {
            let a = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
            let b = std::f32::consts::TAU * (i + 1) as f32 / SEGMENTS as f32;
            out.push(DebugLine {
                from: cap_center + rotation * ring(radius, 0, a),
                to: cap_center + rotation * ring(radius, 0, b),
                color: SHAPE_COLOR,
            });
        }
    }
    for (x, z) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        let side = Vec3::new(x * radius, 0.0, z * radius);
        out.push(DebugLine {
            from: center + rotation * (side - Vec3::Y * half_height),
            to: center + rotation * (side + Vec3::Y * half_height),
            color: SHAPE_COLOR,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_wireframe_has_twelve_edges() {
        let mut lines = Vec::new();
        box_lines(
            Vec3::ONE,
            Vec3::ZERO,
            Quat::IDENTITY,
            SHAPE_COLOR,
            &mut lines,
        );
        assert_eq!(lines.len(), 12);
    }

    #[test]
    fn a_sphere_wireframe_is_three_rings() {
        let mut lines = Vec::new();
        circle_lines(1.0, Vec3::ZERO, Quat::IDENTITY, SHAPE_COLOR, &mut lines);
        assert_eq!(lines.len(), 3 * 24);
    }

    #[test]
    fn a_contact_is_a_three_axis_cross() {
        let mut lines = Vec::new();
        contact_lines(Vec3::new(1.0, 2.0, 3.0), &mut lines);
        assert_eq!(lines.len(), 3);
        for line in &lines {
            assert!((line.from.distance(line.to) - 2.0 * CONTACT_CROSS).abs() < 1.0e-6);
        }
    }
}
