//! The viewport's line geometry: the gizmo batch and the selection outline.
//!
//! `mirror` owns the gizmo's maths and hands back `(from, to, axis)` segments;
//! the colours and the box edges are this UI's convention, so they live here.

use bifrost::passes::LineVertex;
use glam::Vec3;
use mirror::gizmo::{GizmoMode, GizmoTransform};
use mirror::Axis;

use crate::app::EditorApp;

/// The gizmo as a line batch: two vertices per segment, one index pair each.
pub(crate) fn gizmo_batch(mode: GizmoMode, target: &GizmoTransform) -> (Vec<LineVertex>, Vec<u32>) {
    let segments = mirror::gizmo::segments(mode, target);
    let mut vertices = Vec::with_capacity(segments.len() * 2);
    let mut indices = Vec::with_capacity(segments.len() * 2);

    for (segment, (from, to, axis)) in segments.into_iter().enumerate() {
        let color = axis_color(axis);
        vertices.push(LineVertex {
            position: from.to_array(),
            color,
        });
        vertices.push(LineVertex {
            position: to.to_array(),
            color,
        });
        let first = (segment * 2) as u32;
        indices.push(first);
        indices.push(first + 1);
    }

    (vertices, indices)
}

/// The gizmo's axis colours: x red, y green, z blue, as every editor has them.
fn axis_color(axis: Axis) -> [f32; 4] {
    match axis {
        Axis::X => [0.90, 0.20, 0.25, 1.0],
        Axis::Y => [0.35, 0.85, 0.30, 1.0],
        Axis::Z => [0.30, 0.45, 0.95, 1.0],
    }
}

/// An axis-aligned box as a 12-edge line batch, for the selection outline.
pub(crate) fn box_batch(min: Vec3, max: Vec3, color: [f32; 4]) -> (Vec<LineVertex>, Vec<u32>) {
    let corner = |bits: usize| {
        Vec3::new(
            if bits & 1 == 0 { min.x } else { max.x },
            if bits & 2 == 0 { min.y } else { max.y },
            if bits & 4 == 0 { min.z } else { max.z },
        )
    };
    let vertices: Vec<LineVertex> = (0..8)
        .map(|bits| LineVertex {
            position: corner(bits).to_array(),
            color,
        })
        .collect();
    let edges = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let indices = edges
        .into_iter()
        .flat_map(|(a, b)| [a as u32, b as u32])
        .collect();
    (vertices, indices)
}

/// Appends a batch after `vertices`, rebasing its indices to match.
pub(crate) fn append(
    vertices: &mut Vec<LineVertex>,
    indices: &mut Vec<u32>,
    add_v: Vec<LineVertex>,
    add_i: Vec<u32>,
) {
    let base = vertices.len() as u32;
    vertices.extend(add_v);
    indices.extend(add_i.into_iter().map(|index| index + base));
}

/// The selected entity's world-space bounds, folded from every mesh primitive's
/// own box. `None` when the entity has no drawable mesh.
///
/// `mirror` folds bounds the same way for picking; this is the editor's copy so
/// the outline and the hit test agree on what a mesh occupies.
pub(crate) fn selection_bounds(app: &EditorApp, entity: meats::Entity) -> Option<(Vec3, Vec3)> {
    let engine = app.engine.engine();
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for sample in engine.scene.meshes() {
        if sample.entity != entity {
            continue;
        }
        let Some(primitives) = engine.assets.meshes(sample.mesh) else {
            continue;
        };
        for mesh in primitives {
            if mesh.positions.is_empty() {
                continue;
            }
            let min = Vec3::from_array(mesh.bounds_min);
            let max = Vec3::from_array(mesh.bounds_max);
            for bits in 0..8 {
                let local = Vec3::new(
                    if bits & 1 == 0 { min.x } else { max.x },
                    if bits & 2 == 0 { min.y } else { max.y },
                    if bits & 4 == 0 { min.z } else { max.z },
                );
                let world = sample.transform.transform_point3(local);
                bounds = Some(match bounds {
                    Some((lo, hi)) => (lo.min(world), hi.max(world)),
                    None => (world, world),
                });
            }
        }
    }
    bounds
}
