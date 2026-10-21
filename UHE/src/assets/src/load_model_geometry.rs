//! CPU-side geometry extraction from a glTF mesh primitive.
//!
//! Touches no device: the buffer handles stay null here, so this half stays
//! compilable into a headless harness.
//!
//! **Not done:** the meshoptimizer passes. Vertex-cache and vertex-fetch
//! reordering are pure permutations kept for device performance, vertex merging
//! deduplicates byte-identical vertices, and tangent generation synthesises a
//! frame where the file carries none. meshoptimizer is a C library with no Rust
//! equivalent in the manifest, and adding one is out of scope. Geometry is
//! therefore correct and *un*-optimised: `TANGENT` is read when the file
//! provides it and left empty when it does not, so a normal-mapped primitive
//! needs a frame from the file until a generator is wired in. See the note in
//! [`crate::load_model`] for the compressed-view half of the same cut.

use gltf::buffer::Data as BufferData;

use crate::load_model::view_bytes;
use crate::mesh::{MaterialData, MeshData};
use crate::report::LoadReport;

/// Extracts one primitive, or `None` when it cannot be drawn.
///
/// A primitive with no `POSITION` and one whose indices come out empty are both
/// dropped: an entry the renderer cannot draw is worse than no entry at all.
pub(crate) fn extract_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[BufferData],
    materials: &[MaterialData],
    name: &str,
    report: &mut LoadReport,
) -> Option<MeshData> {
    let reader = primitive.reader(|buffer| view_bytes(buffer, buffers));
    let positions: Vec<[f32; 3]> = match reader.read_positions() {
        Some(positions) => positions.collect(),
        None => {
            skip_primitive(report, name, "no POSITION attribute");
            return None;
        }
    };
    if positions.is_empty() {
        skip_primitive(report, name, "POSITION is empty");
        return None;
    }
    let vertex_count = positions.len();

    let mut mesh = MeshData {
        name: name.to_owned(),
        positions,
        ..Default::default()
    };

    let mut has_normals = false;
    if let Some(normals) = reader.read_normals() {
        let normals: Vec<[f32; 3]> = normals.collect();
        if normals.len() == vertex_count {
            mesh.normals = normals;
            has_normals = true;
        } else {
            report.warning(format!(
                "{name}: NORMAL count {} does not match POSITION count {vertex_count}; synthesising",
                normals.len()
            ));
        }
    }
    if !has_normals {
        mesh.normals = vec![[0.0, 1.0, 0.0]; vertex_count];
    }

    if let Some(uvs) = reader.read_tex_coords(0).map(|uvs| uvs.into_f32()) {
        let uvs: Vec<[f32; 2]> = uvs.collect();
        if uvs.len() == vertex_count {
            // The V flip lives in `uv`, with the transform that has to account
            // for it, so the two cannot disagree.
            mesh.uvs = uvs.into_iter().map(crate::uv::import_uv).collect();
        } else {
            report.warning(format!(
                "{name}: TEXCOORD_0 count {} does not match POSITION count {vertex_count}; ignoring",
                uvs.len()
            ));
        }
    }
    if mesh.uvs.is_empty() {
        mesh.uvs = vec![[0.0, 0.0]; vertex_count];
    }

    if let Some(colors) = reader.read_colors(0) {
        let colors: Vec<[f32; 4]> = colors.into_rgba_f32().collect();
        if colors.len() == vertex_count {
            mesh.colors = colors;
            mesh.has_vertex_color = true;
        } else {
            // A partial colour array would leave a gradient at the seam, so the
            // attribute is dropped rather than half-applied.
            report.warning(format!(
                "{name}: COLOR_0 count {} does not match POSITION count {vertex_count}; ignoring vertex colours",
                colors.len()
            ));
        }
    }

    if let Some(joints) = reader.read_joints(0) {
        let joints: Vec<[u16; 4]> = joints.into_u16().collect();
        if joints.len() == vertex_count {
            mesh.joints = joints;
        }
    }
    if let Some(weights) = reader.read_weights(0) {
        let weights: Vec<[f32; 4]> = weights.into_f32().collect();
        if weights.len() == vertex_count {
            mesh.weights = weights;
        }
    }

    if let Some(tangents) = reader.read_tangents() {
        let tangents: Vec<[f32; 4]> = tangents.collect();
        if tangents.len() == vertex_count {
            mesh.tangents = tangents;
        } else {
            report.warning(format!(
                "{name}: TANGENT count {} does not match POSITION count {vertex_count}; ignoring",
                tangents.len()
            ));
        }
    }

    mesh.indices = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        // Non-indexed geometry is legal glTF. Without synthesised indices the
        // draw is silently skipped.
        None => (0..vertex_count as u32).collect(),
    };

    // Out-of-range indices would read past the vertex arrays at draw time; a
    // trailing partial triangle would be read as garbage.
    mesh.indices
        .retain(|index| (*index as usize) < vertex_count);
    mesh.indices
        .truncate(mesh.indices.len() - mesh.indices.len() % 3);

    if !has_normals {
        synthesize_normals(&mut mesh);
    }

    // A named material may point past the end of the list; the clamp happens
    // here so the fault does not surface at draw time, far from the load.
    match primitive.material().index() {
        Some(index) if index < materials.len() => mesh.material = materials.get(index).cloned(),
        Some(index) => {
            let fallback = if materials.is_empty() {
                "using the default material".to_owned()
            } else {
                "using material 0".to_owned()
            };
            report.substitution(
                format!("material ({name})"),
                format!(
                    "primitive references material {index} but the file declares {}; {fallback}",
                    materials.len()
                ),
            );
            mesh.material = materials.first().cloned();
        }
        // A missing material means the spec default, not the file's first one.
        None => mesh.material = None,
    }

    compute_bounds(&mut mesh);

    if mesh.indices.is_empty() {
        skip_primitive(report, name, "no drawable indices");
        return None;
    }
    report.loaded.primitives += 1;
    Some(mesh)
}

fn skip_primitive(report: &mut LoadReport, name: &str, reason: &str) {
    report.skipped.primitives += 1;
    report.warning(format!("{name}: {reason}"));
}

/// Area-weighted vertex normals from the triangle geometry, for a primitive
/// that carries no `NORMAL`.
///
/// Without it an un-normalled model defaults to `(0, 1, 0)` on every vertex and
/// shades black on its side and bottom faces. The cross product is *not*
/// normalised per triangle on purpose: its length is twice the triangle's area,
/// which is what makes the sum area-weighted.
fn synthesize_normals(mesh: &mut MeshData) {
    mesh.normals.fill([0.0, 0.0, 0.0]);

    for triangle in mesh.indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let (Some(p0), Some(p1), Some(p2)) = (
            mesh.positions.get(i0).copied(),
            mesh.positions.get(i1).copied(),
            mesh.positions.get(i2).copied(),
        ) else {
            continue;
        };
        let edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let face = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];
        for index in [i0, i1, i2] {
            if let Some(normal) = mesh.normals.get_mut(index) {
                normal[0] += face[0];
                normal[1] += face[1];
                normal[2] += face[2];
            }
        }
    }

    for normal in &mut mesh.normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        *normal = if length > 1e-6 {
            [normal[0] / length, normal[1] / length, normal[2] / length]
        } else {
            // A degenerate triangle (or a vertex no triangle uses) has no
            // direction to take; up is the fallback.
            [0.0, 1.0, 0.0]
        };
    }
}

fn compute_bounds(mesh: &mut MeshData) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for position in &mesh.positions {
        min = [
            min[0].min(position[0]),
            min[1].min(position[1]),
            min[2].min(position[2]),
        ];
        max = [
            max[0].max(position[0]),
            max[1].max(position[1]),
            max[2].max(position[2]),
        ];
    }
    if min[0] <= max[0] {
        mesh.bounds_min = min;
        mesh.bounds_max = max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_are_area_weighted_and_normalised() {
        // One triangle in the XY plane: every normal must come out +Z.
        let mut mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0]; 3],
            indices: vec![0, 1, 2],
            ..Default::default()
        };
        synthesize_normals(&mut mesh);
        for normal in &mesh.normals {
            assert!((normal[0]).abs() < 1e-6);
            assert!((normal[1]).abs() < 1e-6);
            assert!((normal[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn a_degenerate_triangle_falls_back_to_up() {
        let mut mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 0.0]; 3],
            indices: vec![0, 1, 2],
            ..Default::default()
        };
        synthesize_normals(&mut mesh);
        assert_eq!(mesh.normals[0], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn bounds_span_the_positions() {
        let mut mesh = MeshData {
            positions: vec![[-1.0, 0.0, 2.0], [3.0, -4.0, 2.0]],
            ..Default::default()
        };
        compute_bounds(&mut mesh);
        assert_eq!(mesh.bounds_min, [-1.0, -4.0, 2.0]);
        assert_eq!(mesh.bounds_max, [3.0, 0.0, 2.0]);
    }
}
