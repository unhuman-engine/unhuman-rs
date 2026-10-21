//! glTF/GLB model loading entry point and scene-graph walk.
//!
//! This file owns everything that needs the whole asset (extension reporting,
//! the scene walk, resource lifetime), while material assembly, primitive
//! extraction, lights and animation parsing live in their own modules.

use std::collections::HashMap;
use std::path::Path;

use glam::Mat4;
use gltf::buffer::Data as BufferData;

use crate::error::{Error, Result};
use crate::extensions;
use crate::load_lights::{node_light, parse_lights};
use crate::load_material::load_materials;
use crate::load_model_animation::{parse_animations, parse_skins};
use crate::load_model_geometry::extract_primitive;
use crate::mesh::{MaterialData, MeshData, ModelData, ModelNode};
use crate::report::{ExtensionClass, LoadReport};

pub(crate) struct LoadedModel {
    pub meshes: Vec<MeshData>,
    pub model: ModelData,
}

/// The bytes of one buffer view, for the accessor readers.
///
/// The returned slice borrows from `buffers`, not from the `Buffer` handle, so
/// the two lifetimes are independent: a reader needs the bytes to outlive its
/// own borrow of the document, and tying them together would refuse that.
pub(crate) fn view_bytes<'a, 'b>(
    buffer: gltf::Buffer<'a>,
    buffers: &'b [BufferData],
) -> Option<&'b [u8]> {
    buffers.get(buffer.index()).map(|data| data.0.as_slice())
}

/// The local transform of one node, `T * R * S` or the file's own matrix.
pub(crate) fn node_transform(node: &gltf::Node<'_>) -> Mat4 {
    let matrix = node.transform().matrix();
    Mat4::from_cols_array_2d(&matrix)
}

/// Loads a glTF or GLB file.
///
/// `path` is the physical file the bytes came from: a `.gltf` may reference
/// buffers and images relative to itself, and that directory is the only thing
/// that makes those references resolvable.
pub(crate) fn load(path: &Path, bytes: &[u8], report: &mut LoadReport) -> Result<LoadedModel> {
    let parsed = match gltf::Gltf::from_slice(bytes) {
        Ok(parsed) => parsed,
        // A file that fails the crate's validation is retried unvalidated
        // rather than refused.
        Err(gltf::Error::Validation(_)) => gltf::Gltf::from_slice_without_validation(bytes)
            .map_err(|error| Error::Decode(format!("glTF: {error}")))?,
        Err(error) => return Err(Error::Decode(format!("glTF: {error}"))),
    };

    let gltf::Gltf { document, blob } = parsed;

    // Extensions first: a required one this loader does not implement is an
    // error before any work, and a compression extension actually applied means
    // the buffers cannot be read at all.
    report.extensions = extensions::check(&document)?;
    let mut unsupported_extensions = Vec::new();
    // Collected first: recording a warning mutates the report the loop reads.
    let extensions = report.extensions.clone();
    for extension in &extensions {
        match extension.classification {
            ExtensionClass::Handled => {}
            ExtensionClass::Refused => {
                report.warning(format!(
                    "extension '{}' is refused by this build but the file does not apply it",
                    extension.name
                ));
            }
            ExtensionClass::Ignored => {
                report.warning(format!(
                    "extension '{}' is not implemented; ignoring it",
                    extension.name
                ));
                if !unsupported_extensions
                    .iter()
                    .any(|seen: &String| seen == &extension.name)
                {
                    unsupported_extensions.push(extension.name.clone());
                }
            }
        }
    }
    if let Some(name) = extensions::compression_in_use(&document) {
        let reason = match extensions::classify(name) {
            extensions::Classification::Refused(reason) => reason,
            _ => "this crate has no decoder",
        };
        return Err(Error::Unsupported(format!("{name}: {reason}")));
    }

    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let buffers = gltf::import_buffers(&document, Some(base), blob)
        .map_err(|error| Error::Decode(format!("glTF buffers: {error}")))?;

    let materials = load_materials(&document, base, &buffers, report)?;
    report.loaded.materials = materials.len();
    let punctual_lights = parse_lights(&document)?;
    let (skins, skeleton) = parse_skins(&document, &buffers);
    let animations = parse_animations(&document, &buffers);

    let nodes: Vec<gltf::Node> = document.nodes().collect();
    let mut walk = Walk {
        nodes,
        buffers: buffers.as_slice(),
        materials: materials.as_slice(),
        light_count: punctual_lights.len(),
        geometry: HashMap::new(),
        mesh_data: Vec::new(),
        model_nodes: Vec::new(),
        report,
    };

    let mut roots = Vec::new();
    // Fall back to scene 0 when the file names no default scene.
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    if let Some(scene) = scene {
        let root_indices: Vec<usize> = scene.nodes().map(|node| node.index()).collect();
        for index in root_indices {
            let slot = walk.walk(index, Mat4::IDENTITY);
            if slot != u32::MAX {
                roots.push(slot);
            }
        }
    }

    // A model's clips and skeleton belong to the model, but the contract puts
    // them on `MeshData`; the first mesh carries them and the rest stay empty.
    let mut meshes = walk.mesh_data;
    if let Some(first) = meshes.first_mut() {
        first.animations = animations;
        if !skins.is_empty() {
            first.skeleton = Some(skeleton);
        }
    }

    let report = walk.report.clone();
    let model = ModelData {
        nodes: walk.model_nodes,
        roots,
        skins,
        punctual_lights,
        unsupported_extensions,
        report,
    };
    Ok(LoadedModel { meshes, model })
}

/// The glTF scene walk: node records in depth-first order, geometry extracted
/// once per `glTF` mesh however many nodes reference it.
struct Walk<'a, 'b> {
    nodes: Vec<gltf::Node<'a>>,
    buffers: &'b [BufferData],
    materials: &'b [MaterialData],
    light_count: usize,
    /// `glTF` mesh index -> indices into `mesh_data`.
    geometry: HashMap<usize, Vec<u32>>,
    mesh_data: Vec<MeshData>,
    model_nodes: Vec<ModelNode>,
    report: &'b mut LoadReport,
}

impl<'a, 'b> Walk<'a, 'b> {
    /// Appends one node and its subtree, returning its index in
    /// `model_nodes`, or `u32::MAX` for a node index that does not exist.
    ///
    /// Parent and child links are set *after* recursing because a reference
    /// into `model_nodes` would dangle the moment a child was appended; keep
    /// the index, not the reference.
    fn walk(&mut self, node_index: usize, parent_transform: Mat4) -> u32 {
        let Some(node) = self.nodes.get(node_index).cloned() else {
            return u32::MAX;
        };

        let (translation, rotation, scale) = node.transform().decomposed();
        let world_transform = parent_transform * node_transform(&node);
        let name = node
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Node_{node_index}"));

        let (mesh, primitives) = match node.mesh() {
            Some(mesh) => (mesh.index() as i32, self.geometry_for(&mesh)),
            None => (-1, Vec::new()),
        };

        let slot = self.model_nodes.len() as u32;
        self.model_nodes.push(ModelNode {
            name,
            parent: -1,
            children: Vec::new(),
            mesh,
            primitives,
            translation,
            rotation,
            scale,
            world_transform: world_transform.to_cols_array_2d(),
            light: node_light(&node, self.light_count),
        });

        let children: Vec<usize> = node.children().map(|child| child.index()).collect();
        for child_index in children {
            let child_slot = self.walk(child_index, world_transform);
            if child_slot == u32::MAX {
                continue;
            }
            if let Some(child) = self.model_nodes.get_mut(child_slot as usize) {
                child.parent = slot as i32;
            }
            if let Some(parent) = self.model_nodes.get_mut(slot as usize) {
                parent.children.push(child_slot);
            }
        }

        slot
    }

    /// Extracts a `glTF` mesh at most once, however many nodes use it.
    ///
    /// This is what makes per-node transforms affordable: repeated props and
    /// foliage reference one set of vertex arrays instead of one each.
    fn geometry_for(&mut self, mesh: &gltf::Mesh<'a>) -> Vec<u32> {
        if let Some(cached) = self.geometry.get(&mesh.index()) {
            return cached.clone();
        }

        let name = mesh
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| "Unnamed_Mesh".to_owned());
        let mut indices = Vec::new();
        for primitive in mesh.primitives() {
            if let Some(extracted) =
                extract_primitive(&primitive, self.buffers, self.materials, &name, self.report)
            {
                indices.push(self.mesh_data.len() as u32);
                self.mesh_data.push(extracted);
            }
        }

        if indices.is_empty() {
            self.report.skipped.meshes += 1;
        } else {
            self.report.loaded.meshes += 1;
        }
        self.geometry.insert(mesh.index(), indices.clone());
        indices
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_nameless_file_is_not_a_valid_document() {
        assert!(gltf::Gltf::from_slice(b"").is_err());
    }
}
