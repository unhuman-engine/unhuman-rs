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
use crate::load_lights::{node_light, parse_lights};
use crate::load_material::load_materials;
use crate::load_model_animation::{parse_animations, parse_skins};
use crate::load_model_geometry::extract_primitive;
use crate::mesh::{MaterialData, MeshData, ModelData, ModelNode};

/// Extensions whose values reach this crate's output.
///
/// The list is the C++ loader's contract, minus `KHR_draco_mesh_compression`
/// (no decoder, deliberately absent) and with `EXT_meshopt_compression` guarded
/// separately below: its declared buffer views cannot be decoded here, so a file
/// that actually compresses with it is refused before this list is consulted.
/// An extension belongs here only if its values are actually shaded with,
/// because a parsed-but-ignored extension renders subtly wrong with no error
/// anywhere.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "KHR_materials_pbrSpecularGlossiness",
    "KHR_texture_transform",
    "KHR_materials_unlit",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_clearcoat",
    "KHR_materials_specular",
    "KHR_materials_sheen",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_iridescence",
    "KHR_materials_anisotropy",
    "KHR_materials_diffuse_transmission",
    "KHR_lights_punctual",
    "KHR_texture_basisu",
    "EXT_meshopt_compression",
];

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
pub(crate) fn load(path: &Path, bytes: &[u8]) -> Result<LoadedModel> {
    let parsed = match gltf::Gltf::from_slice(bytes) {
        Ok(parsed) => parsed,
        // A file that fails the crate's validation is retried unvalidated
        // rather than refused.
        Err(gltf::Error::Validation(_)) => gltf::Gltf::from_slice_without_validation(bytes)
            .map_err(|error| Error::Decode(format!("glTF: {error}")))?,
        Err(error) => return Err(Error::Decode(format!("glTF: {error}"))),
    };

    let gltf::Gltf { document, blob } = parsed;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let buffers = gltf::import_buffers(&document, Some(base), blob)
        .map_err(|error| Error::Decode(format!("glTF buffers: {error}")))?;

    // A compressed view cannot be handed to the accessor readers, which see the
    // buffer, not the view; the glTF crate has no hook to decode one. Refusing
    // when a view actually uses the extension beats handing geometry bytes to a
    // shader as if they were vertices.
    if uses_meshopt(&document) {
        return Err(Error::Unsupported(
            "EXT_meshopt_compression: the file's buffer views are meshopt-compressed and this \
             crate has no decoder, so the compressed bytes would decode as garbage geometry"
                .to_owned(),
        ));
    }

    let unsupported_extensions = collect_unsupported(&document);
    for name in &unsupported_extensions {
        log::warn!("glTF declares unimplemented extension '{name}' - shading may be wrong");
    }

    let materials = load_materials(&document, base, &buffers);
    let punctual_lights = parse_lights(&document);
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

    let model = ModelData {
        nodes: walk.model_nodes,
        roots,
        skins,
        punctual_lights,
        unsupported_extensions,
    };
    Ok(LoadedModel { meshes, model })
}

/// Whether any buffer view is `EXT_meshopt_compression`-compressed.
fn uses_meshopt(document: &gltf::Document) -> bool {
    document
        .views()
        .any(|view| view.extension_value("EXT_meshopt_compression").is_some())
}

/// The declared extensions this loader does not honour, each once.
///
/// Both lists are consulted: `extensionsRequired` is a hard requirement the file
/// cannot be correct without, and a file may require an extension it does not
/// list as used.
fn collect_unsupported(document: &gltf::Document) -> Vec<String> {
    let mut unsupported: Vec<String> = Vec::new();
    for name in document
        .extensions_used()
        .chain(document.extensions_required())
    {
        if !SUPPORTED_EXTENSIONS.contains(&name)
            && !unsupported.iter().any(|seen| seen.as_str() == name)
        {
            unsupported.push(name.to_owned());
        }
    }
    unsupported
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
                extract_primitive(&primitive, self.buffers, self.materials, &name)
            {
                indices.push(self.mesh_data.len() as u32);
                self.mesh_data.push(extracted);
            }
        }

        self.geometry.insert(mesh.index(), indices.clone());
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draco_is_not_in_the_supported_list() {
        assert!(!SUPPORTED_EXTENSIONS.contains(&"KHR_draco_mesh_compression"));
        assert!(SUPPORTED_EXTENSIONS.contains(&"KHR_materials_clearcoat"));
        assert!(SUPPORTED_EXTENSIONS.contains(&"KHR_texture_basisu"));
    }

    #[test]
    fn a_nameless_file_is_not_a_valid_document() {
        assert!(gltf::Gltf::from_slice(b"").is_err());
    }
}
