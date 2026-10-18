//! Meshes on the GPU, and the cache that owns them.
//!
//! One vertex buffer and one index buffer per mesh, uploaded once and shared by
//! every entity that instances it.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use chassis::{
    Buffer, BufferDesc, BufferUsage, IndexFormat, Resources, TextureFormat, VertexAttribute,
    VertexLayout,
};
use vault::AssetId;

/// One vertex, interleaved. Position, normal, UV, tangent and colour.
///
/// `JOINTS_0`/`WEIGHTS_0` are not here yet because nothing skins: the palette
/// comes from `bifrost::animation`, whose clip-to-`Animator` glue from a loaded
/// model is not in place, so an attribute nothing reads would be stride paid for
/// by every vertex in every mesh. Adding the two is a change to this struct, the
/// layout below and the shader; `MeshKey` keys on the layout, so the mesh cache
/// re-uploads by itself when it happens.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// `xyz` tangent, `w` handedness, as glTF's TANGENT accessor stores it.
    pub tangent: [f32; 4],
    /// `COLOR_0`, white when the mesh has none. The shader multiplies it into the
    /// base colour unconditionally, so an absent `COLOR_0` needs no flag.
    pub color: [f32; 4],
}

/// Attribute locations, which the shader's `@location` bindings must match.
pub const LOCATION_POSITION: u32 = 0;
pub const LOCATION_NORMAL: u32 = 1;
pub const LOCATION_UV: u32 = 2;
pub const LOCATION_TANGENT: u32 = 3;
pub const LOCATION_COLOR: u32 = 6;

/// The one vertex layout every mesh is uploaded with.
pub fn vertex_layout() -> VertexLayout {
    VertexLayout {
        stride: std::mem::size_of::<Vertex>() as u32,
        attributes: vec![
            VertexAttribute {
                location: LOCATION_POSITION,
                format: TextureFormat::R32G32B32Float,
                offset: std::mem::offset_of!(Vertex, position) as u32,
            },
            VertexAttribute {
                location: LOCATION_NORMAL,
                format: TextureFormat::R32G32B32Float,
                offset: std::mem::offset_of!(Vertex, normal) as u32,
            },
            VertexAttribute {
                location: LOCATION_UV,
                format: TextureFormat::R32G32Float,
                offset: std::mem::offset_of!(Vertex, uv) as u32,
            },
            VertexAttribute {
                location: LOCATION_TANGENT,
                format: TextureFormat::Rgba32Float,
                offset: std::mem::offset_of!(Vertex, tangent) as u32,
            },
            // Location 6, matching the shader's colour attribute.
            VertexAttribute {
                location: LOCATION_COLOR,
                format: TextureFormat::Rgba32Float,
                offset: std::mem::offset_of!(Vertex, color) as u32,
            },
        ],
    }
}

/// The layout the depth prepass draws with: position alone.
///
/// The prepass shares the mesh's vertex buffer, so the stride and the position
/// offset are the full vertex's, but the other attributes are not declared:
/// `depth.wgsl` reads position and nothing else, and a declared attribute the
/// shader never consumes is still fetched and warned about.
pub fn depth_vertex_layout() -> VertexLayout {
    VertexLayout {
        stride: std::mem::size_of::<Vertex>() as u32,
        attributes: vec![VertexAttribute {
            location: LOCATION_POSITION,
            format: TextureFormat::R32G32B32Float,
            offset: std::mem::offset_of!(Vertex, position) as u32,
        }],
    }
}

/// A layout, as a hashable cache key.
///
/// `VertexLayout` is not `Hash`, and a cache key has to be: the layout a mesh was
/// uploaded with is what makes it reusable, so two meshes with the same layout
/// share one buffer, and two with different ones never do.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VertexLayoutKey(u64);

impl VertexLayoutKey {
    pub fn of(layout: &VertexLayout) -> Self {
        let mut hasher = DefaultHasher::new();
        layout.stride.hash(&mut hasher);
        for attribute in &layout.attributes {
            attribute.location.hash(&mut hasher);
            attribute.format.hash(&mut hasher);
            attribute.offset.hash(&mut hasher);
        }
        Self(hasher.finish())
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

/// What a mesh is cached under: which asset, interleaved how.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshKey {
    pub asset: AssetId,
    pub layout: VertexLayoutKey,
}

impl MeshKey {
    pub fn new(asset: AssetId, layout: VertexLayoutKey) -> Self {
        Self { asset, layout }
    }
}

/// A mesh resident on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMesh {
    pub vertex: Buffer,
    pub index: Buffer,
    pub index_count: u32,
    pub vertex_count: u32,
}

/// Interleaves the asset's parallel attribute arrays into vertices.
///
/// Missing attributes take defaults rather than being dropped: a normal of +Z and
/// a tangent of +X/right-handed, which makes a mesh with no normals shade rather
/// than vanish.
pub fn interleave(data: &vault::MeshData) -> Vec<Vertex> {
    (0..data.positions.len())
        .map(|i| Vertex {
            position: data.positions[i],
            normal: data.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
            uv: data.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            tangent: data
                .tangents
                .get(i)
                .copied()
                .unwrap_or([1.0, 0.0, 0.0, 1.0]),
            // White: a zero default would tint every mesh black and a
            // gamma-decoded one would tint every mesh wrongly.
            color: data.colors.get(i).copied().unwrap_or([1.0; 4]),
        })
        .collect()
}

/// Uploaded meshes, keyed by asset and layout.
pub struct MeshCache {
    entries: HashMap<MeshKey, GpuMesh>,
}

impl MeshCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn get(&self, key: &MeshKey) -> Option<GpuMesh> {
        self.entries.get(key).copied()
    }

    /// The resident mesh, or an upload of it.
    ///
    /// The second call for a key returns the same handles as the first: geometry
    /// is uploaded once per glTF mesh, not once per node that references it.
    pub fn get_or_upload(
        &mut self,
        resources: &mut dyn Resources,
        key: MeshKey,
        data: &vault::MeshData,
    ) -> Result<GpuMesh, chassis::Error> {
        if let Some(mesh) = self.entries.get(&key) {
            return Ok(*mesh);
        }

        let vertices = interleave(data);
        let index_count = data.indices.len() as u32;

        let vertex = resources.create_buffer(
            BufferDesc::new(
                std::mem::size_of_val(vertices.as_slice()) as u64,
                BufferUsage::VERTEX,
            ),
            Some(bytemuck::cast_slice(&vertices)),
        )?;

        let index = resources.create_buffer(
            BufferDesc::new(
                std::mem::size_of_val(data.indices.as_slice()) as u64,
                BufferUsage::INDEX,
            ),
            Some(bytemuck::cast_slice(&data.indices)),
        )?;

        let mesh = GpuMesh {
            vertex,
            index,
            index_count,
            vertex_count: vertices.len() as u32,
        };
        self.entries.insert(key, mesh);
        Ok(mesh)
    }

    /// Drops one mesh through the RHI. True when there was one.
    pub fn destroy(&mut self, resources: &mut dyn Resources, key: &MeshKey) -> bool {
        let Some(mesh) = self.entries.remove(key) else {
            return false;
        };
        resources.destroy_buffer(mesh.vertex);
        resources.destroy_buffer(mesh.index);
        true
    }

    pub fn clear(&mut self, resources: &mut dyn Resources) {
        for (_, mesh) in self.entries.drain() {
            resources.destroy_buffer(mesh.vertex);
            resources.destroy_buffer(mesh.index);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for MeshCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The index format every uploaded mesh uses: `MeshData` indices are `u32`.
pub const INDEX_FORMAT: IndexFormat = IndexFormat::U32;
