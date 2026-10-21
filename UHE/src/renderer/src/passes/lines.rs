//! The editor's grid, and gizmo lines.
//!
//! The grid is six vertices from `@builtin(vertex_index)`, with the view-projection
//! and its inverse; the shader reconstructs a world-space ray per pixel and draws a
//! fading grid on the y = 0 plane. The two matrices are a uniform block because this
//! RHI cannot declare a push-constant range.
//!
//! A line list with a per-vertex colour is the smallest thing that draws a gizmo.

use bytemuck::{Pod, Zeroable};
use chassis::{BindGroup, Buffer, CommandList, PassScope, Pipeline};

/// Bind group number within the grid shader.
pub const GRID_SET: u32 = 0;
/// Bind group number within the line shader.
pub const LINE_DRAW_SET: u32 = 0;

/// The grid's axis colours: the same hues as the gizmo, but dark enough that a
/// selected entity's bright X and Z handles read on top of them.
pub const GRID_AXIS_X: [f32; 4] = [0.50, 0.15, 0.12, 1.0];
pub const GRID_AXIS_Z: [f32; 4] = [0.12, 0.18, 0.50, 1.0];

/// The two matrices the grid shader needs, and its axis colours.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GridUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub inverse_view_projection: [[f32; 4]; 4],
    pub axis_x: [f32; 4],
    pub axis_z: [f32; 4],
}

impl GridUniforms {
    pub fn new(view_projection: [[f32; 4]; 4]) -> Self {
        Self {
            view_projection,
            inverse_view_projection: glam::Mat4::from_cols_array_2d(&view_projection)
                .inverse()
                .to_cols_array_2d(),
            axis_x: GRID_AXIS_X,
            axis_z: GRID_AXIS_Z,
        }
    }
}

/// One end of a gizmo line.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// The camera and the batch's own placement, which `line.wgsl` reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

impl LineUniforms {
    pub fn new(view_projection: [[f32; 4]; 4], model: [[f32; 4]; 4]) -> Self {
        Self {
            view_projection,
            model,
        }
    }
}

/// The grid pipeline, drawn as six vertices with no vertex buffer: the quad comes
/// from `@builtin(vertex_index)`.
pub struct GridPass {
    pipeline: Pipeline,
    group: BindGroup,
}

impl GridPass {
    pub fn new(pipeline: Pipeline, group: BindGroup) -> Self {
        Self { pipeline, group }
    }

    pub fn record(
        &self,
        _scope: &PassScope<'_>,
        list: &mut dyn CommandList,
    ) -> Result<(), chassis::Error> {
        list.bind_pipeline(&self.pipeline);
        list.bind_group(GRID_SET, &self.group);
        list.draw(6, 1);
        Ok(())
    }
}

/// The line batch's buffers, uploaded once by whoever builds the gizmos.
#[derive(Clone, Copy, Debug)]
pub struct LineBatch {
    pub vertex: Buffer,
    pub index: Buffer,
    pub index_count: u32,
}

impl LineBatch {
    pub fn is_empty(&self) -> bool {
        self.index_count == 0
    }
}

/// A line-list pass over a prepared batch.
pub struct LinePass {
    pipeline: Pipeline,
    /// Set 0: the batch's model matrix, so a gizmo can be placed.
    uniforms: BindGroup,
    batch: LineBatch,
}

impl LinePass {
    pub fn new(pipeline: Pipeline, uniforms: BindGroup, batch: LineBatch) -> Self {
        Self {
            pipeline,
            uniforms,
            batch,
        }
    }

    pub fn record(
        &self,
        _scope: &PassScope<'_>,
        list: &mut dyn CommandList,
    ) -> Result<(), chassis::Error> {
        if self.batch.is_empty() {
            return Ok(());
        }

        list.bind_pipeline(&self.pipeline);
        list.bind_group(LINE_DRAW_SET, &self.uniforms);
        list.bind_vertex_buffer(0, &self.batch.vertex, 0);
        list.bind_index_buffer(&self.batch.index, 0);
        list.draw_indexed(self.batch.index_count, 1);
        Ok(())
    }
}

/// Packs vertices and indices for a line batch. Pure, so the interleave is
/// testable without a device.
pub fn pack(vertices: &[LineVertex], indices: &[u32]) -> (Vec<u8>, Vec<u8>) {
    (
        bytemuck::cast_slice(vertices).to_vec(),
        bytemuck::cast_slice(indices).to_vec(),
    )
}

/// Turns world-space segments into the line pass's format.
///
/// A segment is one line-list primitive, so it is two vertices and one index
/// pair; the alpha is opaque because the debug view's colours are RGB.
pub fn debug_batch(segments: &[femur::DebugLine]) -> (Vec<LineVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(segments.len() * 2);
    let mut indices = Vec::with_capacity(segments.len() * 2);
    for line in segments {
        push_segment(&mut vertices, &mut indices, line);
    }
    (vertices, indices)
}

/// All three debug layers, in the order the physics backend groups them.
pub fn physics_debug_batch(debug: &femur::PhysicsDebug) -> (Vec<LineVertex>, Vec<u32>) {
    let count = debug.shapes.len() + debug.contacts.len() + debug.broad_phase.len();
    let mut vertices = Vec::with_capacity(count * 2);
    let mut indices = Vec::with_capacity(count * 2);
    for line in debug
        .shapes
        .iter()
        .chain(&debug.contacts)
        .chain(&debug.broad_phase)
    {
        push_segment(&mut vertices, &mut indices, line);
    }
    (vertices, indices)
}

fn push_segment(vertices: &mut Vec<LineVertex>, indices: &mut Vec<u32>, line: &femur::DebugLine) {
    let first = vertices.len() as u32;
    let color = [line.color.x, line.color.y, line.color.z, 1.0];
    vertices.push(LineVertex {
        position: line.from.to_array(),
        color,
    });
    vertices.push(LineVertex {
        position: line.to.to_array(),
        color,
    });
    indices.push(first);
    indices.push(first + 1);
}
