//! The geometry pass: opaque draws, then blended ones.
//!
//! Per primitive: bind the pipeline its material selects, bind its uniforms and
//! vertex and index buffers, and draw indexed. The pipeline is switched per
//! primitive because cull mode and blending are pipeline state, not per-draw
//! state a material can ask for. `prepare_draws` orders the draws so consecutive
//! ones share state, and each bind is skipped when it did not change.
//!
//! Transparent draws are not yet a pass of their own. They should be, with their
//! own depth-tested pipeline; `PASS_TRANSPARENT` is the slot it will record into.
//! Until then this pass draws the opaques and then the blended items back to
//! front, which needs depth sorting to avoid artefacts where surfaces overlap.

use chassis::{BindGroup, CommandList, PassScope, Pipeline};

use crate::material::MaterialVariant;
use crate::mesh::GpuMesh;

/// Bind group numbers within the model pipeline's shader.
pub const FRAME_SET: u32 = 0;
pub const MATERIAL_SET: u32 = 1;
pub const DRAW_SET: u32 = 2;

/// The four pipelines one geometry pass can bind, one per [`MaterialVariant`].
///
/// Cull mode and blending are pipeline state, so a material cannot ask for them
/// per draw.
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelines {
    pub opaque: Pipeline,
    pub opaque_double_sided: Pipeline,
    pub transparent: Pipeline,
    pub transparent_double_sided: Pipeline,
}

impl MaterialPipelines {
    pub fn get(&self, variant: MaterialVariant) -> Pipeline {
        match variant {
            MaterialVariant::Opaque => self.opaque,
            MaterialVariant::OpaqueDoubleSided => self.opaque_double_sided,
            MaterialVariant::Transparent => self.transparent,
            MaterialVariant::TransparentDoubleSided => self.transparent_double_sided,
        }
    }
}

/// A draw whose buffers, uniforms and material are already resolved.
#[derive(Clone, Copy, Debug)]
pub struct PreparedDraw {
    pub mesh: GpuMesh,
    /// The set that carries this draw's model matrix, tint and entity id.
    pub uniforms: BindGroup,
    /// `None` draws with the default material.
    pub material: Option<BindGroup>,
    pub variant: MaterialVariant,
}

impl PreparedDraw {
    /// Blended draws are drawn after every opaque draw.
    pub fn transparent(&self) -> bool {
        self.variant.blends()
    }
}

pub struct GeometryPass {
    pipelines: MaterialPipelines,
    frame: BindGroup,
    draws: Vec<PreparedDraw>,
}

impl GeometryPass {
    pub fn new(pipelines: MaterialPipelines, frame: BindGroup, draws: Vec<PreparedDraw>) -> Self {
        Self {
            pipelines,
            frame,
            draws,
        }
    }

    pub fn draws(&self) -> usize {
        self.draws.len()
    }

    pub fn record(
        &self,
        _scope: &PassScope<'_>,
        list: &mut dyn CommandList,
    ) -> Result<(), chassis::Error> {
        if self.draws.is_empty() {
            return Ok(());
        }

        // The pipeline first, and then the frame set: a descriptor set is bound
        // *against* a pipeline layout, and Vulkan has no layout to give it before
        // a pipeline is bound. wgpu does not care, which is how this order stayed
        // invisible until the same frame ran on Vulkan.
        list.bind_pipeline(&self.pipelines.get(self.draws[0].variant));
        list.bind_group(FRAME_SET, &self.frame);

        // Each bind is only issued when it differs from the previous draw. The
        // draw set always changes: it is what makes this draw's model matrix
        // different from the last. The pipeline, material and mesh are shared
        // wherever `prepare_draws` ordered draws that share them next to each
        // other.
        let mut pipeline: Option<MaterialVariant> = Some(self.draws[0].variant);
        let mut material: Option<BindGroup> = None;
        let mut mesh: Option<GpuMesh> = None;

        for draw in &self.draws {
            if pipeline != Some(draw.variant) {
                list.bind_pipeline(&self.pipelines.get(draw.variant));
                pipeline = Some(draw.variant);
            }
            if let Some(bound) = draw.material {
                if material != Some(bound) {
                    list.bind_group(MATERIAL_SET, &bound);
                    material = Some(bound);
                }
            }
            if mesh != Some(draw.mesh) {
                list.bind_vertex_buffer(0, &draw.mesh.vertex, 0);
                list.bind_index_buffer(&draw.mesh.index, 0);
                mesh = Some(draw.mesh);
            }
            list.bind_group(DRAW_SET, &draw.uniforms);
            list.draw_indexed(draw.mesh.index_count, 1);
        }

        Ok(())
    }
}
