//! The depth prepass, and the shadow map it produces.
//!
//! A depth-only draw of every shadow-casting mesh, from whichever matrix the
//! caller gives it; the camera's own depth prepass is the same pass with the
//! camera's matrix, so there is one implementation, not two.
//!
//! A depth-only pipeline has one stage: `ShaderSet::fragment_for` returns `None`
//! for it.

use chassis::{BindGroup, CommandList, PassScope, Pipeline};

use super::geometry::PreparedDraw;

/// Bind group numbers within the depth pipeline's shader.
pub const SHADOW_SET: u32 = 0;
pub const DRAW_SET: u32 = 1;
/// The skin palette, the same buffer the model pass binds at set 3.
pub const PALETTE_SET: u32 = 2;

pub struct ShadowPass {
    pipeline: Pipeline,
    /// Set 0: the light-space view-projection the depth is measured against.
    frame: BindGroup,
    palette: BindGroup,
    draws: Vec<PreparedDraw>,
}

impl ShadowPass {
    pub fn new(
        pipeline: Pipeline,
        frame: BindGroup,
        palette: BindGroup,
        draws: Vec<PreparedDraw>,
    ) -> Self {
        Self {
            pipeline,
            frame,
            palette,
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

        list.bind_pipeline(&self.pipeline);
        list.bind_group(SHADOW_SET, &self.frame);
        list.bind_group(PALETTE_SET, &self.palette);

        for draw in &self.draws {
            list.bind_group(DRAW_SET, &draw.uniforms);
            list.bind_vertex_buffer(0, &draw.mesh.vertex, 0);
            list.bind_index_buffer(&draw.mesh.index, 0);
            list.draw_indexed(draw.mesh.index_count, 1);
        }

        Ok(())
    }
}
