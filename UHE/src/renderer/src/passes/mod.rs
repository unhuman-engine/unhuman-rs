//! The frame's passes, and the seam a backend records them through.
//!
//! The dispatch is by pass name rather than closure, so a compiled frame stays
//! pure data a worker thread can own.

pub mod composite;
pub mod editor;
pub mod geometry;
pub mod lines;
pub mod overlay;
pub mod pipeline;
pub mod readback;
pub mod shadow;
pub mod ssao;
pub mod uniforms;

use std::sync::Arc;

use chassis::graph::{PassRecorder, PassScope, TextureId};
use chassis::{CommandList, Error};
use glam::Mat4;

pub use composite::CompositePass;
pub use editor::EditorPass;
pub use geometry::{GeometryPass, MaterialPipelines, PreparedDraw};
pub use lines::{GridPass, GridUniforms, LineBatch, LinePass, LineUniforms, LineVertex};
pub use overlay::{OverlayFrame, SpriteTextOverlay, PASS_2D, PASS_TEXT};
pub use pipeline::{PipelineCache, PipelineKey, ShaderSet, ShaderVariant};
pub use readback::{ReadbackPass, ReadbackRing, READBACK_LAG, READBACK_RING};
pub use shadow::ShadowPass;
pub use ssao::{SsaoPass, SsaoUniforms};
pub use uniforms::{
    BindSet, CompositeUniforms, DrawArena, DrawUniforms, FrameUniforms, ImageSet, PaletteSet,
    ShadowUniforms,
};

/// Pass names, which are also the keys the dispatch is done on.
pub const PASS_SHADOW: &str = "shadow";
pub const PASS_OPAQUE: &str = "opaque";
/// The blended draws, after the opaques, loading the colour and depth they
/// wrote rather than clearing them.
pub const PASS_TRANSPARENT: &str = "transparent";
pub const PASS_SSAO: &str = "ssao";
/// Depth-aware upsample of the half-resolution occlusion into the full-rate
/// texture composition reads.
pub const PASS_SSAO_UPSAMPLE: &str = "ssao_upsample";
/// The grid and the gizmos, in one pass.
pub const PASS_EDITOR: &str = "editor";
pub const PASS_COMPOSITE: &str = "composite";
/// The composed image copied out for the CPU. Headless only; there is nothing to
/// read back when the frame is presented to a surface.
pub const PASS_READBACK: &str = "readback";

/// The graph slots this frame's passes are attached to.
///
/// Only the backend can resolve `backbuffer`, because a swapchain image has no
/// RHI `Texture` handle to turn into a barrier target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameTargets {
    pub scene_color: TextureId,
    pub scene_depth: TextureId,
    pub shadow: TextureId,
    pub ssao: TextureId,
    pub backbuffer: TextureId,
}

/// Passes this crate does not own: the 2D batch and the text slice.
///
/// `declare` is called while the frame's graph is open, `record` for each pass
/// name it declared.
pub trait OverlayPasses: Send + Sync {
    /// Starts a frame, clearing what the previous one queued.
    fn begin(&self) {}
    /// Moves the producer's queued content into this frame's batches.
    fn edit(&self) {}
    /// Uploads the batches, after `edit` and before recording.
    fn prepare(
        &self,
        _resources: &mut dyn chassis::Resources,
        _view_projection: Mat4,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn declare(&self, graph: &mut chassis::graph::RenderGraph, targets: &FrameTargets);
    fn record(
        &self,
        pass: &str,
        scope: &PassScope<'_>,
        list: &mut dyn CommandList,
    ) -> Result<(), Error>;
    /// Drops the overlay's own GPU resources.
    fn destroy(&self, _resources: &mut dyn chassis::Resources) {}
}

/// Every recorder of one frame, already prepared.
///
/// Built once the frame's textures and uniform buffers exist, then shared with
/// the backend as one `Arc<dyn PassRecorder>`; nothing mutates while recording,
/// so a backend may record passes on worker threads.
pub struct FrameRecorder {
    pub shadow: Option<ShadowPass>,
    pub geometry: Option<GeometryPass>,
    pub transparent: Option<GeometryPass>,
    pub ssao: Option<SsaoPass>,
    pub ssao_upsample: Option<SsaoPass>,
    pub editor: Option<EditorPass>,
    pub composite: Option<CompositePass>,
    pub readback: Option<ReadbackPass>,
    pub overlay: Option<Arc<dyn OverlayPasses>>,
}

impl FrameRecorder {
    pub fn new() -> Self {
        Self {
            shadow: None,
            geometry: None,
            transparent: None,
            ssao: None,
            ssao_upsample: None,
            editor: None,
            composite: None,
            readback: None,
            overlay: None,
        }
    }
}

impl Default for FrameRecorder {
    fn default() -> Self {
        Self::new()
    }
}

fn missing(pass: &str) -> Error {
    Error::InvalidDescriptor(format!(
        "pass '{pass}' is in the frame's graph with no recorder behind it"
    ))
}

impl PassRecorder for FrameRecorder {
    fn record(&self, scope: &PassScope<'_>, list: &mut dyn CommandList) -> Result<(), Error> {
        match scope.name {
            PASS_SHADOW => match &self.shadow {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_SHADOW)),
            },
            PASS_OPAQUE => match &self.geometry {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_OPAQUE)),
            },
            PASS_TRANSPARENT => match &self.transparent {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_TRANSPARENT)),
            },
            PASS_SSAO => match &self.ssao {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_SSAO)),
            },
            PASS_SSAO_UPSAMPLE => match &self.ssao_upsample {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_SSAO_UPSAMPLE)),
            },
            PASS_EDITOR => match &self.editor {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_EDITOR)),
            },
            PASS_COMPOSITE => match &self.composite {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_COMPOSITE)),
            },
            PASS_READBACK => match &self.readback {
                Some(pass) => pass.record(scope, list),
                None => Err(missing(PASS_READBACK)),
            },
            other => match &self.overlay {
                Some(overlay) => overlay.record(other, scope, list),
                None => Err(Error::InvalidDescriptor(format!(
                    "no recorder for pass '{other}': it is neither a bifrost pass nor one \
                     declared by an installed overlay"
                ))),
            },
        }
    }
}
