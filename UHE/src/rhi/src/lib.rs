//! The RHI interface.
//!
//! The test that keeps this crate honest: **no Vulkan type appears in any
//! signature outside the backend.** That is enforced here by the dependency
//! list being empty, not by discipline.

/// The render graph: declaration, versioning, compilation.
///
/// Backend-neutral by construction: nothing here imports `ash` or any driver,
/// which is what lets both backends use the same one.
pub mod graph;

pub mod backend;
mod desc;
mod format;
mod handle;
mod rhi;
pub mod settings;

pub use backend::{
    flag_value, BackendChoice, BackendSelection, BackendSource, UnknownBackend,
    ENVIRONMENT_VARIABLE as BACKEND_ENVIRONMENT_VARIABLE, FLAG as BACKEND_FLAG,
};
pub use desc::{
    copy_row_pitch, AddressMode, BindingDesc, BlendState, BufferDesc, ColorAttachmentDesc,
    ColorTargetState, DepthAttachmentDesc, DepthState, DescriptorKind, DescriptorSetLayoutDesc,
    Extent, Filter, IndexFormat, LoadOp, PipelineDesc, PrimitiveState, RenderPassDesc, SamplerDesc,
    ShaderStage, ShaderStageDesc, StageMask, TextureDesc, VertexAttribute, VertexLayout,
    COPY_BYTES_PER_ROW_ALIGNMENT,
};
pub use format::{BufferUsage, SampleCount, TextureFormat, TextureUsage};
pub use graph::{PassRecorder, PassScope, SlotTable};
pub use handle::{
    BindGroup, BindGroupTag, Buffer, BufferTag, Handle, ImageView, ImageViewTag, Pipeline,
    PipelineTag, Registry, Sampler, SamplerTag, Shader, ShaderTag, Texture, TextureTag,
};
pub use rhi::{
    CommandList, Error, Extent2d, Gpu, GpuInfo, PassId, RenderGraph, Resources, Result, Swapchain,
    SyncTier,
};
