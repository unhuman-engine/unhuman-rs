//! The Vulkan backend.
//!
//! Everything the engine knows about graphics lives behind `chassis`, so this
//! crate is the only place a Vulkan type is allowed to appear. The two files
//! that matter architecturally are `extension_check` (which sync tier a device
//! gets) and `barrier_encoder` (the single place a graph barrier becomes a
//! Vulkan command).
//!
//! Files keep their `Vulkan*` names while the module identifier stays
//! snake_case through `#[path]`, which keeps `use crate::vulkan_buffer::...`
//! idiomatic and avoids a non_snake_case warning on every declaration.

#[path = "VulkanAllocation.rs"]
pub mod vulkan_allocation;
#[path = "VulkanBarrierEncoder.rs"]
pub mod vulkan_barrier_encoder;
#[path = "VulkanBuffer.rs"]
pub mod vulkan_buffer;
#[path = "VulkanCommandBuffer.rs"]
pub mod vulkan_command_buffer;
#[path = "VulkanCommandPool.rs"]
pub mod vulkan_command_pool;
#[path = "VulkanComputePipeline.rs"]
pub mod vulkan_compute_pipeline;
#[path = "VulkanContext.rs"]
pub mod vulkan_context;
#[path = "VulkanDeletionQueue.rs"]
pub mod vulkan_deletion_queue;
#[path = "VulkanDescriptorManager.rs"]
pub mod vulkan_descriptor_manager;
#[path = "VulkanDescriptorPool.rs"]
pub mod vulkan_descriptor_pool;
#[path = "VulkanDescriptorSet.rs"]
pub mod vulkan_descriptor_set;
#[path = "VulkanDevice.rs"]
pub mod vulkan_device;
#[path = "VulkanError.rs"]
pub mod vulkan_error;
#[path = "VulkanExtensionCheck.rs"]
pub mod vulkan_extension_check;
#[path = "VulkanFence.rs"]
pub mod vulkan_fence;
#[path = "VulkanFrameContext.rs"]
pub mod vulkan_frame_context;
#[path = "VulkanFramebuffer.rs"]
pub mod vulkan_framebuffer;
#[path = "VulkanGraphicPipeline.rs"]
pub mod vulkan_graphic_pipeline;
#[path = "VulkanImageView.rs"]
pub mod vulkan_image_view;
#[path = "VulkanInstance.rs"]
pub mod vulkan_instance;
#[path = "VulkanLogicalDevice.rs"]
pub mod vulkan_logical_device;
#[path = "VulkanPhysicalDevice.rs"]
pub mod vulkan_physical_device;
#[path = "VulkanPipelineCache.rs"]
pub mod vulkan_pipeline_cache;
#[path = "VulkanPipelineState.rs"]
pub mod vulkan_pipeline_state;
#[path = "VulkanScopes.rs"]
pub mod vulkan_scopes;
#[path = "VulkanSemaphore.rs"]
pub mod vulkan_semaphore;
#[path = "VulkanShader.rs"]
pub mod vulkan_shader;
#[path = "VulkanSubmitEncoder.rs"]
pub mod vulkan_submit_encoder;
#[path = "VulkanSwapChain.rs"]
pub mod vulkan_swapchain;
#[path = "VulkanTexture.rs"]
pub mod vulkan_texture;
#[path = "VulkanTexture2D.rs"]
pub mod vulkan_texture_2d;
#[path = "VulkanTypes.rs"]
pub mod vulkan_types;
#[path = "VulkanUpload.rs"]
pub mod vulkan_upload;
#[path = "VulkanUtils.rs"]
pub mod vulkan_utils;
pub use vulkan_allocation::{
    create_buffer, create_image_memory, find_memory_type, read_buffer, write_buffer,
};
pub use vulkan_barrier_encoder::{buffer_barrier2, buffer_barrier_legacy};
pub use vulkan_barrier_encoder::{image_barrier2, image_barrier_legacy, BarrierEncoder, Target};
pub use vulkan_buffer::{BufferResource, Buffers};
pub use vulkan_command_buffer::Recording;
pub use vulkan_compute_pipeline::ComputePipelineDesc;
pub use vulkan_context::Context;
pub use vulkan_deletion_queue::{DeletionQueue, Pending};
pub use vulkan_descriptor_manager::Descriptors;
pub use vulkan_descriptor_pool::{DescriptorPool, PoolSizes};
pub use vulkan_descriptor_set::{descriptor_type_of, DescriptorSet};
pub use vulkan_device::{create_device, device_available, DeviceOptions};
pub use vulkan_error::{BackendError, Result};
pub use vulkan_extension_check::{select_sync_tier, Capabilities, QueueFamily, SyncTier};
pub use vulkan_extension_check::{version_major, version_minor};
pub use vulkan_fence::Fence;
pub use vulkan_frame_context::Recorder;
pub use vulkan_graphic_pipeline as graphic;
pub use vulkan_pipeline_cache::PipelineCache;
pub use vulkan_pipeline_state::{PipelineHash, PipelineResource, Pipelines};
pub use vulkan_semaphore::Semaphore;
pub use vulkan_shader::{content_hash, ShaderResource, Shaders};
pub use vulkan_texture::{TextureResource, Textures};
pub use vulkan_upload::UploadRing;
#[path = "VulkanGpu.rs"]
pub mod vulkan_gpu;
#[path = "RenderGraph/VulkanRenderGraphBarriers.rs"]
pub mod vulkan_render_graph_barriers;
#[path = "RenderGraph/VulkanRenderGraphExecutor.rs"]
pub mod vulkan_render_graph_executor;
#[path = "RenderGraph/VulkanRenderGraphResources.rs"]
pub mod vulkan_render_graph_resources;
#[path = "VulkanRenderPass.rs"]
pub mod vulkan_render_pass;
#[path = "VulkanResources.rs"]
pub mod vulkan_resources;
