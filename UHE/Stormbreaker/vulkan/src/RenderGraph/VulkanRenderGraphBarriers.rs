//! Recording one pass: its barriers and its render scope.
//!
//! ## Why this is a separate file
//!
//! `VulkanRenderGraphExecutor.rs` was one file holding the executor, the
//! registration table, the barrier lowering and the scope opening -- 757 lines,
//! none of which you could read without reading the rest. Barrier lowering is
//! the only place that knows about both `vk::ImageMemoryBarrier` and
//! `chassis::graph::ImageBarrier` at once, and it is where you go when a frame
//! has a hazard.
//!
//! ## Why barriers are not in the graph
//!
//! The graph derives barriers from declared reads and writes but cannot emit
//! them: `vk::ImageMemoryBarrier` is a Vulkan type, and the graph is
//! backend-neutral so that wgpu can use it without depending on ash. wgpu does
//! not need this file at all -- it inserts transitions itself.
//!
//! ## What is not here
//!
//! Handle resolution. These functions take `&Textures` and ask it for a
//! resource; deciding *which* resource a slot means belongs to
//! `vulkan_render_graph_resources`. Doing it per barrier here would mean the
//! same lookup twice per barrier, with two chances to disagree.

use chassis::graph::pass::SlotTable;
// The trait, not just the type: `begin_rendering` and `set_viewport` are trait
// methods, and `VulkanCommandList`'s inherent methods do not cover them.
use chassis::CommandList as _;

use crate::vulkan_barrier_batch::BarrierBatch;
use crate::vulkan_barrier_encoder::BarrierEncoder;
use crate::vulkan_command_buffer::CommandList as VulkanCommandList;
use crate::vulkan_error::{BackendError, Result};
use crate::vulkan_texture::Textures;

/// Record a pass's pre barriers.
///
/// Barriers go in *before* the pass records anything: a transition that lands
/// after the draw it was meant to protect is not a barrier, it is a comment.
pub fn record_pre_barriers(
    encoder: &BarrierEncoder,
    pass: &chassis::graph::CompiledPass,
    slots: &SlotTable,
    textures: &Textures,
    buffers: &crate::vulkan_buffer::Buffers,
    list: &mut VulkanCommandList<'_>,
) -> Result<u32> {
    let buffer = list.raw().handle();
    let mut batch = BarrierBatch::new(encoder);

    for barrier in &pass.pre_barriers {
        if barrier.is_empty() {
            continue; // the batch would drop it; do not count it either
        }
        let Some(handle) = slots.texture(barrier.texture) else {
            return Err(BackendError::InvalidHandle);
        };
        let resource = textures.get(handle).ok_or(BackendError::InvalidHandle)?;
        batch.image(resource.barrier_target(), barrier, resource.format());
    }

    // Buffer barriers go through the batch rather than the encoder, because a
    // buffer has no subresource range and no format -- the encoder's `Target` is
    // an image concept and would be a lie here.
    for barrier in &pass.pre_buffer_barriers {
        let Some(handle) = slots.buffer(barrier.buffer) else {
            return Err(BackendError::InvalidHandle);
        };
        let resource = buffers.get(handle).ok_or(BackendError::InvalidHandle)?;
        let (offset, size) = match (barrier.offset, barrier.size) {
            // The graph says "the rest of it" because it knows a buffer exists,
            // not how large it is. The executor is the first place that knows.
            (_, chassis::graph::WHOLE_SIZE) => (0, resource.size()),
            (offset, size) => (offset, size),
        };
        batch.buffer(resource.handle(), offset, size, barrier);
    }

    Ok(batch.flush(buffer))
}

/// Record a pass's post barriers, after the pass body and the render scope.
///
/// Separate from [`record_barriers`] because the order is the whole point. Both
/// halves used to be emitted together at the top of the pass, which transitioned
/// the attachment to its *final* layout -- `TRANSFER_SRC_OPTIMAL` for a readback
/// target -- before anything had drawn into it. Validation named it precisely:
/// VUID-vkCmdBeginRendering-pRenderingInfo-09592, the attachment was in
/// TRANSFER_SRC_OPTIMAL where COLOR_ATTACHMENT_OPTIMAL was required.
///
/// A barrier recorded after the draw it protects is a comment. A post barrier
/// recorded before the draw it follows is worse: it is a wrong one.
pub fn record_post_barriers(
    encoder: &BarrierEncoder,
    pass: &chassis::graph::CompiledPass,
    slots: &SlotTable,
    textures: &Textures,
    list: &mut VulkanCommandList<'_>,
) -> Result<u32> {
    let buffer = list.raw().handle();
    let mut batch = BarrierBatch::new(encoder);
    for barrier in &pass.post_barriers {
        if barrier.is_empty() {
            continue;
        }
        let Some(handle) = slots.texture(barrier.texture) else {
            return Err(BackendError::InvalidHandle);
        };
        let resource = textures.get(handle).ok_or(BackendError::InvalidHandle)?;
        batch.image(resource.barrier_target(), barrier, resource.format());
    }
    Ok(batch.flush(buffer))
}

/// Begin dynamic rendering for a pass, if it has attachments. Returns whether a
/// scope was opened, so the caller closes exactly what it opened.
///
/// A pass with no attachments -- a compute pass, or a transfer -- gets no scope,
/// and `vkCmdBeginRendering` on a pass with nothing to render into is an error.
pub fn open_render_scope(
    pass: &chassis::graph::CompiledPass,
    slots: &SlotTable,
    textures: &Textures,
    extent: &chassis::Extent2d,
    list: &mut VulkanCommandList<'_>,
) -> Result<bool> {
    if pass.colors.is_empty() && pass.depth.is_none() {
        return Ok(false);
    }

    // Handles and formats, not views: the `CommandList` adapter resolves views
    // through the texture registry, which is the only place that knows how a
    // texture was created. Building views here would duplicate that and let the
    // two disagree.
    let mut desc = chassis::RenderPassDesc::default();
    for color in &pass.colors {
        let Some(handle) = slots.texture(color.texture) else {
            return Err(BackendError::InvalidHandle);
        };
        if textures.get(handle).is_none() {
            return Err(BackendError::InvalidHandle);
        }
        desc.colors.push(chassis::ColorAttachmentDesc {
            texture: handle,
            // The graph's `ColorAttachment` has no format field -- the pipeline
            // fixed it at creation -- so it is read back off the texture, which
            // is where it came from.
            format: textures.get(handle).expect("checked above").format(),
            load: match color.load {
                chassis::graph::LoadOp::Clear => chassis::LoadOp::Clear,
                chassis::graph::LoadOp::Load => chassis::LoadOp::Load,
                chassis::graph::LoadOp::DontCare => chassis::LoadOp::DontCare,
            },
            clear: color.clear,
            store: true,
            resolve: chassis::Texture::null(),
        });
    }
    if let Some(depth) = &pass.depth {
        let Some(handle) = slots.texture(depth.texture) else {
            return Err(BackendError::InvalidHandle);
        };
        let resource = textures.get(handle).ok_or(BackendError::InvalidHandle)?;
        desc.depth = Some(chassis::DepthAttachmentDesc {
            texture: handle,
            format: resource.format(),
            load: match depth.load {
                chassis::graph::LoadOp::Clear => chassis::LoadOp::Clear,
                chassis::graph::LoadOp::Load => chassis::LoadOp::Load,
                chassis::graph::LoadOp::DontCare => chassis::LoadOp::DontCare,
            },
            clear_depth: depth.clear_depth,
            store: true,
        });
    }

    list.begin_rendering(&desc);
    // Dynamic viewport and scissor, which the pipeline declared. Validation is
    // explicit that a dynamic state has to be set per command buffer before the
    // draw that uses it: VUID-vkCmdDraw-None-07831.
    list.set_viewport(*extent);
    list.set_scissor(0, 0, extent.width, extent.height);
    Ok(true)
}
