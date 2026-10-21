//! The single lowering point from graph barriers to Vulkan commands.
//!
//! The render graph emits backend-neutral [`ImageBarrier`]s; this turns
//! them into `vkCmdPipelineBarrier2` or `vkCmdPipelineBarrier` depending on the
//! device's `SyncTier`. There is no second path: a pass cannot hand-write a
//! barrier, because `chassis::CommandList` has no barrier method at all.
//!
//! Two owners of synchronization cannot agree reliably: a pass writing its own
//! barriers while the graph synthesizes others needs a handshake to reconcile
//! them, and the handshake is where a hazard slips through. One lowering point
//! removes that class of bug.

use ash::vk;
use chassis::graph::types::{BufferBarrier, ImageBarrier};
use chassis::TextureFormat;

use crate::vulkan_extension_check::SyncTier;
use crate::vulkan_types;

/// Which handle the barrier applies to. Resolved by the caller from the graph's
/// texture slot; the graph itself never sees a Vulkan handle.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub image: vk::Image,
    pub subresource: vk::ImageSubresourceRange,
}

/// The layout a graph state lowers to.
///
/// The compiler emits backend-neutral `ImageState`s and this is the one place
/// they become `vk::ImageLayout`, so a second copy of this mapping elsewhere is
/// how a layout ends up disagreeing with the barrier that set it.
pub fn layout_for(state: chassis::graph::types::ImageState) -> vk::ImageLayout {
    use chassis::graph::types::ImageState;
    match state {
        ImageState::Undefined => vk::ImageLayout::UNDEFINED,
        ImageState::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ImageState::DepthAttachment => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        ImageState::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ImageState::Storage => vk::ImageLayout::GENERAL,
        ImageState::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageState::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageState::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

impl Target {
    /// A range whose aspects are the ones `format` actually has.
    ///
    /// The constructor to reach for: [`subresource_for`] intersects this with the
    /// format's own aspects, so a range built with [`Target::color`] for a depth
    /// image intersects to nothing and the barrier is invalid.
    pub fn for_format(
        image: vk::Image,
        format: chassis::TextureFormat,
        mip_levels: u32,
        layers: u32,
    ) -> Self {
        Self {
            image,
            subresource: vk::ImageSubresourceRange {
                aspect_mask: vulkan_types::aspect(format),
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: layers,
            },
        }
    }

    pub fn color(image: vk::Image, mip_levels: u32, layers: u32) -> Self {
        Self {
            image,
            subresource: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: layers,
            },
        }
    }

    pub fn with_aspect(mut self, aspect: vk::ImageAspectFlags) -> Self {
        self.subresource.aspect_mask = aspect;
        self
    }
}

/// The buffer counterpart of [`image_barrier2`].
///
/// Same tier split, and the same reason for existing as a separate function:
/// `vk::BufferMemoryBarrier2` carries stage and access masks in the struct and
/// `vk::BufferMemoryBarrier` does not, so one cannot be built from the other.
pub fn buffer_barrier2(
    buffer: vk::Buffer,
    offset: u64,
    size: vk::DeviceSize,
    barrier: BufferBarrier,
) -> vk::BufferMemoryBarrier2<'static> {
    vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vulkan_types::stage(barrier.src_stage))
        .src_access_mask(vulkan_types::access(barrier.src_access))
        .dst_stage_mask(vulkan_types::stage(barrier.dst_stage))
        .dst_access_mask(vulkan_types::access(barrier.dst_access))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size)
}

pub fn buffer_barrier_legacy(
    buffer: vk::Buffer,
    offset: u64,
    size: vk::DeviceSize,
    barrier: BufferBarrier,
) -> vk::BufferMemoryBarrier<'static> {
    vk::BufferMemoryBarrier::default()
        .src_access_mask(vulkan_types::access_legacy(barrier.src_access))
        .dst_access_mask(vulkan_types::access_legacy(barrier.dst_access))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size)
}

/// ash gives these structs a lifetime parameter that only exists for its
/// builder. Nothing here borrows, so `'static` is accurate.
///
/// Split out from recording so the lowering is testable with no command buffer,
/// no device and no driver; these are plain structs.
pub fn image_barrier2(
    target: Target,
    barrier: ImageBarrier,
    format: TextureFormat,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vulkan_types::stage(barrier.src_stage))
        .src_access_mask(vulkan_types::access(barrier.src_access))
        .dst_stage_mask(vulkan_types::stage(barrier.dst_stage))
        .dst_access_mask(vulkan_types::access(barrier.dst_access))
        .old_layout(vulkan_types::layout_for(barrier.src_state, format))
        .new_layout(vulkan_types::layout_for(barrier.dst_state, format))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(target.image)
        .subresource_range(subresource_for(target, format))
}

/// The `vk::ImageMemoryBarrier` encoding for the pre-1.3 path. Stages and
/// accesses live in the mask arguments rather than the struct, which is why the
/// two encodings are separate functions instead of one struct with fields left
/// empty.
pub fn image_barrier_legacy(
    target: Target,
    barrier: ImageBarrier,
    format: TextureFormat,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(vulkan_types::access_legacy(barrier.src_access))
        .dst_access_mask(vulkan_types::access_legacy(barrier.dst_access))
        .old_layout(vulkan_types::layout_for(barrier.src_state, format))
        .new_layout(vulkan_types::layout_for(barrier.dst_state, format))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(target.image)
        .subresource_range(subresource_for(target, format))
}

/// The target's range, narrowed to the aspects the format actually has. A
/// stencil aspect on a depth-only format is a validation error, and a colour
/// aspect on a depth image is worse.
fn subresource_for(target: Target, format: TextureFormat) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vulkan_types::aspect(format) & target.subresource.aspect_mask,
        ..target.subresource
    }
}

pub struct BarrierEncoder {
    tier: SyncTier,
    /// Owned, not borrowed.
    ///
    /// One encoder is shared by every recording thread, so it must be `'static`
    /// to go into whatever the threads capture. An `ash::Device` is a
    /// refcounted handle, so cloning it is two words and a driver refcount bump
    /// -- once per frame, not once per barrier.
    device: ash::Device,
}

impl BarrierEncoder {
    pub fn new(tier: SyncTier, device: &ash::Device) -> Self {
        Self {
            tier,
            device: device.clone(),
        }
    }

    pub fn tier(&self) -> SyncTier {
        self.tier
    }

    /// The device barriers are recorded on.
    ///
    /// Public because a buffer barrier has no `Target` -- there is no subresource
    /// range for a buffer -- so it goes through the free lowering functions and
    /// the caller needs the device to record them.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }
}
