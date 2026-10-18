//! Descriptor sets.
//!
//! Sets live here and are bound at an explicit set index by
//! `vulkan_command_buffer`. There is no global set, so the failure mode where a
//! `BindPipeline` rebinds one process-wide set has no representation.

use ash::vk;

use crate::vulkan_error::{BackendError, Result};

/// Re-exported from the RHI rather than defined here.
///
/// There were two `DescriptorKind` enums with the same name and six overlapping
/// variants, one of which -- `InputAttachment` -- only the backend had. Any
/// comparison between a layout the RHI described and one the backend built was
/// comparing distinct types, and the compiler was happy to allow it.
pub use chassis::DescriptorKind;

/// The driver type for an RHI descriptor kind.
///
/// A free function rather than an inherent method: `DescriptorKind` belongs to
/// `chassis`, and a crate cannot add methods to a type it does not define.
pub fn descriptor_type_of(kind: DescriptorKind) -> vk::DescriptorType {
    match kind {
        DescriptorKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        DescriptorKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        DescriptorKind::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        // The same descriptor as `SampledImage`, and the aspect is the image's
        // own -- `VulkanTypes.aspect` reads it from the format, not the kind.
        // Nothing about the layout is a choice here: the graph puts a read in
        // `SHADER_READ_ONLY_OPTIMAL` for a depth image too.
        DescriptorKind::SampledDepth => vk::DescriptorType::SAMPLED_IMAGE,
        DescriptorKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        DescriptorKind::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DescriptorKind::Sampler => vk::DescriptorType::SAMPLER,
        DescriptorKind::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
    }
}

/// A live `VkDescriptorSet` with the layout it was allocated against, and the
/// pool that owns it.
///
/// The pool matters because a set is freed through it, and because more than one
/// pool exists: a pool that has no room for a new set is left alone rather than
/// reset, since resetting it would free the sets live bind groups still hold.
#[derive(Clone, Copy, Debug)]
pub struct DescriptorSet {
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,
    pool: usize,
}

impl DescriptorSet {
    pub fn new(set: vk::DescriptorSet, layout: vk::DescriptorSetLayout, pool: usize) -> Self {
        Self { set, layout, pool }
    }

    pub fn handle(&self) -> vk::DescriptorSet {
        self.set
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn pool(&self) -> usize {
        self.pool
    }
}

/// Applies writes to a set.
///
/// The writes borrow their descriptors, so this is a borrow-scoped call rather
/// than a stored value: the caller has to keep the buffers and images alive
/// until it returns, which is exactly the constraint Vulkan imposes.
pub fn update(
    device: &ash::Device,
    set: DescriptorSet,
    buffer_writes: &[vk::WriteDescriptorSet<'_>],
    image_writes: &[vk::WriteDescriptorSet<'_>],
) -> Result<()> {
    let mut writes: Vec<vk::WriteDescriptorSet<'_>> =
        Vec::with_capacity(buffer_writes.len() + image_writes.len());
    writes.extend_from_slice(buffer_writes);
    writes.extend_from_slice(image_writes);

    for write in &mut writes {
        write.dst_set = set.handle();
    }

    if writes.is_empty() {
        return Ok(());
    }

    unsafe { device.update_descriptor_sets(&writes, &[]) };
    Ok(())
}

/// Vulkan guarantees at least this many bindings per set.
pub const MAX_BINDINGS: u32 = 128;

pub fn check_binding_count(count: usize) -> Result<()> {
    if count as u32 > MAX_BINDINGS {
        return Err(BackendError::TooManyBindings { count });
    }
    Ok(())
}

/// The Vulkan stage flags for a binding's declared stages.
///
/// One function for both things that build a `vk::DescriptorSetLayoutBinding`: a
/// pipeline's layout and a bind group's. Two copies is how a set ends up
/// incompatible with the pipeline it is bound to -- the stage flags are part of
/// the layout's definition, not a hint, so a set that says `ALL` where the
/// pipeline says `FRAGMENT` is a different layout.
pub fn stage_flags_of(mask: chassis::StageMask) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if mask.contains(chassis::ShaderStage::Vertex) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if mask.contains(chassis::ShaderStage::Fragment) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if mask.contains(chassis::ShaderStage::Compute) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    if mask.contains(chassis::ShaderStage::Geometry) {
        flags |= vk::ShaderStageFlags::GEOMETRY;
    }
    if mask.contains(chassis::ShaderStage::TessellationControl) {
        flags |= vk::ShaderStageFlags::TESSELLATION_CONTROL;
    }
    if mask.contains(chassis::ShaderStage::TessellationEvaluation) {
        flags |= vk::ShaderStageFlags::TESSELLATION_EVALUATION;
    }
    flags
}
