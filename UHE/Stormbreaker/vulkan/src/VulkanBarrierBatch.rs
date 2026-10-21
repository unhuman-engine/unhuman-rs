//! A pass's barriers, recorded as one driver call.
//!
//! Barriers are independent, so a pass that transitions five resources does not
//! need five `vkCmdPipelineBarrier2` calls. Each call is a validation-layer hook
//! and a driver entry point, and a frame pays that once per barrier per pass.
//!
//! The two spellings stay here rather than in the encoder because they differ in
//! a way the encoder's one-barrier `encode` does not: the legacy
//! `vkCmdPipelineBarrier` takes its synchronisation scope in command arguments,
//! not per barrier, so a batch has to union the stages where a single barrier
//! does not.

use ash::vk;
use chassis::graph::types::{BufferBarrier, ImageBarrier};
use chassis::TextureFormat;

use crate::vulkan_barrier_encoder as encode;
use crate::vulkan_types;

/// Accumulates a pass's barriers, then records them together.
pub struct BarrierBatch<'a> {
    encoder: &'a encode::BarrierEncoder,
    images2: Vec<vk::ImageMemoryBarrier2<'static>>,
    images_legacy: Vec<vk::ImageMemoryBarrier<'static>>,
    buffers2: Vec<vk::BufferMemoryBarrier2<'static>>,
    buffers_legacy: Vec<vk::BufferMemoryBarrier<'static>>,
    /// Legacy only: the union of the barriers' stages, since the command carries
    /// one scope for all of them.
    legacy_src: vk::PipelineStageFlags,
    legacy_dst: vk::PipelineStageFlags,
    count: u32,
}

impl<'a> BarrierBatch<'a> {
    pub fn new(encoder: &'a encode::BarrierEncoder) -> Self {
        Self {
            encoder,
            images2: Vec::new(),
            images_legacy: Vec::new(),
            buffers2: Vec::new(),
            buffers_legacy: Vec::new(),
            legacy_src: vk::PipelineStageFlags::empty(),
            legacy_dst: vk::PipelineStageFlags::empty(),
            count: 0,
        }
    }

    /// Adds an image transition, dropping the empty ones the graph emits.
    pub fn image(&mut self, target: encode::Target, barrier: &ImageBarrier, format: TextureFormat) {
        if barrier.is_empty() {
            return;
        }
        if self.encoder.tier().uses_sync2() {
            self.images2
                .push(encode::image_barrier2(target, *barrier, format));
        } else {
            self.legacy_src |= vulkan_types::stage_legacy(barrier.src_stage);
            self.legacy_dst |= vulkan_types::stage_legacy(barrier.dst_stage);
            self.images_legacy
                .push(encode::image_barrier_legacy(target, *barrier, format));
        }
        self.count += 1;
    }

    pub fn buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: u64,
        size: vk::DeviceSize,
        barrier: &BufferBarrier,
    ) {
        if self.encoder.tier().uses_sync2() {
            self.buffers2
                .push(encode::buffer_barrier2(buffer, offset, size, *barrier));
        } else {
            // A buffer barrier has no stage of its own here, so the legacy scope
            // is conservative. The single-barrier path did the same.
            self.legacy_src |= vk::PipelineStageFlags::ALL_COMMANDS;
            self.legacy_dst |= vk::PipelineStageFlags::ALL_COMMANDS;
            self.buffers_legacy.push(encode::buffer_barrier_legacy(
                buffer, offset, size, *barrier,
            ));
        }
        self.count += 1;
    }

    /// Records everything accumulated and returns how many barriers it held.
    pub fn flush(self, command_buffer: vk::CommandBuffer) -> u32 {
        if self.count == 0 {
            return 0;
        }
        if self.encoder.tier().uses_sync2() {
            let dependency = vk::DependencyInfo::default()
                .dependency_flags(vk::DependencyFlags::empty())
                .image_memory_barriers(&self.images2)
                .buffer_memory_barriers(&self.buffers2);
            unsafe {
                self.encoder
                    .device()
                    .cmd_pipeline_barrier2(command_buffer, &dependency);
            }
        } else {
            unsafe {
                self.encoder.device().cmd_pipeline_barrier(
                    command_buffer,
                    self.legacy_src,
                    self.legacy_dst,
                    vk::DependencyFlags::empty(),
                    &[],
                    &self.buffers_legacy,
                    &self.images_legacy,
                );
            }
        }
        self.count
    }
}
