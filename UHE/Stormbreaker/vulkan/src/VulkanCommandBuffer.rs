//! The command buffer: raw recording operations, and the `chassis::CommandList`
//! adapter that resolves handles on top of them.
//!
//! Each worker records into its own buffer, so passes need no chaining and the
//! compiled order survives to the GPU. One shared primary buffer for the whole
//! frame would serialise every pass to its predecessor and defeat the DAG.
//!
//! The buffer also has **no barrier API**. Barriers arrive pre-baked from the
//! render graph, so no separate present-rendered handshake is needed.
//!
//! Two layers, deliberately:
//!
//! * [`Recording`]: raw Vulkan. Takes `vk::Pipeline`, `vk::Buffer`. No registry
//!   lookups, no locking, nothing that could stall a worker thread.
//! * [`CommandList`]: implements `chassis::CommandList`. Takes generational
//!   handles and resolves them through the resource registries.
//!
//! The split exists so the hot path does no handle resolution. A pass recording
//! on a worker thread should be able to hold its pipelines by raw handle for the
//! whole frame, and resolve `chassis::Pipeline` once when the pass is built.

use std::collections::HashMap;

use ash::{vk, vk::Handle as _, Device};
use chassis::Texture;

use crate::vulkan_barrier_encoder;
use crate::vulkan_descriptor_manager::Descriptors;
use crate::vulkan_error::Result;
use crate::vulkan_pipeline_state::Pipelines;
use crate::vulkan_texture::Textures;
use crate::vulkan_types;

/// A command buffer being recorded on one thread. Raw Vulkan only.
pub struct Recording {
    device: Device,
    buffer: vk::CommandBuffer,
    slot: u32,
    /// Inside `vkCmdBeginRendering` / `vkCmdEndRendering`, tracked so a
    /// mismatched pair is caught here rather than by the driver.
    rendering: bool,
    /// The layout of the pipeline bound right now. Null until one is bound.
    ///
    /// Here rather than on `CommandList`, because recording state is what this
    /// is: a bind group's validity is defined relative to the bound pipeline, so
    /// the thing that has to remember it is the thing that records the bind.
    layout: vk::PipelineLayout,
}

impl Recording {
    /// A recording must already be begun; `Recorder::take` does that.
    pub fn new(device: Device, buffer: vk::CommandBuffer, slot: u32) -> Self {
        Self {
            device,
            buffer,
            slot,
            rendering: false,
            layout: vk::PipelineLayout::null(),
        }
    }

    /// Gives the buffer back to the recorder. Consuming, because a recording must
    /// not outlive the frame that took it.
    pub fn into_buffer(self) -> vk::CommandBuffer {
        self.buffer
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.buffer
    }

    pub fn slot(&self) -> u32 {
        self.slot
    }

    // Rendering scope.

    /// Dynamic rendering. Attachments come from the graph's compiled pass, not
    /// from a `VkRenderPass`, which this backend has no concept of.
    ///
    /// No format parameter: under dynamic rendering the attachment formats are
    /// declared once in `VkPipelineRenderingCreateInfo` when the pipeline is
    /// compiled, and the frame loop cannot change them. A pass whose views do not
    /// match its pipeline's formats is a validation error, and that is the check.
    pub fn begin_rendering(
        &mut self,
        color_attachments: &[vk::RenderingAttachmentInfo],
        depth_attachment: Option<vk::RenderingAttachmentInfo>,
        extent: vk::Extent2D,
    ) {
        let info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            },
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: color_attachments.len() as u32,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: depth_attachment
                .as_ref()
                .map(|a| a as *const vk::RenderingAttachmentInfo)
                .unwrap_or(std::ptr::null()),
            p_stencil_attachment: std::ptr::null(),
            ..Default::default()
        };

        unsafe { self.device.cmd_begin_rendering(self.buffer, &info) };
        self.rendering = true;
    }

    pub fn end_rendering(&mut self) {
        debug_assert!(self.rendering, "end_rendering without begin_rendering");
        unsafe { self.device.cmd_end_rendering(self.buffer) };
        self.rendering = false;
    }

    // State.

    pub fn set_viewport(&mut self, extent: vk::Extent2D) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe { self.device.cmd_set_viewport(self.buffer, 0, &[viewport]) };
    }

    pub fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y as i32,
            },
            extent: vk::Extent2D { width, height },
        };
        unsafe { self.device.cmd_set_scissor(self.buffer, 0, &[scissor]) };
    }

    pub fn bind_pipeline(&mut self, pipeline: vk::Pipeline, layout: vk::PipelineLayout) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.buffer, vk::PipelineBindPoint::GRAPHICS, pipeline)
        };
        // Remembered, not discarded: `vkCmdBindDescriptorSets` must be given the
        // layout of the currently bound pipeline, so binding a group against a
        // null layout is a validation error.
        self.layout = layout;
    }

    pub fn bind_compute_pipeline(&mut self, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.buffer, vk::PipelineBindPoint::COMPUTE, pipeline)
        };
    }

    pub fn bind_group(&mut self, set: u32, group: vk::DescriptorSet, layout: vk::PipelineLayout) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.buffer,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                set,
                &[group],
                &[],
            );
        }
    }

    pub fn set_push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        data: &[u8],
    ) {
        unsafe {
            self.device
                .cmd_push_constants(self.buffer, layout, stages, 0, data);
        }
    }

    pub fn bind_vertex_buffer(&mut self, slot: u32, buffer: vk::Buffer, offset: vk::DeviceSize) {
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.buffer, slot, &[buffer], &[offset]);
        }
    }

    pub fn bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        format: vk::IndexType,
    ) {
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.buffer, buffer, offset, format)
        };
    }

    // Draws.

    pub fn draw(&mut self, vertices: u32, instances: u32) {
        unsafe { self.device.cmd_draw(self.buffer, vertices, instances, 0, 0) };
    }

    pub fn draw_indexed(&mut self, indices: u32, instances: u32) {
        unsafe {
            self.device
                .cmd_draw_indexed(self.buffer, indices, instances, 0, 0, 0)
        };
    }

    pub fn dispatch(&mut self, groups_x: u32, groups_y: u32, groups_z: u32) {
        unsafe {
            self.device
                .cmd_dispatch(self.buffer, groups_x, groups_y, groups_z)
        };
    }

    // Transfers.

    pub fn copy_buffer_to_buffer(
        &mut self,
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        unsafe { self.device.cmd_copy_buffer(self.buffer, src, dst, regions) };
    }

    /// The destination layout is taken explicitly: `GENERAL` is only legal when
    /// the image really is in `GENERAL`, and the render graph puts images in
    /// `TRANSFER_DST` or `SHADER_READ_ONLY`. The reverse copy takes the source
    /// layout for the same reason.
    pub fn copy_buffer_to_image(
        &mut self,
        src: vk::Buffer,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device
                .cmd_copy_buffer_to_image(self.buffer, src, dst, dst_layout, regions)
        };
    }

    pub fn copy_buffer_to_image_general(
        &mut self,
        src: vk::Buffer,
        dst: vk::Image,
        regions: &[vk::BufferImageCopy],
    ) {
        self.copy_buffer_to_image(src, dst, vk::ImageLayout::GENERAL, regions);
    }

    /// Image to host-visible buffer: the readback path.
    pub fn copy_image_to_buffer(
        &mut self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device
                .cmd_copy_image_to_buffer(self.buffer, src, src_layout, dst, regions)
        };
    }

    pub fn fill_buffer(&mut self, buffer: vk::Buffer, size: vk::DeviceSize, data: u32) {
        unsafe {
            self.device
                .cmd_fill_buffer(self.buffer, buffer, 0, size, data)
        };
    }

    // Barriers, lowered.

    pub fn barrier_image(&mut self, barriers: &[vk::ImageMemoryBarrier2<'static>]) {
        let dependency = vk::DependencyInfo::default()
            .dependency_flags(vk::DependencyFlags::empty())
            .image_memory_barriers(barriers);
        unsafe { self.device.cmd_pipeline_barrier2(self.buffer, &dependency) };
    }

    /// A legacy-format image barrier, for a device on the `Legacy` sync tier.
    ///
    /// `barrier_image` above is the Vulkan 1.3 spelling. This is the 1.0 one, and
    /// the two take different structs -- there is no conversion, which is the
    /// whole reason the tier exists.
    ///
    /// Stages are top and bottom of the pipe because the caller has said it has
    /// no memory dependency to express: the one place that uses this is the
    /// present transition, which is a layout change with nothing on either side.
    pub fn barrier_image_legacy(&mut self, barriers: &[vk::ImageMemoryBarrier<'static>]) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                barriers,
            );
        }
    }

    pub fn barrier_buffer(&mut self, barriers: &[vk::BufferMemoryBarrier2<'static>]) {
        let dependency = vk::DependencyInfo::default()
            .dependency_flags(vk::DependencyFlags::empty())
            .buffer_memory_barriers(barriers);
        unsafe { self.device.cmd_pipeline_barrier2(self.buffer, &dependency) };
    }

    pub fn is_inside_rendering(&self) -> bool {
        self.rendering
    }

    /// Ends recording. The buffer stays owned by the recorder until submit.
    pub fn end(self) -> Result<Recording> {
        unsafe { self.device.end_command_buffer(self.buffer)? };
        Ok(self)
    }
}

/// `chassis::CommandList` over a [`Recording`].
///
/// Holds the resource registries by reference so it can turn generational handles
/// into Vulkan handles. Callers resolve once when a pass is set up, not per draw.
pub struct CommandList<'a> {
    recording: &'a mut Recording,
    pipelines: &'a Pipelines,
    textures: &'a Textures,
    buffers: &'a crate::vulkan_buffer::Buffers,
    descriptors: &'a Descriptors,
    /// What the graph says each texture is in, per pass.
    ///
    /// Set by the executor from the compiled pass's declared read states, not
    /// tracked by mutating as commands are recorded. Per-pass is the right scope:
    /// lists record in parallel, so a shared running total would be a data race
    /// and a per-list total would not know what the *previous* list left behind.
    pub layouts: HashMap<Texture, chassis::graph::types::ImageState>,
}

impl<'a> CommandList<'a> {
    pub fn new(
        recording: &'a mut Recording,
        pipelines: &'a Pipelines,
        textures: &'a Textures,
        buffers: &'a crate::vulkan_buffer::Buffers,
        descriptors: &'a Descriptors,
    ) -> Self {
        Self {
            recording,
            pipelines,
            textures,
            buffers,
            descriptors,
            layouts: HashMap::new(),
        }
    }

    /// Tell the list what layout the graph says a texture is in.
    pub fn declare_layout(&mut self, texture: Texture, state: chassis::graph::types::ImageState) {
        self.layouts.insert(texture, state);
    }

    /// The raw recording, for anything `chassis` does not cover yet.
    pub fn raw(&mut self) -> &mut Recording {
        self.recording
    }
}

impl CommandList<'_> {
    /// A stale handle here means the buffer was destroyed while a pass still
    /// referenced it. Binding a null buffer would fail later and less clearly, so
    /// this panics at the call site instead.
    fn resolve_buffer(&self, handle: chassis::Buffer) -> vk::Buffer {
        self.buffers.get(handle).map_or_else(
            || panic!("bind on a stale buffer handle {handle:?}"),
            crate::vulkan_buffer::BufferResource::handle,
        )
    }
}

impl chassis::CommandList for CommandList<'_> {
    fn begin_rendering(&mut self, desc: &chassis::RenderPassDesc) {
        let color_attachments: Vec<vk::RenderingAttachmentInfo> = desc
            .colors
            .iter()
            .filter_map(|c| self.textures.get(c.texture).map(|t| (c, t)))
            .map(|(c, t)| vk::RenderingAttachmentInfo {
                image_view: t.view(),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                resolve_image_view: vk::ImageView::null(),
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: attachment_load(c.load),
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue { float32: c.clear },
                },
                ..Default::default()
            })
            .collect();
        let depth_attachment = desc
            .depth
            .as_ref()
            .and_then(|d| self.textures.get(d.texture).map(|t| (d, t)))
            .map(|(d, t)| vk::RenderingAttachmentInfo {
                image_view: t.view(),
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_image_view: vk::ImageView::null(),
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: attachment_load(d.load),
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: d.clear_depth,
                        stencil: 0,
                    },
                },
                ..Default::default()
            });
        // The render area is the first attachment's, colour or depth: a
        // depth-only pass -- the shadow map -- has no colour attachment and still
        // has an area. Falling back to a zero extent was
        // `VUID-VkRenderingInfo-None-08994`, reported as a render pass with no
        // area at all.
        let area = desc
            .colors
            .first()
            .map(|color| color.texture)
            .or_else(|| desc.depth.as_ref().map(|depth| depth.texture));
        let extent = area.and_then(|texture| self.textures.get(texture)).map_or(
            vk::Extent2D {
                width: 0,
                height: 0,
            },
            |t| vulkan_types::vk_extent_2d(t.extent()),
        );

        self.recording
            .begin_rendering(&color_attachments, depth_attachment, extent);
    }

    fn end_rendering(&mut self) {
        self.recording.end_rendering();
    }

    fn set_viewport(&mut self, extent: chassis::Extent2d) {
        self.recording.set_viewport(vk::Extent2D {
            width: extent.width,
            height: extent.height,
        });
    }

    fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.recording.set_scissor(x, y, width, height);
    }

    fn bind_pipeline(&mut self, pipeline: &chassis::Pipeline) {
        match self.pipelines.get(*pipeline) {
            Some(resource) => self
                .recording
                .bind_pipeline(resource.handle(), resource.layout()),
            // An unresolved handle means the pipeline was destroyed while a pass
            // still referenced it. Binding nothing would fail later and less
            // clearly, so this is reported now.
            None => panic!("bind_pipeline on a stale pipeline handle"),
        }
    }

    fn bind_group(&mut self, set: u32, group: &chassis::BindGroup) {
        // The layout is the bound pipeline's, remembered by `bind_pipeline`.
        let Some(handle) = self.descriptors.set(*group) else {
            // A stale handle. Panicking matches `resolve_buffer`: a bind against
            // a destroyed resource is a caller bug, and silently dropping the
            // bind would render the frame with the previous group's contents and
            // look like a shader bug.
            panic!("bind on a stale bind group handle {group:?}");
        };
        assert!(
            !self.recording.layout.is_null(),
            "bind_group with no pipeline bound: a descriptor set has no meaning \
             without the pipeline layout it was allocated against"
        );
        self.recording
            .bind_group(set, handle.handle(), self.recording.layout);
    }

    fn set_push_constants(&mut self, data: &[u8]) {
        self.recording.set_push_constants(
            vk::PipelineLayout::null(),
            vk::ShaderStageFlags::ALL,
            data,
        );
    }

    fn bind_vertex_buffer(&mut self, slot: u32, buffer: &chassis::Buffer, offset: u64) {
        let handle = self.resolve_buffer(*buffer);
        self.recording.bind_vertex_buffer(slot, handle, offset);
    }

    fn bind_index_buffer(&mut self, buffer: &chassis::Buffer, offset: u64) {
        let handle = self.resolve_buffer(*buffer);
        self.recording
            .bind_index_buffer(handle, offset, vk::IndexType::UINT32);
    }

    fn draw(&mut self, vertices: u32, instances: u32) {
        self.recording.draw(vertices, instances);
    }

    fn draw_indexed(&mut self, indices: u32, instances: u32) {
        self.recording.draw_indexed(indices, instances);
    }

    fn dispatch(&mut self, groups_x: u32, groups_y: u32, groups_z: u32) {
        self.recording.dispatch(groups_x, groups_y, groups_z);
    }

    fn copy_texture_to_buffer(&mut self, src: &chassis::Texture, dst: &chassis::Buffer) {
        let Some(texture) = self.textures.get(*src) else {
            panic!("copy_texture_to_buffer: source handle did not resolve");
        };
        let Some(buffer) = self.buffers.get(*dst) else {
            panic!("copy_texture_to_buffer: destination handle did not resolve");
        };

        // The layout comes from what the graph declared, not from a guess.
        // Guessing is how a copy reads a texture the previous pass left in
        // COLOR_ATTACHMENT_OPTIMAL, which validation reports as
        // VUID-vkCmdCopyImageToBuffer-srcImage-00113 and names no layout at all.
        let Some(state) = self.layouts.get(src).copied() else {
            panic!(
                "copy_texture_to_buffer: no layout was declared for this texture; \
                 the executor sets it from the graph's read state"
            );
        };
        let extent = texture.extent();
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            // The aligned pitch, in *texels* (`bufferRowLength` is texels, not
            // bytes). The same pitch the wgpu backend uses, so a caller reading
            // the buffer back strips the same padding either way.
            buffer_row_length: chassis::copy_row_pitch(extent.width, 4) / 4,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
        };
        self.recording.copy_image_to_buffer(
            texture.image(),
            vulkan_barrier_encoder::layout_for(state),
            buffer.handle(),
            &[region],
        );
    }

    fn copy_buffer(&mut self, src: &chassis::Buffer, dst: &chassis::Buffer) {
        let src = self.resolve_buffer(*src);
        let dst = self.resolve_buffer(*dst);
        self.recording.copy_buffer_to_buffer(
            src,
            dst,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: vk::WHOLE_SIZE,
            }],
        );
    }
}

/// The RHI's load op as Vulkan's. `DontCare` is safe to name here: the caller
/// declared it, and this backend carries no unsafe marker to launder.
fn attachment_load(load: chassis::LoadOp) -> vk::AttachmentLoadOp {
    match load {
        chassis::LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
        chassis::LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        chassis::LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}
