//! Host-to-device uploads.
//!
//! One ring per thread: its own staging buffer, command pool, command buffer and
//! fence. A shared pool and fence would serialise every upload across every
//! thread, and resetting that pool per call would drop an in-flight command
//! buffer; two threads on one pool would race.
//!
//! An upload records and returns; the caller decides when to wait. The ring
//! refuses to wrap while bytes the GPU has not read are still in the buffer, so
//! "uploaded while in flight" is an explicit wait rather than a corrupted frame.

use ash::vk;

use crate::vulkan_allocation;
use crate::vulkan_error::{BackendError, Result};

/// A per-thread staging ring.
pub struct UploadRing {
    device: ash::Device,
    queue: vk::Queue,
    family: u32,

    buffer: vk::Buffer,
    device_memory: vk::DeviceMemory,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    capacity: vk::DeviceSize,
    /// Write cursor within the ring.
    offset: vk::DeviceSize,
    /// True while a submit made from this ring has not been waited on.
    pending: bool,
}

impl UploadRing {
    pub fn new(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        queue: vk::Queue,
        family: u32,
        capacity: vk::DeviceSize,
    ) -> Result<Self> {
        let (buffer, device_memory, _) = vulkan_allocation::create_buffer(
            device,
            memory_properties,
            capacity,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // `RESET_COMMAND_BUFFER` because the ring re-records its one command
        // buffer after every submit: without it, the second upload is
        // `VUID-vkBeginCommandBuffer-commandBuffer-00050`.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(family);
        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(e) => {
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(device_memory, None);
                }
                return Err(BackendError::vk("vkCreateCommandPool", e));
            }
        };

        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = match unsafe { device.allocate_command_buffers(&allocate) } {
            Ok(mut buffers) => buffers.remove(0),
            Err(e) => {
                unsafe {
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_buffer(buffer, None);
                    device.free_memory(device_memory, None);
                }
                return Err(BackendError::vk("vkAllocateCommandBuffers", e));
            }
        };

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = match unsafe { device.create_fence(&fence_info, None) } {
            Ok(fence) => fence,
            Err(e) => {
                unsafe {
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_buffer(buffer, None);
                    device.free_memory(device_memory, None);
                }
                return Err(BackendError::vk("vkCreateFence", e));
            }
        };

        Ok(Self {
            device: device.clone(),
            queue,
            family,
            buffer,
            device_memory,
            command_pool,
            command_buffer,
            fence,
            capacity,
            offset: 0,
            pending: false,
        })
    }

    pub fn capacity(&self) -> vk::DeviceSize {
        self.capacity
    }

    /// Bytes handed to the GPU and not yet waited on.
    pub fn outstanding(&self) -> vk::DeviceSize {
        if self.pending {
            self.offset
        } else {
            0
        }
    }

    pub fn has_pending(&self) -> bool {
        self.pending
    }

    /// Records a copy of `data` into `dst`. Returns once the submit is queued,
    /// not once the GPU has read it.
    pub fn copy_to_buffer(
        &mut self,
        dst: vk::Buffer,
        dst_offset: vk::DeviceSize,
        data: &[u8],
    ) -> Result<()> {
        let size = data.len() as vk::DeviceSize;
        if size == 0 {
            return Ok(());
        }
        if size > self.capacity {
            return Err(BackendError::UploadTooLarge {
                size,
                capacity: self.capacity,
            });
        }

        // Wrapping would overwrite bytes the previous submit may still be
        // reading, so finish it first rather than corrupting it.
        if self.offset + size > self.capacity {
            self.wait()?;
        }

        unsafe {
            let ptr = self.device.map_memory(
                self.device_memory,
                self.offset,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(self.device_memory);
        }

        self.retire()?;

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let command_buffer = self.command_buffer;
        unsafe {
            self.device.begin_command_buffer(command_buffer, &begin)?;
            self.device.cmd_copy_buffer(
                command_buffer,
                self.buffer,
                dst,
                &[vk::BufferCopy {
                    src_offset: self.offset,
                    dst_offset,
                    size,
                }],
            );
            self.device.end_command_buffer(command_buffer)?;

            self.device.reset_fences(&[self.fence])?;
            let command_buffers = [command_buffer];
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: command_buffers.as_ptr(),
                    ..Default::default()
                }],
                self.fence,
            )?;
        }

        self.offset += size;
        self.pending = true;
        Ok(())
    }

    /// Copies CPU bytes into an image.
    ///
    /// ## The barrier is the whole point
    ///
    /// An image created with no layout is `UNDEFINED`, and `vkCmdCopyBufferToImage`
    /// requires the destination to be in
    /// `TRANSFER_DST_OPTIMAL` or `GENERAL`. It is *also* required by validation to
    /// name an old layout the image is actually in. So this transitions
    /// `UNDEFINED -> TRANSFER_DST_OPTIMAL` before the copy and
    /// `TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL` after, and the second
    /// transition is the one that matters: a texture left in
    /// `TRANSFER_DST_OPTIMAL` is a texture every later descriptor write refuses
    /// with `VUID-VkWriteDescriptorSet-descriptorType-04149`.
    ///
    /// Which is the shape of the bug this whole path exists to prevent: upload
    /// works, the texture is created, the bind group is created, and the draw is
    /// wrong because nobody moved the image out of its transfer layout.
    ///
    /// `UNDEFINED` on the way in discards whatever was there, which is always
    /// legal and is the only old layout that is legal for an image whose contents
    /// nobody has defined.
    pub fn copy_to_image(
        &mut self,
        dst: vk::Image,
        dst_extent: vk::Extent3D,
        _format: vk::Format,
        data: &[u8],
    ) -> Result<()> {
        let size = data.len() as vk::DeviceSize;
        if size == 0 {
            return Err(BackendError::InvalidSize);
        }
        if size > self.capacity {
            return Err(BackendError::UploadTooLarge {
                size,
                capacity: self.capacity,
            });
        }
        if self.offset + size > self.capacity {
            self.wait()?;
        }

        unsafe {
            let ptr = self.device.map_memory(
                self.device_memory,
                self.offset,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(self.device_memory);
        }

        let barrier = |old: vk::ImageLayout, new: vk::ImageLayout, access: vk::AccessFlags| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(access)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        };

        self.retire()?;

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let command_buffer = self.command_buffer;
        unsafe {
            self.device.begin_command_buffer(command_buffer, &begin)?;

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                )],
            );

            // The 1.0 spelling, and the layout is an explicit argument rather
            // than derived from the barrier we just recorded. Passing the wrong
            // one here is not caught by validation -- it silently produces a
            // frame where the texture is undefined.
            self.device.cmd_copy_buffer_to_image(
                command_buffer,
                self.buffer,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: self.offset,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: dst_extent,
                    ..Default::default()
                }],
            );

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::SHADER_READ,
                )],
            );

            self.device.end_command_buffer(command_buffer)?;

            self.device.reset_fences(&[self.fence])?;
            let command_buffers = [command_buffer];
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: command_buffers.as_ptr(),
                    ..Default::default()
                }],
                self.fence,
            )?;
        }

        self.offset += size;
        self.pending = true;
        Ok(())
    }

    /// Blocks until every queued upload has been read, then rewinds the ring.
    pub fn wait(&mut self) -> Result<()> {
        self.retire()?;
        self.offset = 0;
        Ok(())
    }

    /// Waits the previous submit out and makes the command buffer recordable again.
    ///
    /// One command buffer and one fence per ring, so neither can be reused while
    /// a submit is pending: re-beginning a pending buffer is
    /// `VUID-vkBeginCommandBuffer-commandBuffer-00049`, submitting one twice is
    /// `VUID-vkQueueSubmit-pCommandBuffers-00071`, and resetting a fence a submit
    /// still owns is `VUID-vkResetFences-pFences-01123`. All three are this call
    /// missing. Unlike [`UploadRing::wait`] it leaves the write cursor alone: the
    /// bytes just written are at the cursor, and rewinding here would copy from
    /// the wrong offset.
    fn retire(&mut self) -> Result<()> {
        if !self.pending {
            return Ok(());
        }
        unsafe {
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        self.pending = false;
        Ok(())
    }

    pub fn queue_family(&self) -> u32 {
        self.family
    }
}

impl Drop for UploadRing {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.wait_for_fences(&[self.fence], true, u64::MAX);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.device_memory, None);
        }
    }
}
