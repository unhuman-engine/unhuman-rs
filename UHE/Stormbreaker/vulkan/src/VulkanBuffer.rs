//! Buffers.
//!
//! The handle is a typed generational handle from `chassis::Registry`, so a
//! buffer cannot be passed where a texture belongs and a freed handle stops
//! resolving.

use ash::vk;
use chassis::{Buffer, BufferDesc, BufferUsage, Registry};

use crate::vulkan_allocation;
use crate::vulkan_error::{BackendError, Result};

pub struct BufferResource {
    buffer: vk::Buffer,
    device_memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    host_visible: bool,
    label: String,
}

impl BufferResource {
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn is_host_visible(&self) -> bool {
        self.host_visible
    }

    /// Reads the whole buffer. Only valid when it was created host-visible;
    /// mapping a `DEVICE_LOCAL` allocation is undefined.
    pub fn read(&self, device: &ash::Device) -> Result<Vec<u8>> {
        if !self.host_visible {
            return Err(BackendError::NotHostVisible {
                label: self.label.clone(),
            });
        }
        vulkan_allocation::read_buffer(device, self.device_memory, self.size)
    }

    pub fn write(&self, device: &ash::Device, data: &[u8]) -> Result<()> {
        self.write_at(device, 0, data)
    }

    /// Writes `data` at `offset`. Only valid when the buffer is host-visible; a
    /// buffer created with `TRANSFER_DST` is.
    pub fn write_at(&self, device: &ash::Device, offset: u64, data: &[u8]) -> Result<()> {
        if !self.host_visible {
            return Err(BackendError::NotHostVisible {
                label: self.label.clone(),
            });
        }
        vulkan_allocation::write_buffer_at(device, self.device_memory, self.size, offset, data)
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.device_memory, None);
        }
    }
}

/// Owns every buffer and hands out generational handles.
pub struct Buffers {
    device: ash::Device,
    registry: Registry<BufferResource, chassis::BufferTag>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Buffers {
    pub fn new(
        device: &ash::Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        Self {
            device: device.clone(),
            registry: Registry::new(),
            memory_properties,
        }
    }

    pub fn create(
        &mut self,
        device: &ash::Device,
        desc: &BufferDesc,
        initial: Option<&[u8]>,
    ) -> Result<Buffer> {
        let usage = to_vk_usage(desc.usage);
        if usage.is_empty() {
            return Err(BackendError::EmptyUsage { what: "buffer" });
        }

        // Host-visible when the CPU has to touch it, which is two cases and not
        // one.
        //
        // A staging buffer is written by the CPU, and a buffer created *with*
        // data is too -- `write` maps it. Keying only off `TRANSFER_DST` missed
        // the second: a vertex buffer created with initial contents was given
        // DEVICE_LOCAL memory and then could not be written at all. The caller
        // asked for data in the constructor and got an error instead.
        //
        // Otherwise DEVICE_LOCAL, because paying for the slower host path on a
        // buffer the CPU never touches is a real regression on tiled hardware.
        let host_accessible = desc.usage.contains(BufferUsage::TRANSFER_DST) || initial.is_some();
        let required = if host_accessible {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };

        // A readback buffer is read by the CPU every frame. Without HOST_CACHED
        // RADV hands back uncached memory on this APU and the 9 MB copy runs at
        // ~250 MB/s, which cost more than the whole frame. Cached is a
        // preference because not every device offers it; coherent keeps the
        // read legal without an explicit invalidate.
        let cached = required | vk::MemoryPropertyFlags::HOST_CACHED;
        let wants_cache = host_accessible && desc.usage.contains(BufferUsage::MAP_READ);
        let (buffer, device_memory, allocated) = if wants_cache {
            match vulkan_allocation::create_buffer(
                device,
                &self.memory_properties,
                desc.size,
                usage,
                cached,
            ) {
                Ok(created) => created,
                Err(BackendError::NoMemoryType { .. }) => vulkan_allocation::create_buffer(
                    device,
                    &self.memory_properties,
                    desc.size,
                    usage,
                    required,
                )?,
                Err(error) => return Err(error),
            }
        } else {
            vulkan_allocation::create_buffer(
                device,
                &self.memory_properties,
                desc.size,
                usage,
                required,
            )?
        };

        let host_visible = required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let resource = BufferResource {
            buffer,
            device_memory,
            size: desc.size,
            host_visible,
            label: format!("buffer({} bytes, {allocated} allocated)", desc.size),
        };

        let handle = self.registry.create(resource);

        if let Some(data) = initial {
            if let Err(e) = self
                .registry
                .get(handle)
                .expect("just created")
                .write(device, data)
            {
                self.registry.remove(handle);
                return Err(e);
            }
        }

        Ok(handle)
    }

    /// A persistent upload buffer: host-visible, coherent, transfer destination.
    pub fn create_staging(&mut self, device: &ash::Device, size: u64) -> Result<Buffer> {
        self.create(
            device,
            &BufferDesc::new(size, BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC),
            None,
        )
    }

    pub fn get(&self, handle: Buffer) -> Option<&BufferResource> {
        self.registry.get(handle)
    }

    pub fn get_mut(&mut self, handle: Buffer) -> Option<&mut BufferResource> {
        self.registry.get_mut(handle)
    }

    pub fn destroy(&mut self, device: &ash::Device, handle: Buffer) -> bool {
        match self.registry.remove(handle) {
            Some(resource) => {
                resource.destroy(device);
                true
            }
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.registry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Buffer, &BufferResource)> {
        self.registry.iter()
    }
}

/// Owning handles means destroying them. Leaving that to the caller leaks every
/// buffer and trips VUID-vkDestroyDevice-device-05137 at device teardown.
impl Drop for Buffers {
    fn drop(&mut self) {
        for (_, resource) in self.registry.iter() {
            resource.destroy(&self.device);
        }
    }
}

pub fn to_vk_usage(usage: BufferUsage) -> vk::BufferUsageFlags {
    let mut out = vk::BufferUsageFlags::empty();
    if usage.contains(BufferUsage::VERTEX) {
        out |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDEX) {
        out |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        out |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(BufferUsage::STORAGE) {
        out |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(BufferUsage::TRANSFER_SRC) {
        out |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(BufferUsage::TRANSFER_DST) {
        out |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        out |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    out
}
