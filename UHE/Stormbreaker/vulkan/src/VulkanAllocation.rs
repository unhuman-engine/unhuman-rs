//! Device memory and buffer/texture creation.
//!
//! Allocated directly against `vk::DeviceMemory` rather than through `vma`.
//! The `vma` crate at 0.3.1 exposes `destroy_buffer` and `destroy_image` but no
//! `new_buffer`, `new_slice` or `map_slice`, so it cannot allocate anything yet.
//! Everything here goes through `Allocator`, so dropping a working VMA binding in
//! later is a change to this file and nothing above it.

use ash::vk;

use crate::vulkan_error::{BackendError, Result};

/// Finds the cheapest memory type that satisfies `type_bits` and any required
/// property flags.
pub fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for index in 0..memory_properties.memory_type_count {
        if type_bits & (1 << index) == 0 {
            continue;
        }
        let properties = memory_properties.memory_types[index as usize];
        if properties.property_flags.contains(required) {
            return Ok(index);
        }
    }
    Err(BackendError::NoMemoryType { required })
}

pub fn create_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    required_properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, u64)> {
    if size == 0 {
        return Err(BackendError::InvalidSize);
    }

    let buffer_info = vk::BufferCreateInfo {
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_memory_type(
        memory_properties,
        requirements.memory_type_bits,
        required_properties,
    )
    .inspect_err(|_| {
        // Nothing else can succeed with this buffer, so do not leak it.
        unsafe { device.destroy_buffer(buffer, None) };
    })?;

    let allocate_info = vk::MemoryAllocateInfo {
        allocation_size: requirements.size,
        memory_type_index: memory_type,
        ..Default::default()
    };
    let memory = match unsafe { device.allocate_memory(&allocate_info, None) } {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(e.into());
        }
    };

    if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(e.into());
    }

    Ok((buffer, memory, requirements.size))
}

/// Binds an image to freshly allocated device memory.
pub fn create_image_memory(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    size: u64,
) -> Result<vk::DeviceMemory> {
    if size == 0 {
        return Err(BackendError::InvalidSize);
    }
    let type_index = find_memory_type(
        memory_properties,
        memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let allocate_info = vk::MemoryAllocateInfo {
        allocation_size: size,
        memory_type_index: type_index,
        ..Default::default()
    };
    Ok(unsafe { device.allocate_memory(&allocate_info, None)? })
}

/// Maps device memory and runs a closure over the slice.
///
/// # Safety
/// The caller must not touch the region while `f` runs, and must not pass a
/// buffer whose memory is `DEVICE_LOCAL`.
pub unsafe fn with_mapped_memory<T>(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    size: u64,
    flags: vk::MemoryMapFlags,
    f: impl FnOnce(&mut [u8]) -> T,
) -> Result<T> {
    let ptr = device.map_memory(memory, 0, size, flags)?;
    let result = f(std::slice::from_raw_parts_mut(
        ptr as *mut u8,
        size as usize,
    ));
    device.unmap_memory(memory);
    Ok(result)
}

/// Zero-copy read of a buffer created with `HOST_VISIBLE`.
pub fn read_buffer(device: &ash::Device, memory: vk::DeviceMemory, size: u64) -> Result<Vec<u8>> {
    unsafe {
        with_mapped_memory(device, memory, size, vk::MemoryMapFlags::empty(), |slice| {
            slice.to_vec()
        })
    }
}

/// Copies into a buffer created with `HOST_VISIBLE`.
pub fn write_buffer(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    size: u64,
    data: &[u8],
) -> Result<()> {
    write_buffer_at(device, memory, size, 0, data)
}

/// Copies `data` into a host-visible buffer at `offset`.
///
/// The whole buffer is mapped and the slice written into it: mapping a sub-range
/// would have to satisfy `minMemoryMapAlignment`, and the offsets a uniform arena
/// uses (256-aligned) are not guaranteed to.
pub fn write_buffer_at(
    device: &ash::Device,
    memory: vk::DeviceMemory,
    size: u64,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    if offset + data.len() as u64 > size {
        return Err(BackendError::InvalidSize);
    }
    unsafe {
        with_mapped_memory(device, memory, size, vk::MemoryMapFlags::empty(), |slice| {
            slice[offset as usize..offset as usize + data.len()].copy_from_slice(data);
        })
    }
}
