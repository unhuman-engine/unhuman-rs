//! Logical device creation.
//!
//! Feature enabling mirrors the probe in `vulkan_physical_device.rs`, with the
//! same constraint: the 1.2 node only appears on the chain when the device
//! reported 1.2 or newer. Enabling a feature through a struct the device does not
//! know is a validation error, and on some drivers a device loss.

use ash::{vk, Device, Instance};

use crate::vulkan_error::{BackendError, Result};
use crate::vulkan_extension_check::{Capabilities, QueueFamily, SyncTier};
use crate::vulkan_physical_device::{self, Probe};

/// What a successful logical-device creation yields.
pub struct LogicalDevice {
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub queue_family: u32,
    pub capabilities: Capabilities,
}

pub fn create(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    probe: Probe,
    families: &[QueueFamily],
    check: &crate::vulkan_extension_check::ExtensionCheck,
) -> Result<LogicalDevice> {
    let api_version = properties.api_version;
    let sync_tier = sync_tier_for(api_version, probe);

    let graphics_family = families
        .iter()
        .find(|f| f.graphics)
        .map(|f| f.index)
        .ok_or(BackendError::NoGraphicsQueue)?;

    // The extension list mirrors what the tier will actually use, so a legacy
    // path never carries an unused extension.
    let mut extension_ptrs: Vec<*const std::os::raw::c_char> = Vec::new();
    if sync_tier.supports_timeline_semaphores() {
        if !check.supports_timeline_semaphores() {
            return Err(BackendError::ExtensionUnavailable {
                name: vulkan_physical_device::TIMELINE_SEMAPHORE.into(),
                required_for: "Sync2Timeline",
            });
        }
        extension_ptrs.push(c"VK_KHR_timeline_semaphore".as_ptr());
    }

    // `VK_KHR_swapchain`, whenever it is on offer.
    //
    // It was not requested at all before, so `Swapchain::new` built an
    // `ash::khr::swapchain::Device` that could not load `vkAcquireNextImageKHR`
    // and the windowed path aborted. The tier comment above applies here too, but
    // the reason this one is unconditional is different: there is no tier that
    // renders without presenting.
    //
    // Only if offered. A compute-only driver without it must still be able to
    // make a headless device, and failing at `vkCreateDevice` because of an
    // extension nothing has asked for yet would make the backend unusable
    // rather than merely windowless.
    if vulkan_physical_device::has_device_extension(instance, physical_device, "VK_KHR_swapchain") {
        extension_ptrs.push(c"VK_KHR_swapchain".as_ptr());
    }

    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(graphics_family)
        .queue_priorities(&queue_priorities);

    // Enable chain, same ordering rule as the probe. These locals must outlive
    // `vkCreateDevice`; the driver copies the values during the call.
    let mut enable_v12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(sync_tier.supports_timeline_semaphores())
        .descriptor_indexing(probe.descriptor_indexing);
    let mut enable_v13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(sync_tier.supports_sync2())
        .dynamic_rendering(probe.dynamic_rendering);

    let base_features = vk::PhysicalDeviceFeatures::default();
    let queue_infos = [queue_info];
    let mut device_info = vk::DeviceCreateInfo {
        p_enabled_features: &base_features,
        ..Default::default()
    }
    .queue_create_infos(&queue_infos)
    .enabled_extension_names(&extension_ptrs);

    if vulkan_physical_device::at_least(api_version, 1, 3) {
        enable_v13.p_next = (&raw mut enable_v12).cast::<std::os::raw::c_void>();
        device_info.p_next = (&raw mut enable_v13).cast::<std::os::raw::c_void>();
    }

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .map_err(|e| BackendError::vk("vkCreateDevice", e))?;
    let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };

    let capabilities = Capabilities {
        device_name: vulkan_physical_device::read_device_name(&properties.device_name),
        api_version,
        driver_version: properties.driver_version,
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        sync_tier,
        queue_families: families.to_vec(),
        graphics_queue_family: Some(graphics_family),
        dynamic_rendering: probe.dynamic_rendering,
        memory_budget: check.supports_memory_budget(),
        descriptor_indexing: probe.descriptor_indexing,
        max_texture_dimension_2d: properties.limits.max_image_dimension2_d,
        timestamp_period: properties.limits.timestamp_period,
        min_uniform_buffer_offset_alignment: properties.limits.min_uniform_buffer_offset_alignment,
    };

    Ok(LogicalDevice {
        device,
        graphics_queue,
        queue_family: graphics_family,
        capabilities,
    })
}

/// The tier, from the cached check rather than re-deriving it.
///
/// `select_sync_tier` is still the ladder -- one implementation, tested as a pure
/// function -- and `ExtensionCheck::sync_tier` is how a device asks for it. This
/// wrapper is what keeps the two from drifting: it passes the *probe* results,
/// which on a 1.3 device come from the feature structs rather than from strings,
/// and that is the better answer of the two.
fn sync_tier_for(api_version: u32, probe: Probe) -> SyncTier {
    crate::vulkan_extension_check::select_sync_tier(
        api_version,
        probe.synchronization2,
        probe.timeline_semaphore,
    )
}
