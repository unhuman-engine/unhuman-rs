//! Host-side fences.
//!
//! One fence per frame in flight; the binary primitive shared with semaphores
//! lives in `semaphore.rs`.

use std::time::Duration;

use ash::vk;

use crate::vulkan_error::{BackendError, Result};

/// One fence per frame in flight.
pub struct Fence {
    device: ash::Device,
    fence: vk::Fence,
    name: String,
}

impl Fence {
    /// Created signalled, so the first wait cannot block on a fence nobody has
    /// signalled yet.
    pub fn new(device: &ash::Device, name: impl Into<String>) -> Result<Self> {
        let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&info, None) }
            .map_err(|e| BackendError::vk("vkCreateFence", e))?;
        Ok(Self {
            device: device.clone(),
            fence,
            name: name.into(),
        })
    }

    pub fn handle(&self) -> vk::Fence {
        self.fence
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_signalled(&self) -> bool {
        // NOT_READY is `Ok(false)`, not an error; `is_ok` would call it signalled.
        matches!(
            unsafe { self.device.get_fence_status(self.fence) },
            Ok(true)
        )
    }

    /// Blocks until signalled. Returns false on timeout.
    ///
    /// Prefer a timeout on the frame path: `wait_forever` on a fence that will
    /// never signal is how a frame hangs with no diagnostic.
    pub fn wait(&self, timeout: Duration) -> Result<bool> {
        let nanos = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        match unsafe { self.device.wait_for_fences(&[self.fence], true, nanos) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(BackendError::vk("vkWaitForFences", e)),
        }
    }

    /// Blocks indefinitely. Only correct at shutdown, or where the caller already
    /// knows the work completed.
    pub fn wait_forever(&self) -> Result<()> {
        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| BackendError::vk("vkWaitForFences", e))
    }

    pub fn reset(&self) -> Result<()> {
        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|e| BackendError::vk("vkResetFences", e))
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.fence, None) };
    }
}
