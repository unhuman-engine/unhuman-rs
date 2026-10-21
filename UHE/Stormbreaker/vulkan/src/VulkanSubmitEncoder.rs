//! Submit-chain construction.
//!
//! The only place in the engine that decides *how* an ordering edge is expressed.
//!
//! One API, two encodings, chosen by `SyncTier`:
//!
//! | tier | edge between list `i` and `i+1` | why |
//! |---|---|---|
//! | `Sync2` / `Sync2Timeline` | one timeline semaphore, value `i+1` gated on `i` | `vkQueueSubmit2` carries the value with the submit; timeline semaphores need Vulkan 1.2 |
//! | `Legacy` | one binary semaphore per link | that is all a 1.0 device has |
//!
//! Both express the same dependency. The caller never asks which, which is the
//! point: `Recorder::uses_timeline` exists only for diagnostics.
//!
//! ## Batched submits
//!
//! A caller with its own synchronisation -- the render graph, which emits a
//! barrier for every cross-pass dependency -- can put every list in one
//! `vkQueueSubmit2`/`vkQueueSubmit` and pay no semaphore at all. Command buffers
//! in one submit execute in submission order, and a barrier's first scope reaches
//! back across that order, so the graph's barriers are the ordering. This is
//! `batched`: no wait, no signal, one submit. A caller with no barriers (a raw
//! `Recorder::submit`) keeps the chain, which is the only thing ordering its
//! lists.

use ash::{vk, Device};

use crate::vulkan_error::{BackendError, Result};
use crate::vulkan_extension_check::SyncTier;

/// Chain inputs. The buffers are in compiled order.
pub struct Chain<'a> {
    pub device: &'a Device,
    pub queue: vk::Queue,
    pub fence: vk::Fence,
    pub buffers: &'a [vk::CommandBuffer],
    /// One timeline semaphore, when the tier has one.
    pub timeline: Option<vk::Semaphore>,
    /// One binary semaphore per link, otherwise. Indexed by the *waiting* list.
    pub binary_chain: &'a [vk::Semaphore],
    pub tier: SyncTier,
    /// The timeline value this chain starts from. Frames in flight advance it by
    /// the chain length.
    pub value_base: u64,
    /// One submit with no semaphores, for a caller that supplies its own
    /// synchronisation. See the module comment.
    pub batched: bool,
}

impl Chain<'_> {
    pub fn submit(&self) -> Result<()> {
        if self.buffers.is_empty() {
            return Ok(());
        }
        if self.batched {
            return match self.tier {
                SyncTier::Sync2 | SyncTier::Sync2Timeline => self.submit_sync2_batched(),
                SyncTier::Legacy => self.submit_legacy_batched(),
            };
        }
        match self.tier {
            SyncTier::Sync2 | SyncTier::Sync2Timeline => self.submit_sync2(),
            SyncTier::Legacy => self.submit_legacy(),
        }
    }

    /// One `vkQueueSubmit2` holding every list, with no semaphores.
    fn submit_sync2_batched(&self) -> Result<()> {
        let command_infos: Vec<vk::CommandBufferSubmitInfo> = self
            .buffers
            .iter()
            .map(|b| vk::CommandBufferSubmitInfo {
                command_buffer: *b,
                device_mask: 0,
                ..Default::default()
            })
            .collect();

        let submit = vk::SubmitInfo2 {
            command_buffer_info_count: command_infos.len() as u32,
            p_command_buffer_infos: command_infos.as_ptr(),
            ..Default::default()
        };

        unsafe { self.device.queue_submit2(self.queue, &[submit], self.fence) }
            .map_err(|e| BackendError::vk("vkQueueSubmit2", e))
    }

    /// One `vkQueueSubmit` holding every list, with no semaphores.
    fn submit_legacy_batched(&self) -> Result<()> {
        let submit = vk::SubmitInfo {
            command_buffer_count: self.buffers.len() as u32,
            p_command_buffers: self.buffers.as_ptr(),
            ..Default::default()
        };

        unsafe { self.device.queue_submit(self.queue, &[submit], self.fence) }
            .map_err(|e| BackendError::vk("vkQueueSubmit", e))
    }

    /// `vkQueueSubmit2`: one semaphore, values `base + 1 ..= base + n`.
    ///
    /// The values must *strictly increase*. VUID-VkSubmitInfo2-semaphore-03882
    /// requires a signal value greater than the semaphore's current value, which
    /// on entry is `value_base`. So the first link signals `value_base + 1`, not
    /// `value_base`.
    fn submit_sync2(&self) -> Result<()> {
        let timeline = self.timeline.ok_or(BackendError::NotTimeline)?;

        // Built first and never grown again: `SubmitInfo2` holds raw pointers
        // into these, so a reallocation would dangle them.
        let mut waits: Vec<vk::SemaphoreSubmitInfo> = Vec::with_capacity(self.buffers.len());
        let mut signals: Vec<vk::SemaphoreSubmitInfo> = Vec::with_capacity(self.buffers.len());
        for index in 0..self.buffers.len() {
            waits.push(if index == 0 {
                vk::SemaphoreSubmitInfo::default()
            } else {
                vk::SemaphoreSubmitInfo {
                    semaphore: timeline,
                    value: self.value_base + index as u64,
                    ..Default::default()
                }
            });
            signals.push(vk::SemaphoreSubmitInfo {
                semaphore: timeline,
                value: self.value_base + index as u64 + 1,
                ..Default::default()
            });
        }

        let command_infos: Vec<vk::CommandBufferSubmitInfo> = self
            .buffers
            .iter()
            .map(|b| vk::CommandBufferSubmitInfo {
                command_buffer: *b,
                device_mask: 0,
                ..Default::default()
            })
            .collect();

        let mut submits: Vec<vk::SubmitInfo2> = Vec::with_capacity(self.buffers.len());
        for index in 0..self.buffers.len() {
            let waits_slice: &[vk::SemaphoreSubmitInfo] = if index == 0 {
                &[]
            } else {
                &waits[index..index + 1]
            };
            let infos = &command_infos[index..index + 1];
            submits.push(vk::SubmitInfo2 {
                wait_semaphore_info_count: waits_slice.len() as u32,
                p_wait_semaphore_infos: waits_slice.as_ptr(),
                command_buffer_info_count: 1,
                p_command_buffer_infos: infos.as_ptr(),
                signal_semaphore_info_count: 1,
                p_signal_semaphore_infos: signals[index..index + 1].as_ptr(),
                ..Default::default()
            });
        }

        unsafe { self.device.queue_submit2(self.queue, &submits, self.fence) }
            .map_err(|e| BackendError::vk("vkQueueSubmit2", e))
    }

    /// `vkQueueSubmit`: one binary semaphore per link.
    ///
    /// The wait stage mask has to cover every stage the *waiting* list could be
    /// blocked on. A copy waits on nothing narrower than transfer, so the mask is
    /// deliberately broad; a narrow mask here is a race that shows up as torn
    /// frames on one driver and not another.
    fn submit_legacy(&self) -> Result<()> {
        if self.binary_chain.len() < self.buffers.len() {
            return Err(BackendError::ChainTooShort {
                semaphores: self.binary_chain.len(),
                lists: self.buffers.len(),
            });
        }

        let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut submits: Vec<vk::SubmitInfo> = Vec::with_capacity(self.buffers.len());

        for index in 0..self.buffers.len() {
            let waits: &[vk::Semaphore] = if index == 0 {
                &[]
            } else {
                &self.binary_chain[index - 1..index]
            };
            let signals: &[vk::Semaphore] = &self.binary_chain[index..index + 1];
            submits.push(vk::SubmitInfo {
                wait_semaphore_count: waits.len() as u32,
                p_wait_semaphores: waits.as_ptr(),
                p_wait_dst_stage_mask: if waits.is_empty() {
                    std::ptr::null()
                } else {
                    wait_stages.as_ptr()
                },
                command_buffer_count: 1,
                p_command_buffers: self.buffers[index..].as_ptr(),
                signal_semaphore_count: signals.len() as u32,
                p_signal_semaphores: signals.as_ptr(),
                ..Default::default()
            });
        }

        unsafe { self.device.queue_submit(self.queue, &submits, self.fence) }
            .map_err(|e| BackendError::vk("vkQueueSubmit", e))
    }
}
