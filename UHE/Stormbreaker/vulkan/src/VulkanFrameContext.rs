//! Per-thread command list recording and graph-ordered submission.
//!
//! This is the orchestrator: it owns the per-list pools from `command_pool.rs`,
//! hands out buffers wrapped as `vulkan_command_buffer::Recording`, and turns a
//! batch of finished recordings into a submission through `submit_encoder.rs`.
//!
//! Recording is parallel and submission is serial, which is what Vulkan
//! requires: every worker owns a pool and a buffer and records with no lock, and
//! submission then walks the *compiled* order. The spec's external-sync object
//! for recording is the pool, not the buffer, so one pool per worker makes
//! concurrent recording sound.

use std::sync::Arc;

use ash::{vk, Device};

use crate::vulkan_command_buffer::Recording;
use crate::vulkan_command_pool;
use crate::vulkan_error::{BackendError, Result};
use crate::vulkan_extension_check::SyncTier;
use crate::vulkan_submission_ring::{SubmissionRing, SubmissionToken, FRAMES_IN_FLIGHT};
use crate::vulkan_submit_encoder::Chain;

/// Secondary buffers, allocated once and recycled.
pub struct Recorder {
    device: Device,
    queue: vk::Queue,
    family: u32,
    tier: SyncTier,

    /// One pool per list. Slot `i` belongs to pool `i`, and the contract is that
    /// list `i` is recorded on thread `i`.
    pools: Vec<vk::CommandPool>,
    /// `(slot, buffer)`. The slot is the pool index, so it is a stable identity
    /// rather than a per-frame position.
    free: Vec<(u32, vk::CommandBuffer)>,
    /// Buffers submitted on each fence slot, held until that fence signals.
    ///
    /// A buffer here is still in the queue's hands. Returning it to `free` before
    /// the fence completes lets the next frame reset a buffer the GPU is reading.
    pending: Vec<Vec<(u32, vk::CommandBuffer)>>,

    /// One timeline semaphore carries the whole chain when the tier has one.
    timeline: Option<vk::Semaphore>,
    /// Otherwise one binary semaphore per link.
    chain: Vec<vk::Semaphore>,
    /// The fence ring every submit path shares.
    submissions: Arc<SubmissionRing>,
}

impl Recorder {
    pub fn new(
        device: &Device,
        queue: vk::Queue,
        family: u32,
        tier: SyncTier,
        max_lists: usize,
    ) -> Result<Self> {
        if max_lists == 0 {
            return Err(BackendError::ListArenaExhausted {
                requested: 1,
                available: 0,
            });
        }

        let mut pools = Vec::with_capacity(max_lists);
        let mut free = Vec::with_capacity(max_lists);
        for slot in 0..max_lists {
            match vulkan_command_pool::create(device, family)
                .and_then(|pool| vulkan_command_pool::allocate(device, pool).map(|buf| (pool, buf)))
            {
                Ok((pool, buffer)) => {
                    pools.push(pool);
                    free.push((slot as u32, buffer));
                }
                Err(e) => {
                    for existing in pools.drain(..) {
                        vulkan_command_pool::reset_and_destroy(device, existing);
                    }
                    return Err(e);
                }
            }
        }

        let submissions = match SubmissionRing::new(device, FRAMES_IN_FLIGHT) {
            Ok(ring) => Arc::new(ring),
            Err(e) => {
                for existing in pools.drain(..) {
                    vulkan_command_pool::reset_and_destroy(device, existing);
                }
                return Err(e);
            }
        };

        let mut timeline = None;
        let mut chain = Vec::new();
        if tier.supports_timeline_semaphores() {
            let type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo {
                p_next: &type_info as *const _ as *const _,
                ..Default::default()
            };
            match unsafe { device.create_semaphore(&info, None) } {
                Ok(s) => timeline = Some(s),
                Err(e) => {
                    for existing in pools.drain(..) {
                        vulkan_command_pool::reset_and_destroy(device, existing);
                    }
                    return Err(BackendError::vk("vkCreateSemaphore", e));
                }
            }
        } else {
            let info = vk::SemaphoreCreateInfo::default();
            for _ in 0..max_lists {
                match unsafe { device.create_semaphore(&info, None) } {
                    Ok(s) => chain.push(s),
                    Err(e) => {
                        for existing in pools.drain(..) {
                            vulkan_command_pool::reset_and_destroy(device, existing);
                        }
                        return Err(BackendError::vk("vkCreateSemaphore", e));
                    }
                }
            }
        }

        Ok(Self {
            device: device.clone(),
            queue,
            family,
            tier,
            pools,
            free,
            pending: vec![Vec::new(); FRAMES_IN_FLIGHT],
            timeline,
            chain,
            submissions,
        })
    }

    /// The pool backing list `slot`, for callers allocating their own buffers.
    pub fn pool_for_slot(&self, slot: u32) -> Option<vk::CommandPool> {
        self.pools.get(slot as usize).copied()
    }

    /// How many lists the arena holds. One per recording thread.
    pub fn capacity(&self) -> usize {
        self.pools.len()
    }

    /// The fence ring, shared with `VulkanResources` for readback waits.
    pub fn submissions(&self) -> Arc<SubmissionRing> {
        Arc::clone(&self.submissions)
    }

    /// Hands out one recording per list, in ascending slot order.
    ///
    /// Fails rather than truncating: a frame that asked for more lists than the
    /// arena holds would otherwise silently drop passes.
    ///
    /// Blocks only when the arena is short, and then only on the oldest pending
    /// fence, because those are the submissions that can return a buffer. A
    /// buffer whose fence has not signalled is never handed out.
    pub fn take(&mut self, count: usize) -> Result<Vec<Recording>> {
        if count > self.pools.len() {
            return Err(BackendError::ListArenaExhausted {
                requested: count,
                available: self.pools.len(),
            });
        }
        self.make_room(count)?;
        if count > self.free.len() {
            return Err(BackendError::ListArenaExhausted {
                requested: count,
                available: self.free.len(),
            });
        }

        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let (slot, buffer) = self.free.pop().expect("checked above");
            let info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                // A recycled buffer is in the executable state after its last
                // submit; re-recording without a reset is invalid usage.
                self.device
                    .reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty())?;
                self.device.begin_command_buffer(buffer, &info)?;
            }
            out.push(Recording::new(self.device.clone(), buffer, slot));
        }
        out.sort_by_key(Recording::slot);
        Ok(out)
    }

    /// Returns buffers whose fence has signalled, without blocking.
    fn release_completed(&mut self) -> usize {
        let completed = self.submissions.completed();
        completed
            .into_iter()
            .map(|index| self.release_slot(index))
            .sum()
    }

    fn release_slot(&mut self, index: usize) -> usize {
        let buffers = std::mem::take(&mut self.pending[index]);
        let count = buffers.len();
        self.free.extend(buffers);
        count
    }

    /// Waits for one pending slot, oldest first, and recycles it.
    ///
    /// "Oldest" is the slot the ring reuses next, not the first non-empty one in
    /// Vec order: waiting the newest would stall the CPU on work it does not
    /// need yet.
    fn wait_one_pending(&mut self) -> Result<bool> {
        let depth = self.pending.len();
        let start = self.submissions.next_index();
        for offset in 0..depth {
            let index = (start + offset) % depth;
            if self.pending[index].is_empty() {
                continue;
            }
            self.submissions.wait_index(index)?;
            self.release_slot(index);
            return Ok(true);
        }
        Ok(false)
    }

    fn make_room(&mut self, count: usize) -> Result<()> {
        while self.free.len() < count {
            self.release_completed();
            if self.free.len() >= count {
                break;
            }
            if !self.wait_one_pending()? {
                break;
            }
        }
        Ok(())
    }

    /// Submits in slot order, each waiting on its predecessor.
    ///
    /// Returns the token that names the fence this submission signals. The call
    /// does not wait for that fence; a caller that needs the result calls
    /// [`Recorder::wait`] with the token.
    ///
    /// `value_base` is the timeline value this chain starts from; frames in
    /// flight advance it by the chain length.
    ///
    /// Use this when the lists carry no barriers between them: the semaphore
    /// chain is the only thing ordering one against the next. A caller with its
    /// own barriers wants [`Recorder::submit_batched`].
    pub fn submit(
        &mut self,
        recordings: Vec<Recording>,
        value_base: u64,
    ) -> Result<SubmissionToken> {
        self.submit_inner(recordings, value_base, false)
    }

    /// Submits every list in one `vkQueueSubmit2`/`vkQueueSubmit`, no semaphores.
    ///
    /// The render graph's barriers order its passes, so the per-link semaphore
    /// chain is redundant work: one submit is a single queue operation instead of
    /// one per pass, and no semaphore signal/wait per link. Only correct when the
    /// lists are already ordered by their own barriers.
    pub fn submit_batched(&mut self, recordings: Vec<Recording>) -> Result<SubmissionToken> {
        self.submit_inner(recordings, 0, true)
    }

    fn submit_inner(
        &mut self,
        recordings: Vec<Recording>,
        value_base: u64,
        batched: bool,
    ) -> Result<SubmissionToken> {
        let mut ordered = recordings;
        ordered.sort_by_key(Recording::slot);
        if ordered.is_empty() {
            return Ok(SubmissionToken::NONE);
        }
        let buffers: Vec<vk::CommandBuffer> = ordered.iter().map(Recording::handle).collect();

        // Waits the fence about to be reused, then resets it. This is the only
        // fence the new submit may safely signal: the other two still belong to
        // submissions the CPU has not caught up with.
        let token = self.submissions.acquire()?;
        let index = token.index() as usize;
        self.release_slot(index);

        let chain = Chain {
            device: &self.device,
            queue: self.queue,
            fence: self.submissions.handle(token)?,
            buffers: &buffers,
            timeline: self.timeline,
            binary_chain: &self.chain,
            tier: self.tier,
            value_base,
            batched,
        };
        if let Err(error) = chain.submit() {
            // The fence was reset but no submission will signal it. Recreate it
            // signalled, or the next reuse would wait forever.
            self.submissions.abandon(token)?;
            for recording in ordered {
                self.free.push((recording.slot(), recording.into_buffer()));
            }
            return Err(error);
        }

        // The queue owns these buffers until the fence signals. They go back to
        // `free` from `release_slot` on a later lap, never before.
        self.pending[index] = ordered
            .into_iter()
            .map(|recording| (recording.slot(), recording.into_buffer()))
            .collect();
        Ok(token)
    }

    /// Waits for exactly the submission this token names.
    ///
    /// Buffers that fence owned come back to the arena as part of the wait; a
    /// caller that only waits through `VulkanResources` leaves them for the next
    /// `take`, which polls the same fences.
    pub fn wait(&mut self, token: SubmissionToken) -> Result<()> {
        self.submissions.wait(token)?;
        self.release_completed();
        Ok(())
    }

    /// Whether the submission this token names is still on the GPU.
    pub fn is_pending(&self, token: SubmissionToken) -> bool {
        self.submissions.is_pending(token)
    }

    /// Hand command buffers back without submitting them.
    ///
    /// For the path where a frame fails after `take`: the buffers were begun and
    /// have to be ended, and then they have to go back on the free list or the
    /// arena shrinks by one every failed frame. Without this the buffers are
    /// still allocated when the pool is destroyed, which validation reports as
    /// VUID-vkDestroyDevice-device-05137 and the driver answers with a
    /// segfault.
    pub fn recycle(&mut self, recordings: Vec<Recording>) {
        for recording in recordings {
            self.free.push((recording.slot(), recording.into_buffer()));
        }
        self.free.sort_by_key(|(slot, _)| *slot);
    }

    /// Command buffers not owned by an in-flight submission.
    ///
    /// Falls as `take` succeeds and rises when a fence signals or `recycle`
    /// returns an unsubmitted list. A number that keeps falling is a leak.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn tier(&self) -> SyncTier {
        self.tier
    }

    /// The device the recorder submits on.
    ///
    /// Public because an executor building a `BarrierEncoder` needs it, and the
    /// barrier encoder needs a device reference that outlives one frame.
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue_family(&self) -> u32 {
        self.family
    }

    /// True when the chain rides one timeline semaphore rather than one binary
    /// semaphore per link. Diagnostics only; callers do not branch on it.
    pub fn uses_timeline(&self) -> bool {
        self.timeline.is_some()
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // Every submitted command buffer still belongs to the queue until its
        // fence signals. Resetting the pools first would free buffers in use.
        let _ = self.submissions.wait_all();
        unsafe {
            if let Some(timeline) = self.timeline {
                self.device.destroy_semaphore(timeline, None);
            }
            for semaphore in self.chain.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
        for pool in self.pools.drain(..) {
            vulkan_command_pool::reset_and_destroy(&self.device, pool);
        }
    }
}
