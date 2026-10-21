//! The readback pass, and the ring of buffers a headless frame reads through.
//!
//! A headless frame cannot present, so the composed image is copied into a
//! readable buffer and the CPU reads it. Reading the buffer a frame just
//! submitted blocks that frame on the GPU, so the frontend keeps a ring instead:
//! the frame copies into slot `f % N`, and reads slot `f - (N-1)`, which the GPU
//! retired a submission ago.
//!
//! ## How the wait stays scoped
//!
//! Each frame's `Gpu::record_frame` returns a [`Submission`], stored in the ring
//! slot it wrote. Reading a slot then calls
//! [`Resources::read_buffer_after`], which waits only for that frame's
//! submission. wgpu lowers the token to
//! `device.poll(PollType::Wait { submission_index })`; Vulkan's submit already
//! waited on its fence, so the token is retrospective and the read is immediate.
//!
//! ## Who owns the throttle
//!
//! The renderer's frame ring, not this one. It waits for the oldest frame before
//! reusing a slot; [`READBACK_LAG`] equals that ring's depth, so the slot read
//! here was retired by the same wait and does not stall again. Both waits name a
//! submission already made, so neither can deadlock the other.

use chassis::graph::PassScope;
use chassis::{
    Buffer, BufferDesc, BufferUsage, CommandList, Error, Resources, Submission, Texture,
};

/// Readback buffers in the ring. One more than the frame ring's depth, so the
/// slot being read is the one the frame ring has just retired.
pub const READBACK_RING: usize = 4;

/// Frames between a buffer's write and its read. Equal to the frame ring's
/// depth, so the frame-ring wait covers it.
pub const READBACK_LAG: u64 = READBACK_RING as u64 - 1;

/// Copies the composed image into a readable buffer, for a caller with no
/// swapchain to present to.
///
/// A pass rather than a side channel because the graph owns the transition: the
/// copy needs the image in `TransferSrc`, and declaring the read is what emits
/// that barrier. `copy_texture_to_buffer` on its own would read whichever layout
/// the composition left behind.
pub struct ReadbackPass {
    pub source: Texture,
    pub target: Buffer,
}

impl ReadbackPass {
    pub fn record(&self, _scope: &PassScope<'_>, list: &mut dyn CommandList) -> Result<(), Error> {
        list.copy_texture_to_buffer(&self.source, &self.target);
        Ok(())
    }
}

/// The buffers a sequence of frames reads back through.
///
/// All [`READBACK_RING`] are created once, at `size` -- a packed frame, because
/// the render width is a multiple of 64 texels -- and reused until the extent
/// changes and the ring is rebuilt.
pub struct ReadbackRing {
    buffers: Vec<Buffer>,
    /// The submission that wrote each slot, so a read can wait for just that one.
    submissions: Vec<Option<Submission>>,
    /// Frames the ring has served; the write slot is `frames % READBACK_RING`.
    frames: u64,
}

impl ReadbackRing {
    /// Allocates the ring's buffers, each `size` bytes.
    pub fn new(resources: &mut dyn Resources, size: u64) -> Result<Self, Error> {
        let mut buffers = Vec::with_capacity(READBACK_RING);
        for _ in 0..READBACK_RING {
            // `MAP_READ` is what wgpu needs to map it; Vulkan ignores the flag and
            // makes the buffer host-visible because it is a transfer destination.
            match resources.create_buffer(
                BufferDesc::new(size, BufferUsage::TRANSFER_DST | BufferUsage::MAP_READ),
                None,
            ) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    // A partial ring is not a ring: drop what was made rather
                    // than leaking the handles behind a failed construction.
                    for buffer in buffers.drain(..) {
                        resources.destroy_buffer(buffer);
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self {
            submissions: vec![None; buffers.len()],
            buffers,
            frames: 0,
        })
    }

    /// The buffer this frame's readback copies into, or `None` if the ring is
    /// empty (`READBACK_RING == 0`, only reachable through a bad constant).
    pub fn target(&self) -> Option<Buffer> {
        self.buffers
            .get((self.frames % READBACK_RING as u64) as usize)
            .copied()
    }

    /// Remembers the submission that wrote this frame's slot.
    ///
    /// Call once per frame, after [`Self::target`] and with the token
    /// `record_frame` returned.
    pub fn record_submission(&mut self, submission: Submission) {
        let slot = (self.frames % READBACK_RING as u64) as usize;
        if let Some(entry) = self.submissions.get_mut(slot) {
            *entry = Some(submission);
        }
    }

    /// The submission that wrote `frame`'s slot, if the ring still holds it.
    pub fn submission(&self, frame: u64) -> Option<Submission> {
        self.submissions
            .get((frame % READBACK_RING as u64) as usize)
            .and_then(Clone::clone)
    }

    /// The buffer written [`READBACK_LAG`] submissions ago, and the frame it
    /// holds, or `None` while the ring is still warming up.
    pub fn readable(&self) -> Option<(Buffer, u64)> {
        if self.frames < READBACK_LAG {
            return None;
        }
        let frame = self.frames - READBACK_LAG;
        self.buffers
            .get((frame % READBACK_RING as u64) as usize)
            .copied()
            .map(|buffer| (buffer, frame))
    }

    /// Moves the write slot on. One call per frame that used [`Self::target`].
    pub fn advance(&mut self) {
        self.frames += 1;
    }

    pub fn destroy(self, resources: &mut dyn Resources) {
        for buffer in self.buffers {
            resources.destroy_buffer(buffer);
        }
    }
}
