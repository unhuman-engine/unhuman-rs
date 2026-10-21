//! Per-frame GPU data, ringed by the frame that last used it.
//!
//! The backend's frame ring lets several submissions overlap. A frontend that
//! keeps one buffer and one bind group per frame would then overwrite memory the
//! GPU is still reading, so every slot here is reused only after the submission
//! that last wrote it retired.

use chassis::{Error, Gpu, Submission};

/// Frames the renderer keeps in flight when a backend reports no depth.
const DEFAULT_FRAMES_IN_FLIGHT: usize = 3;

/// The ring depth a backend reports, with a sane fallback for `0`.
pub(crate) fn frame_depth(reported: u32) -> usize {
    match reported {
        0 => DEFAULT_FRAMES_IN_FLIGHT,
        depth => depth as usize,
    }
}

/// Per-frame GPU data ringed by the frame that last used it.
///
/// Slot `f % depth` was last touched by frame `f - depth`; `Bifrost::render`
/// waits for that frame before touching the slot again.
pub(crate) struct FrameRing<T> {
    slots: Vec<T>,
}

impl<T: Default> FrameRing<T> {
    pub(crate) fn new(depth: usize) -> Self {
        assert!(depth > 0, "a frame ring with no slots cannot render");
        Self {
            slots: (0..depth).map(|_| T::default()).collect(),
        }
    }

    pub(crate) fn depth(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn get(&self, index: usize) -> &T {
        &self.slots[index]
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> &mut T {
        &mut self.slots[index]
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut()
    }
}

impl FrameRing<Option<Submission>> {
    /// Waits for the submission that last wrote `index`, if it exists.
    ///
    /// Taking the token is deliberate: after this returns the slot is retired,
    /// so a frame that fails before submitting leaves nothing to wait on.
    pub(crate) fn wait(&mut self, gpu: &mut dyn Gpu, index: usize) -> Result<(), Error> {
        let Some(submission) = self.slots[index].take() else {
            return Ok(());
        };
        gpu.wait_for_submission(submission)
    }

    pub(crate) fn wait_all(&mut self, gpu: &mut dyn Gpu) -> Result<(), Error> {
        for index in 0..self.slots.len() {
            self.wait(gpu, index)?;
        }
        Ok(())
    }

    pub(crate) fn record(&mut self, index: usize, submission: Submission) {
        self.slots[index] = Some(submission);
    }
}
