//! A ring of fences tracking queue submissions.
//!
//! One fence per submission would grow without bound; one fence for every
//! submission serialises the CPU with the GPU. A ring of [`FRAMES_IN_FLIGHT`]
//! fences is the middle: a submit picks the next fence, waits only that fence
//! (the one about to be reused), and returns without waiting for the frame it
//! just submitted.
//!
//! Waiting on the fence about to be reused is what makes reuse legal. A fence
//! that a submission is still going to signal must not be reset, and the oldest
//! fence in the ring is the one whose submission is guaranteed to be behind the
//! two new submissions the ring can still accept.
//!
//! The token is `(index, generation)`. A generation is bumped every time its
//! slot is reused, so a token from an earlier lap can be distinguished from the
//! current submission on the same fence and answered without waiting.

use std::sync::Mutex;

use ash::{vk, Device};

use crate::vulkan_error::{BackendError, Result};

/// Fence ring depth.
///
/// Three, matching the renderer's frame ring: the oldest submission is waited
/// before its slot is reused, so two submissions may be in flight behind the
/// CPU. More would buy nothing for a frontend that reuses a slot every lap;
/// fewer would make the wait land on a submission the queue just accepted.
pub const FRAMES_IN_FLIGHT: usize = 3;

/// Which fence a submission signals, and which lap of the ring it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubmissionToken {
    index: u32,
    generation: u64,
}

impl SubmissionToken {
    /// A token for a frame that submitted nothing.
    ///
    /// Waiting on it waits for every fence, which is the only answer that is
    /// correct when the caller's write could have been made by an earlier
    /// submission.
    pub const NONE: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    pub fn index(self) -> u32 {
        self.index
    }

    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn is_none(self) -> bool {
        self.index == u32::MAX
    }
}

struct Slot {
    fence: vk::Fence,
    generation: u64,
}

struct State {
    slots: Vec<Slot>,
    next: usize,
}

/// The fences and the cursor that picks the next one.
pub struct SubmissionRing {
    device: Device,
    state: Mutex<State>,
}

impl SubmissionRing {
    pub fn new(device: &Device, depth: usize) -> Result<Self> {
        assert!(depth > 0, "a fence ring with no fences cannot submit");
        let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut slots = Vec::with_capacity(depth);
        for _ in 0..depth {
            match unsafe { device.create_fence(&info, None) } {
                Ok(fence) => slots.push(Slot {
                    fence,
                    generation: 0,
                }),
                Err(error) => {
                    for slot in slots {
                        unsafe { device.destroy_fence(slot.fence, None) };
                    }
                    return Err(BackendError::vk("vkCreateFence", error));
                }
            }
        }
        Ok(Self {
            device: device.clone(),
            state: Mutex::new(State { slots, next: 0 }),
        })
    }

    pub fn depth(&self) -> usize {
        self.state.lock().expect("fence ring lock").slots.len()
    }

    /// The slot the next submit will reuse; the oldest submission's fence.
    pub fn next_index(&self) -> usize {
        self.state.lock().expect("fence ring lock").next
    }

    /// Reserves the next fence and returns the token its submission will answer.
    ///
    /// Waits on that fence before resetting and submitting: the previous
    /// submission on this slot is the only in-flight owner of it, and resetting
    /// a fence a queued submit is about to signal is invalid usage.
    pub fn acquire(&self) -> Result<SubmissionToken> {
        let mut state = self.state.lock().expect("fence ring lock");
        let depth = state.slots.len();
        let index = advance(&mut state.next, depth);

        let slot = &mut state.slots[index];
        unsafe {
            self.device
                .wait_for_fences(&[slot.fence], true, u64::MAX)
                .map_err(|error| BackendError::vk("vkWaitForFences", error))?;
            self.device
                .reset_fences(&[slot.fence])
                .map_err(|error| BackendError::vk("vkResetFences", error))?;
        }
        slot.generation += 1;
        Ok(SubmissionToken {
            index: index as u32,
            generation: slot.generation,
        })
    }

    /// The fence a token was acquired for.
    pub fn handle(&self, token: SubmissionToken) -> Result<vk::Fence> {
        let state = self.state.lock().expect("fence ring lock");
        state
            .slots
            .get(token.index as usize)
            .map(|slot| slot.fence)
            .ok_or_else(|| BackendError::Unsupported {
                what: format!("submission fence {} does not exist", token.index),
            })
    }

    /// Waits for exactly the submission the token names.
    ///
    /// An older generation has already been waited on, because reusing a slot
    /// waits before it resets. A newer generation is a token from the future and
    /// is refused rather than silently waited on the wrong fence.
    pub fn wait(&self, token: SubmissionToken) -> Result<()> {
        if token.is_none() {
            return self.wait_all();
        }
        let state = self.state.lock().expect("fence ring lock");
        let Some(slot) = state.slots.get(token.index as usize) else {
            return Err(BackendError::Unsupported {
                what: format!("submission fence {} does not exist", token.index),
            });
        };
        if token.generation > slot.generation {
            return Err(BackendError::Unsupported {
                what: format!(
                    "submission token generation {} is newer than fence {}'s {}",
                    token.generation, token.index, slot.generation
                ),
            });
        }
        if token.generation == slot.generation {
            unsafe {
                self.device
                    .wait_for_fences(&[slot.fence], true, u64::MAX)
                    .map_err(|error| BackendError::vk("vkWaitForFences", error))?;
            }
        }
        Ok(())
    }

    /// Waits for every fence in the ring.
    pub fn wait_all(&self) -> Result<()> {
        let state = self.state.lock().expect("fence ring lock");
        let fences: Vec<vk::Fence> = state.slots.iter().map(|slot| slot.fence).collect();
        unsafe {
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(|error| BackendError::vk("vkWaitForFences", error))?;
        }
        Ok(())
    }

    /// Fences whose submissions have completed, without blocking.
    ///
    /// The command buffers submitted with each one can be handed back only
    /// after this says so.
    pub fn completed(&self) -> Vec<usize> {
        let state = self.state.lock().expect("fence ring lock");
        state
            .slots
            .iter()
            .enumerate()
            // `get_fence_status` returns `Ok(false)` for NOT_READY, so `is_ok`
            // would call an in-flight fence complete.
            .filter(|(_, slot)| {
                matches!(
                    unsafe { self.device.get_fence_status(slot.fence) },
                    Ok(true)
                )
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// Waits on the fence at `index`.
    ///
    /// Used when the command-list arena is short: the caller knows which slot
    /// owns the buffers it needs back.
    pub fn wait_index(&self, index: usize) -> Result<()> {
        let state = self.state.lock().expect("fence ring lock");
        let Some(slot) = state.slots.get(index) else {
            return Err(BackendError::Unsupported {
                what: format!("submission fence {index} does not exist"),
            });
        };
        unsafe {
            self.device
                .wait_for_fences(&[slot.fence], true, u64::MAX)
                .map_err(|error| BackendError::vk("vkWaitForFences", error))?;
        }
        Ok(())
    }

    /// Waits for the next slot to be reused, then reports it.
    ///
    /// Used when the arena is short of free lists: the next slot is the oldest
    /// submission, so waiting on it is the one wait that can free something.
    pub fn wait_next(&self) -> Result<usize> {
        let index = {
            let state = self.state.lock().expect("fence ring lock");
            state.next
        };
        self.wait_index(index)?;
        Ok(index)
    }

    /// Restores a fence whose submit failed before anything could signal it.
    ///
    /// The fence was reset on acquire; leaving it would make the next reuse wait
    /// forever. A fresh signalled fence is the same state the ring started in.
    pub fn abandon(&self, token: SubmissionToken) -> Result<()> {
        if token.is_none() {
            return Ok(());
        }
        let mut state = self.state.lock().expect("fence ring lock");
        let Some(slot) = state.slots.get_mut(token.index as usize) else {
            return Err(BackendError::Unsupported {
                what: format!("submission fence {} does not exist", token.index),
            });
        };
        if slot.generation != token.generation {
            return Ok(());
        }
        let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { self.device.create_fence(&info, None) }
            .map_err(|error| BackendError::vk("vkCreateFence", error))?;
        unsafe { self.device.destroy_fence(slot.fence, None) };
        slot.fence = fence;
        Ok(())
    }

    /// Whether a token still names work the GPU has not finished.
    pub fn is_pending(&self, token: SubmissionToken) -> bool {
        if token.is_none() {
            return false;
        }
        let state = self.state.lock().expect("fence ring lock");
        state.slots.get(token.index as usize).is_some_and(|slot| {
            slot.generation == token.generation
                && matches!(
                    unsafe { self.device.get_fence_status(slot.fence) },
                    Ok(false)
                )
        })
    }
}

impl Drop for SubmissionRing {
    fn drop(&mut self) {
        let _ = self.wait_all();
        let state = self.state.get_mut().expect("fence ring lock");
        for slot in state.slots.drain(..) {
            unsafe { self.device.destroy_fence(slot.fence, None) };
        }
    }
}

/// The cursor arithmetic the ring uses to pick its oldest fence.
fn advance(next: &mut usize, depth: usize) -> usize {
    let index = *next;
    *next = (*next + 1) % depth;
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cursor visits every slot before returning to the first. Anything else
    /// could hand a fence to two submissions before either has signalled.
    #[test]
    fn the_ring_walks_every_slot_before_reusing_one() {
        let mut next = 0;
        let mut visited = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            visited.push(advance(&mut next, FRAMES_IN_FLIGHT));
        }
        assert_eq!(visited, (0..FRAMES_IN_FLIGHT).collect::<Vec<_>>());
        assert_eq!(next, 0, "the ring must come back to its oldest slot");
    }

    #[test]
    fn a_token_from_the_previous_lap_is_not_the_current_submission() {
        let old = SubmissionToken {
            index: 0,
            generation: 1,
        };
        let current = SubmissionToken {
            index: 0,
            generation: 2,
        };
        assert_ne!(old, current);
        assert_eq!(old.index(), current.index());
        assert!(current.generation() > old.generation());
    }
}
