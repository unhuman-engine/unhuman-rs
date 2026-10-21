//! Job system and task graph.
//!
//! This crate schedules CPU-side work only. Physics, the frame's submission and
//! the GPU upload are main-thread by construction -- the drivers' upload paths
//! are not thread-safe -- so they run on the calling thread and must not be
//! moved into a job. A job here is the other kind of work: a texture decode, a
//! mesh extract, a range of a parallel loop.
//!
//! Two knobs shape scheduling:
//!
//! - [`Priority`] separates a frame's recording ([`Priority::High`]) from an
//!   asset decode ([`Priority::Low`]). A frame that is recording is the frame
//!   that is waiting; a decode can be a frame late.
//! - [`CancelToken`] stops work whose reason is gone. A load for a scene that
//!   was closed is cancelled before it starts; a job already running is not
//!   interrupted mid-decode, but its result can be discarded by the caller.
//!
//! Two memory-ordering invariants are load-bearing: the decrement release
//! sequence in `JobCounter::dec` and the final decrement in `resolve`. Both are
//! documented at their definitions.

mod jobs;
mod taskgraph;

pub use jobs::{CancelToken, Job, JobCounter, Jobs, Priority};
pub use taskgraph::{TaskGraph, TaskId};
