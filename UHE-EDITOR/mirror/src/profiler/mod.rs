//! The profiler's rows, and the timer that keeps the expensive ones honest.
//!
//! The loop's period is not the engine's frame: the editor's own paint and the
//! toolkit sit in the same period, and the engine's frame contains the
//! renderer's record/submit plus the readback wait. They are separate fields
//! here, so one number cannot be read as another.
//!
//! CPU, memory and GPU load are read from the OS. Those reads cost more than a
//! frame can spare, so they happen on a timer and the last sample is carried
//! forward; the panel never triggers a read per paint.

mod system;

use std::time::{Duration, Instant};

pub use system::{SystemLoad, SystemSampler};

/// How often the OS rows are sampled. Sixteen frames at 60 Hz: quick enough to
/// see a change, slow enough not to become part of the frame.
pub const DEFAULT_SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

/// The loop's period and the engine's frame, in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// Wall time between two loop ticks: everything the user waited.
    pub loop_ms: f32,
    /// The engine's frame (`advance_frame`), renderer included.
    pub engine_ms: f32,
    /// The renderer's record and submit, inside the engine's frame.
    pub render_ms: f32,
    /// The readback wait the engine frame paid. `None` when it cannot be timed
    /// apart from the render, which is the synchronous readback today.
    pub readback_ms: Option<f32>,
}

impl Timings {
    /// The loop time the engine did not use: the UI's own paint.
    pub fn ui_ms(&self) -> f32 {
        (self.loop_ms - self.engine_ms).max(0.0)
    }

    /// The synchronisation stall: waiting for the GPU's readback.
    pub fn sync_ms(&self) -> Option<f32> {
        self.readback_ms
    }
}

/// Counters `bifrost` already keeps for a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub draw_calls: u32,
    pub passes: u32,
    pub instances: u32,
}

/// One GPU pass's time.
///
/// The name is owned because graph passes are rebuilt with fresh names every
/// frame; an empty list means timing is unavailable, not that passes are free.
#[derive(Clone, Debug, PartialEq)]
pub struct PassTime {
    pub pass: String,
    pub ms: f32,
}

/// One frame as the panel shows it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameStats {
    pub timings: Timings,
    pub counters: Counters,
    pub system: SystemLoad,
    pub gpu_passes: Vec<PassTime>,
}

impl FrameStats {
    /// `false` means no timestamp queries, not that every pass is free.
    pub fn has_gpu_timings(&self) -> bool {
        !self.gpu_passes.is_empty()
    }
}

/// The loop's period and its smoothed reciprocal, which is the status bar's
/// existing pair. The engine's frame is [`FrameStats::timings`].
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Stats {
    /// The loop's period, not the engine's frame.
    pub frame_ms: f32,
    pub fps: f32,
}

impl Stats {
    pub fn update(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.frame_ms = dt * 1000.0;
        // Smoothed, because a raw reciprocal flickers too fast to read.
        let instantaneous = 1.0 / dt;
        self.fps = if self.fps <= 0.0 {
            instantaneous
        } else {
            self.fps * 0.9 + instantaneous * 0.1
        };
    }
}

/// The frame's rows, filled by the editor and sampled by the profiler.
pub struct Profiler {
    stats: Stats,
    frame: FrameStats,
    system: SystemSampler,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self::with_sample_interval(DEFAULT_SAMPLE_INTERVAL)
    }

    pub fn with_sample_interval(interval: Duration) -> Self {
        Self {
            stats: Stats::default(),
            frame: FrameStats::default(),
            system: SystemSampler::new(interval),
        }
    }

    /// Starts a loop tick: updates the loop's period and fps. The engine's own
    /// timing is recorded later, after `advance_frame`.
    pub fn begin_frame(&mut self, dt: f32) {
        self.stats.update(dt);
    }

    /// Publishes the rows the editor timed and samples the OS rows if the timer
    /// is due. The loop period comes from [`Profiler::begin_frame`], not from
    /// `timings`, so there is one answer to that question.
    pub fn record(&mut self, timings: Timings, counters: Counters, now: Instant) -> &FrameStats {
        self.frame.timings = timings;
        self.frame.counters = counters;
        self.publish(now)
    }

    /// Samples the OS rows and keeps the timings already recorded. The editor
    /// calls this every frame even before it has engine timings to offer.
    pub fn publish(&mut self, now: Instant) -> &FrameStats {
        self.frame.timings.loop_ms = self.stats.frame_ms;
        self.frame.system = self.system.poll(now);
        &self.frame
    }

    pub fn frame(&self) -> &FrameStats {
        &self.frame
    }

    /// The rows the editor fills directly: render counters and the GPU pass
    /// times the renderer collected for this frame.
    pub fn frame_mut(&mut self) -> &mut FrameStats {
        &mut self.frame
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// OS samples taken so far, which the timer's tests count.
    pub fn samples(&self) -> u64 {
        self.system.samples()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loop_period_and_the_engine_frame_are_different_rows() {
        let mut profiler = Profiler::new();
        profiler.begin_frame(0.007);
        let timings = Timings {
            loop_ms: 0.0,
            engine_ms: 4.0,
            render_ms: 3.0,
            readback_ms: Some(1.0),
        };
        let frame = profiler.record(timings, Counters::default(), Instant::now());

        assert!((frame.timings.loop_ms - 7.0).abs() < 1e-3);
        assert_eq!(frame.timings.engine_ms, 4.0);
        assert_eq!(frame.timings.render_ms, 3.0);
        assert_eq!(frame.timings.ui_ms(), 3.0);
        assert_eq!(frame.timings.sync_ms(), Some(1.0));
    }

    #[test]
    fn ui_paint_never_goes_negative() {
        let timings = Timings {
            loop_ms: 4.0,
            engine_ms: 9.0,
            ..Default::default()
        };
        assert_eq!(timings.ui_ms(), 0.0);
    }

    #[test]
    fn counters_are_carried_through() {
        let mut profiler = Profiler::new();
        let counters = Counters {
            draw_calls: 12,
            passes: 7,
            instances: 40,
        };
        let frame = profiler.record(Timings::default(), counters, Instant::now());
        assert_eq!(frame.counters, counters);
    }

    #[test]
    fn gpu_pass_timings_are_a_seam_not_a_zero() {
        let frame = FrameStats::default();
        assert!(!frame.has_gpu_timings());
        assert!(frame.gpu_passes.is_empty());
    }

    #[test]
    fn the_expensive_rows_are_sampled_on_a_timer() {
        let interval = Duration::from_millis(250);
        let mut profiler = Profiler::with_sample_interval(interval);
        let start = Instant::now();

        profiler.record(Timings::default(), Counters::default(), start);
        assert_eq!(profiler.samples(), 1);

        profiler.record(
            Timings::default(),
            Counters::default(),
            start + Duration::from_millis(16),
        );
        assert_eq!(profiler.samples(), 1);

        profiler.record(Timings::default(), Counters::default(), start + interval);
        assert_eq!(profiler.samples(), 2);
    }

    #[test]
    fn a_frame_can_be_filled_through_frame_mut() {
        let mut profiler = Profiler::new();
        profiler.frame_mut().counters.draw_calls = 3;
        profiler.frame_mut().gpu_passes.push(PassTime {
            pass: "opaque".to_owned(),
            ms: 1.5,
        });
        assert_eq!(profiler.frame().counters.draw_calls, 3);
        assert!(profiler.frame().has_gpu_timings());
    }
}
