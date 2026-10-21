//! Per-pass GPU timestamps, for the benchmark and for a puzzled frame.
//!
//! A query pool with one pair per pass: the first timestamp after the pass's
//! pre-barriers, the second after its post-barriers. Timestamps are recorded
//! into each pass's own command buffer, which is legal because the pool is a
//! device object and the lists are submitted together in compiled order.
//!
//! The whole-frame pair (queries 0 and 1) is recorded in the first and last
//! list. A pass's time includes its post-barrier because that is work the frame
//! pays for at that point; the pre-barrier is included in the previous pass.

use ash::{vk, Device};

use crate::vulkan_error::{BackendError, Result};

/// One pass's GPU time, named by the graph's own pass name.
pub use chassis::GpuPassTime as GpuPassTiming;

/// Query pairs: one for the frame, one for every pass the recorder can hand out.
const QUERY_CAPACITY: u32 = 2 + 2 * 64;

/// Timestamp queries for one executor.
pub struct GpuProfiler {
    device: Device,
    query_pool: vk::QueryPool,
    period: f32,
    valid_bits: u32,
    names: Vec<String>,
    timings: Vec<GpuPassTiming>,
    frame_ms: Option<f32>,
}

/// The handles a recording thread needs. Copyable, so it can go into a
/// `'static` closure without borrowing the executor.
#[derive(Clone, Copy)]
pub struct PassQueries {
    pub pool: vk::QueryPool,
    pub sync2: bool,
    pub capacity: u32,
}

impl GpuProfiler {
    /// `None` when the queue cannot write timestamps; profiling is then off
    /// rather than reporting zero.
    pub fn new(device: &Device, period: f32, valid_bits: u32) -> Option<Self> {
        if valid_bits == 0 || period <= 0.0 {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(QUERY_CAPACITY);
        let query_pool = unsafe { device.create_query_pool(&info, None) }.ok()?;
        Some(Self {
            device: device.clone(),
            query_pool,
            period,
            valid_bits,
            names: Vec::new(),
            timings: Vec::new(),
            frame_ms: None,
        })
    }

    /// The copyable handles recording threads use.
    pub fn queries(&self, sync2: bool) -> PassQueries {
        PassQueries {
            pool: self.query_pool,
            sync2,
            capacity: QUERY_CAPACITY,
        }
    }

    /// Starts a frame. The caller has waited the previous submissions, so the
    /// reset cannot race a query write still on the GPU.
    pub fn begin_frame(&mut self, pass_names: &[&str]) {
        self.names = pass_names.iter().map(|name| (*name).to_owned()).collect();
        self.timings.clear();
        self.frame_ms = None;
    }

    /// Reads the pair every pass wrote. The caller must have waited the
    /// submission; there is no `WAIT` flag because a benchmark should not hide
    /// the wait it is measuring.
    pub fn collect(&mut self) -> Result<()> {
        let count = self.names.len().min(64) as u32;
        let queries = 2 + count * 2;
        let mut values = vec![0u64; queries as usize];
        unsafe {
            self.device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut values,
                    vk::QueryResultFlags::TYPE_64,
                )
                .map_err(|error| BackendError::vk("vkGetQueryPoolResults", error))?;
        }

        let frame_start = self.ticks(values[0]);
        let frame_end = self.ticks(values[1]);
        self.frame_ms = Some(self.elapsed_ms(frame_start, frame_end) as f32);

        self.timings.clear();
        for (index, name) in self.names.iter().enumerate().take(count as usize) {
            let start = self.ticks(values[2 + index * 2]);
            let end = self.ticks(values[3 + index * 2]);
            self.timings.push(GpuPassTiming {
                name: name.clone(),
                ms: self.elapsed_ms(start, end) as f32,
            });
        }
        Ok(())
    }

    pub fn timings(&self) -> &[GpuPassTiming] {
        &self.timings
    }

    /// GPU time of the whole frame, from the first pass's pre-barrier to the
    /// last pass's post-barrier.
    pub fn frame_ms(&self) -> Option<f32> {
        self.frame_ms
    }

    fn ticks(&self, value: u64) -> u64 {
        if self.valid_bits >= 64 {
            value
        } else {
            value & ((1u64 << self.valid_bits) - 1)
        }
    }

    fn elapsed_ms(&self, start: u64, end: u64) -> f64 {
        let wrap = if self.valid_bits >= 64 {
            0
        } else {
            1u64 << self.valid_bits
        };
        let ticks = if end >= start {
            end - start
        } else {
            (wrap - start) + end
        };
        ticks as f64 * f64::from(self.period) / 1e6
    }
}

impl Drop for GpuProfiler {
    fn drop(&mut self) {
        unsafe { self.device.destroy_query_pool(self.query_pool, None) };
    }
}

/// Writes one timestamp. `sync2` selects `vkCmdWriteTimestamp2`, which takes the
/// stage in the call; the legacy path takes it as a separate argument.
pub fn write_timestamp(
    recording: &mut crate::vulkan_command_buffer::Recording,
    queries: PassQueries,
    index: u32,
) {
    if index >= queries.capacity {
        return;
    }
    if queries.sync2 {
        recording.write_timestamp2(queries.pool, index);
    } else {
        recording.write_timestamp(queries.pool, index);
    }
}
