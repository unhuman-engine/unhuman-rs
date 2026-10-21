//! The OS rows: process CPU, resident memory, and a best-effort GPU load.
//!
//! Reading `/proc` is cheap but not free, and the GPU's counters can block on a
//! sysfs read; none of that belongs in a frame. [`SystemSampler`] reads only
//! when its interval has elapsed and carries the previous sample otherwise.

use std::time::{Duration, Instant};

/// The rows read from the OS. `None` means the platform does not expose it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SystemLoad {
    /// Process CPU as a percent of one core. Above 100 is a working thread pool;
    /// `None` until a second sample gives a delta.
    pub cpu_percent: Option<f32>,
    /// Resident set size in bytes.
    pub resident_bytes: Option<u64>,
    /// Best-effort GPU busy percent from DRM sysfs.
    pub gpu_percent: Option<f32>,
}

pub struct SystemSampler {
    interval: Duration,
    last: Option<Instant>,
    cpu_ticks: Option<u64>,
    load: SystemLoad,
    samples: u64,
}

impl SystemSampler {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last: None,
            cpu_ticks: None,
            load: SystemLoad::default(),
            samples: 0,
        }
    }

    /// Samples if `interval` has passed since the last sample, otherwise returns
    /// the previous values unchanged.
    pub fn poll(&mut self, now: Instant) -> SystemLoad {
        if let Some(last) = self.last {
            if now.saturating_duration_since(last) < self.interval {
                return self.load;
            }
        }

        let elapsed = self.last.map(|last| now.saturating_duration_since(last));
        self.last = Some(now);
        self.samples += 1;
        self.load = self.read(elapsed);
        self.load
    }

    /// How many OS samples have been taken, which is what the timer's test
    /// counts.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    pub fn load(&self) -> SystemLoad {
        self.load
    }

    fn read(&mut self, elapsed: Option<Duration>) -> SystemLoad {
        let ticks = platform::cpu_ticks();
        let cpu_percent = match (ticks, self.cpu_ticks, elapsed) {
            (Some(now), Some(previous), Some(elapsed)) if !elapsed.is_zero() => {
                let delta = now.saturating_sub(previous) as f32;
                let seconds = elapsed.as_secs_f32();
                let hz = platform::ticks_per_second() as f32;
                Some(delta / (hz * seconds) * 100.0)
            }
            _ => None,
        };
        if ticks.is_some() {
            self.cpu_ticks = ticks;
        }

        SystemLoad {
            cpu_percent,
            resident_bytes: platform::resident_bytes(),
            gpu_percent: platform::gpu_percent(),
        }
    }
}

/// Fields after `comm` in `/proc/self/stat`: state is field 3, `utime` is field
/// 14 and `stime` is field 15. The last `)` is the comm's, because a process
/// name may contain spaces or parentheses.
fn parse_cpu_ticks(stat: &str) -> Option<u64> {
    let tail = &stat[stat.rfind(')')? + 1..];
    let mut fields = tail.split_whitespace();
    let utime: u64 = fields.nth(11)?.parse().ok()?;
    let stime: u64 = fields.next()?.parse().ok()?;
    Some(utime + stime)
}

/// `/proc/self/statm`'s second field is the resident page count.
fn parse_resident_pages(statm: &str) -> Option<u64> {
    statm.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(target_os = "linux")]
mod platform {
    use std::fs;

    pub fn cpu_ticks() -> Option<u64> {
        let stat = fs::read_to_string("/proc/self/stat").ok()?;
        super::parse_cpu_ticks(&stat)
    }

    pub fn resident_bytes() -> Option<u64> {
        let statm = fs::read_to_string("/proc/self/statm").ok()?;
        let pages = super::parse_resident_pages(&statm)?;
        Some(pages * page_size())
    }

    fn page_size() -> u64 {
        // SAFETY: `sysconf` only reads process-wide constants.
        let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if value > 0 {
            value as u64
        } else {
            4096
        }
    }

    pub fn ticks_per_second() -> u64 {
        // SAFETY: `sysconf` only reads process-wide constants.
        let value = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if value > 0 {
            value as u64
        } else {
            100
        }
    }

    /// The DRM sysfs attribute some drivers expose; absent elsewhere, which is
    /// why the panel treats it as a best-effort row.
    pub fn gpu_percent() -> Option<f32> {
        let entries = fs::read_dir("/sys/class/drm").ok()?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let path = entry.path().join("device/gpu_busy_percent");
            if let Ok(text) = fs::read_to_string(path) {
                if let Ok(percent) = text.trim().parse::<f32>() {
                    return Some(percent);
                }
            }
        }
        None
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    pub fn cpu_ticks() -> Option<u64> {
        None
    }

    pub fn ticks_per_second() -> u64 {
        100
    }

    pub fn resident_bytes() -> Option<u64> {
        None
    }

    pub fn gpu_percent() -> Option<f32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ticks_parse_past_a_comm_with_spaces_and_parentheses() {
        let stat = "42 (worker (pool) 3) S 1 42 42 0 -1 4194560 100 0 0 0 7 5 0 0";
        assert_eq!(parse_cpu_ticks(stat), Some(12));
    }

    #[test]
    fn a_malformed_stat_is_not_a_measurement() {
        assert_eq!(parse_cpu_ticks("42 (short) S 1"), None);
        assert_eq!(parse_cpu_ticks("no comm"), None);
    }

    #[test]
    fn statm_resident_is_the_second_field() {
        assert_eq!(parse_resident_pages("100 25 10 4 0 50 0"), Some(25));
        assert_eq!(parse_resident_pages("100"), None);
    }

    #[test]
    fn the_sampler_reads_once_per_interval_not_once_per_frame() {
        let interval = Duration::from_millis(250);
        let mut sampler = SystemSampler::new(interval);
        let start = Instant::now();

        sampler.poll(start);
        assert_eq!(sampler.samples(), 1);

        sampler.poll(start + Duration::from_millis(16));
        assert_eq!(sampler.samples(), 1);

        sampler.poll(start + Duration::from_millis(249));
        assert_eq!(sampler.samples(), 1);

        sampler.poll(start + interval);
        assert_eq!(sampler.samples(), 2);
    }

    #[test]
    fn an_os_row_that_is_missing_stays_none() {
        // The Linux reader returns `Some` for memory; other platforms return
        // `None`. Either is a valid answer, and neither invents a zero.
        let mut sampler = SystemSampler::new(Duration::ZERO);
        let load = sampler.poll(Instant::now());
        if cfg!(target_os = "linux") {
            assert!(load.resident_bytes.is_some());
        }
    }
}
