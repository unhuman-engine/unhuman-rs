use std::time::Duration;

/// Frame clock with a fixed-step accumulator.
///
/// The fixed step is what makes headless robotics runs reproducible: the
/// simulation advances in whole, identical increments regardless of how fast the
/// host actually is.
pub struct Time {
    last: std::time::Instant,
    delta: Duration,
    accumulator: Duration,
    elapsed: Duration,
    fixed_dt: Duration,
    scale: f32,
    frame: u64,
}

impl Time {
    /// 60 Hz by default.
    pub fn new() -> Self {
        Self::with_fixed_dt(Duration::from_secs_f64(1.0 / 60.0))
    }

    pub fn with_fixed_dt(fixed_dt: Duration) -> Self {
        Self {
            last: std::time::Instant::now(),
            delta: Duration::ZERO,
            accumulator: Duration::ZERO,
            elapsed: Duration::ZERO,
            fixed_dt: fixed_dt.max(Duration::from_micros(1)),
            scale: 1.0,
            frame: 0,
        }
    }

    /// Samples the clock and advances the accumulator. Returns the real elapsed
    /// time, unscaled and unclamped, for interpolation and diagnostics.
    pub fn tick(&mut self) -> Duration {
        let now = std::time::Instant::now();
        // Clamped so a stalled frame cannot feed the spiral of death: a
        // 10-second hitch would otherwise queue 600 simulation steps.
        let raw = now
            .saturating_duration_since(self.last)
            .min(Duration::from_millis(250));
        self.last = now;

        self.delta = raw;
        self.elapsed += raw;
        self.accumulator += raw.mul_f32(self.scale);
        self.frame += 1;
        raw
    }

    /// Yields the next whole simulation step, or `None` if the accumulator has
    /// not reached one yet.
    pub fn consume_fixed_step(&mut self) -> Option<Duration> {
        if self.accumulator >= self.fixed_dt {
            self.accumulator -= self.fixed_dt;
            Some(self.fixed_dt)
        } else {
            None
        }
    }

    /// Steps remaining in the accumulator. A frame that is badly behind will
    /// report more than one; call `consume_fixed_step` in a loop.
    pub fn pending_steps(&self) -> u32 {
        (self.accumulator.as_nanos() / self.fixed_dt.as_nanos()) as u32
    }

    pub fn delta(&self) -> Duration {
        self.delta
    }

    pub fn fixed_dt(&self) -> Duration {
        self.fixed_dt
    }

    /// How far the accumulator is into the next fixed step, in `[0, 1)`.
    ///
    /// A renderer samples the last two steps with this: the state at the last
    /// step, not the state now, so the blend never extrapolates.
    pub fn alpha(&self) -> f32 {
        (self.accumulator.as_secs_f64() / self.fixed_dt.as_secs_f64()).clamp(0.0, 1.0) as f32
    }

    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Slows or speeds up simulated time. `0.0` freezes the simulation while
    /// the render loop keeps running.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale.clamp(0.0, 1_000.0);
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}
