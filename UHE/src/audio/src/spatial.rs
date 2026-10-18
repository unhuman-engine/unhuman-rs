//! Stereo placement.
//!
//! The panning maths is miniaudio's, floor included; distance falloff moved to
//! [`crate::attenuation`], which now offers more than the linear curve.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use glam::Vec3;
use rodio::source::SeekError;
use rodio::{ChannelCount, Sample, SampleRate, Source};

use adam::Listener;

/// Stereo: what [`Pan`] emits, and the layout miniaudio spatialises into.
pub(crate) const CHANNELS: ChannelCount = rodio::math::nz!(2);

/// miniaudio's `minSpatializationChannelGain`. Its comment says extreme panning
/// "doesn't even remotely represent the real world", so the speaker a sound is
/// turned away from never goes fully silent.
pub const MIN_CHANNEL_GAIN: f32 = 0.2;

/// Where a source sits relative to the listener: -1.0 hard left, 0.0 directly in
/// front of it, +1.0 hard right. The listener's right vector is `forward x up`,
/// which is miniaudio's right-handed convention (forward is -Z).
pub fn pan(listener: &Listener, position: Vec3) -> f32 {
    let offset = position - listener.position;
    let right = listener.forward.cross(listener.up);
    let scale = offset.length() * right.length();
    if scale <= f32::EPSILON {
        return 0.0;
    }
    (offset.dot(right) / scale).clamp(-1.0, 1.0)
}

/// The per-channel gains miniaudio's spatializer gives a source at `pan`: a sound
/// in front reaches both speakers at half gain, and each speaker keeps at least
/// [`MIN_CHANNEL_GAIN`]. A non-spatial voice bypasses this and stays at 1.0/1.0.
pub fn channel_gains(pan: f32) -> (f32, f32) {
    let pan = pan.clamp(-1.0, 1.0);
    (
        ((1.0 - pan) * 0.5).max(MIN_CHANNEL_GAIN),
        ((1.0 + pan) * 0.5).max(MIN_CHANNEL_GAIN),
    )
}

/// The live stereo gains of one voice. The engine writes them once a frame and the
/// audio thread reads them per sample, so a moving listener pans a sound that is
/// already playing.
#[derive(Clone, Debug)]
pub struct PanHandle {
    left: Arc<AtomicU32>,
    right: Arc<AtomicU32>,
}

impl PanHandle {
    /// Centred at full gain, which is what a non-spatial voice wants.
    pub fn centered() -> Self {
        Self::new(1.0, 1.0)
    }

    pub fn new(left: f32, right: f32) -> Self {
        Self {
            left: Arc::new(AtomicU32::new(left.to_bits())),
            right: Arc::new(AtomicU32::new(right.to_bits())),
        }
    }

    pub fn gains(&self) -> (f32, f32) {
        (
            f32::from_bits(self.left.load(Ordering::Relaxed)),
            f32::from_bits(self.right.load(Ordering::Relaxed)),
        )
    }

    pub fn set_gains(&self, left: f32, right: f32) {
        self.left.store(left.to_bits(), Ordering::Relaxed);
        self.right.store(right.to_bits(), Ordering::Relaxed);
    }

    /// Both gains from one pan value, with miniaudio's restrained panning.
    pub fn set_pan(&self, pan: f32) {
        let (left, right) = channel_gains(pan);
        self.set_gains(left, right);
    }
}

impl Default for PanHandle {
    fn default() -> Self {
        Self::centered()
    }
}

/// A stereo source whose per-channel gains change while it plays: this is what
/// makes a spatial voice follow the listener. Its input must be stereo: wrap a
/// mono one in [`rodio::source::UniformSourceIterator`] first.
#[derive(Clone, Debug)]
pub struct Pan<S> {
    input: S,
    gains: PanHandle,
    pending: Option<Sample>,
}

impl<S> Pan<S> {
    pub fn new(input: S, gains: PanHandle) -> Self {
        Self {
            input,
            gains,
            pending: None,
        }
    }
}

impl<S: Source> Iterator for Pan<S> {
    type Item = Sample;

    fn next(&mut self) -> Option<Sample> {
        if let Some(sample) = self.pending.take() {
            return Some(sample);
        }
        let left = self.input.next()?;
        let right = self.input.next()?;
        let (left_gain, right_gain) = self.gains.gains();
        self.pending = Some(right * right_gain);
        Some(left * left_gain)
    }
}

impl<S: Source> Source for Pan<S> {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> ChannelCount {
        CHANNELS
    }

    fn sample_rate(&self) -> SampleRate {
        self.input.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration()
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.pending = None;
        self.input.try_seek(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::buffer::SamplesBuffer;

    fn stereo(samples: Vec<f32>) -> SamplesBuffer {
        SamplesBuffer::new(CHANNELS, rodio::math::nz!(48_000), samples)
    }

    #[test]
    fn pan_spreads_a_source_across_the_listeners_right_vector() {
        let listener = Listener::default();
        assert_eq!(pan(&listener, Vec3::new(-4.0, 0.0, 0.0)), -1.0);
        assert_eq!(pan(&listener, Vec3::new(4.0, 0.0, 0.0)), 1.0);
        assert_eq!(pan(&listener, Vec3::new(0.0, 0.0, -4.0)), 0.0);
        assert_eq!(pan(&listener, Vec3::ZERO), 0.0);
    }

    #[test]
    fn channel_gains_never_silence_a_speaker() {
        assert_eq!(channel_gains(-1.0), (1.0, MIN_CHANNEL_GAIN));
        assert_eq!(channel_gains(1.0), (MIN_CHANNEL_GAIN, 1.0));
        assert_eq!(channel_gains(0.0), (0.5, 0.5));
        assert_eq!(channel_gains(0.5), (0.25, 0.75));
    }

    #[test]
    fn pan_scales_each_channel_of_the_frame() {
        let mut source = Pan::new(stereo(vec![1.0, 2.0, 3.0, 4.0]), PanHandle::new(1.0, 0.0));
        assert_eq!(source.next(), Some(1.0));
        assert_eq!(source.next(), Some(0.0));
        assert_eq!(source.next(), Some(3.0));
        assert_eq!(source.next(), Some(0.0));
        assert_eq!(source.next(), None);
    }

    #[test]
    fn pan_follows_the_handle_while_the_sound_plays() {
        let handle = PanHandle::new(1.0, 1.0);
        let mut source = Pan::new(stereo(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), handle.clone());
        assert_eq!(source.next(), Some(1.0));
        assert_eq!(source.next(), Some(1.0));
        handle.set_gains(0.5, 0.25);
        assert_eq!(source.next(), Some(0.5));
        assert_eq!(source.next(), Some(0.25));
        assert_eq!(source.next(), Some(0.5));
        assert_eq!(source.next(), Some(0.25));
        assert_eq!(source.next(), None);
    }

    #[test]
    fn pan_reports_stereo_whatever_the_input_rate() {
        let source = Pan::new(stereo(vec![0.0; 8]), PanHandle::centered());
        assert_eq!(source.channels().get(), 2);
        assert_eq!(source.sample_rate().get(), 48_000);
        assert_eq!(source.current_span_len(), None);
    }
}
