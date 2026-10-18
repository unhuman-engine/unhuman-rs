//! What a voice plays and how: the per-voice parameters.

use glam::Vec3;
use meats::AudioSource;

use crate::sound::Bus;

/// The range a spatial voice falls off over when nothing names one.
pub const DEFAULT_RANGE: f32 = 50.0;

/// The priority a source gets when nothing names one.
pub const DEFAULT_PRIORITY: i32 = 0;

/// How one voice sounds. In the live path this is copied off the scene's
/// `AudioSource` once a frame, which is how a gain changed in the editor reaches a
/// sound that is already playing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceSpec {
    pub gain: f32,
    pub pitch: f32,
    pub looping: bool,
    pub spatial: bool,
    pub range: f32,
    pub position: Vec3,
    pub bus: Bus,
    /// Higher survives a full voice table; see [`crate::VoicePool::spawn`].
    pub priority: i32,
}

impl VoiceSpec {
    /// What a scene source asks for, at the entity's world position: the component
    /// carries no position or priority of its own.
    pub fn from_source(source: &AudioSource, position: Vec3) -> Self {
        Self {
            gain: source.gain,
            pitch: source.pitch,
            looping: source.looping,
            spatial: source.spatial,
            range: source.range,
            position,
            bus: Bus::Sfx,
            priority: DEFAULT_PRIORITY,
        }
    }
}

impl Default for VoiceSpec {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: false,
            range: DEFAULT_RANGE,
            position: Vec3::ZERO,
            bus: Bus::Sfx,
            priority: DEFAULT_PRIORITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_source_carries_the_component_and_defaults_the_rest() {
        let source = AudioSource {
            gain: 0.25,
            pitch: 1.5,
            looping: true,
            spatial: true,
            range: 12.0,
            ..AudioSource::default()
        };
        let position = Vec3::new(1.0, 2.0, 3.0);
        let spec = VoiceSpec::from_source(&source, position);
        assert_eq!(spec.gain, 0.25);
        assert_eq!(spec.pitch, 1.5);
        assert!(spec.looping);
        assert!(spec.spatial);
        assert_eq!(spec.range, 12.0);
        assert_eq!(spec.position, position);
        assert_eq!(spec.bus, Bus::Sfx);
        assert_eq!(spec.priority, DEFAULT_PRIORITY);
    }
}
