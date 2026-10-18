//! Output settings applied to every voice: master and bus gain, the listener,
//! and the distance and Doppler parameters.

use adam::Listener;
use glam::Vec3;

use crate::attenuation::{AttenuationModel, DEFAULT_ROLLOFF};
use crate::doppler::{self, DEFAULT_DOPPLER_FACTOR};
use crate::sound::Bus;

/// The per-engine output controls. One of these is all the mutable state that is
/// not a voice or a clip, so [`crate::AudioEngine`] stays a table plus a device.
#[derive(Clone, Copy, Debug)]
pub struct Mixer {
    listener: Listener,
    master_gain: f32,
    bus_gains: [f32; Bus::COUNT],
    /// Where the listener was the previous update, for Doppler.
    prev_listener: Vec3,
    listener_velocity: Vec3,
    attenuation: AttenuationModel,
    rolloff: f32,
    doppler: f32,
}

impl Mixer {
    /// Every gain at unity, the default ear, linear falloff, full Doppler.
    pub fn new() -> Self {
        let listener = Listener::default();
        Self {
            listener,
            master_gain: 1.0,
            bus_gains: [1.0; Bus::COUNT],
            prev_listener: listener.position,
            listener_velocity: Vec3::ZERO,
            attenuation: AttenuationModel::default(),
            rolloff: DEFAULT_ROLLOFF,
            doppler: DEFAULT_DOPPLER_FACTOR,
        }
    }

    pub fn listener(&self) -> Listener {
        self.listener
    }

    pub fn set_listener(&mut self, listener: Listener) {
        self.listener = listener;
    }

    pub fn master_gain(&self) -> f32 {
        self.master_gain
    }

    pub fn set_master_gain(&mut self, gain: f32) {
        self.master_gain = gain;
    }

    pub fn bus_gain(&self, bus: Bus) -> f32 {
        self.bus_gains[bus.slot()]
    }

    pub fn set_bus_gain(&mut self, bus: Bus, gain: f32) {
        self.bus_gains[bus.slot()] = gain;
    }

    pub fn attenuation_model(&self) -> AttenuationModel {
        self.attenuation
    }

    /// Choose the falloff curve for every spatial voice.
    pub fn set_attenuation_model(&mut self, model: AttenuationModel) {
        self.attenuation = model;
    }

    pub fn rolloff(&self) -> f32 {
        self.rolloff
    }

    /// Shape the inverse and exponential curves; 1.0 is neutral.
    pub fn set_rolloff(&mut self, rolloff: f32) {
        self.rolloff = rolloff;
    }

    pub fn doppler_factor(&self) -> f32 {
        self.doppler
    }

    /// Strength of the Doppler shift; 0.0 switches it off.
    pub fn set_doppler_factor(&mut self, doppler: f32) {
        self.doppler = doppler;
    }

    pub fn listener_velocity(&self) -> Vec3 {
        self.listener_velocity
    }

    /// Fold this frame's listener motion into a velocity and return it. The first
    /// frame matches the previous position, so it reads as still, not moving.
    pub fn advance_listener(&mut self, dt: f32) -> Vec3 {
        self.listener_velocity = doppler::velocity(self.prev_listener, self.listener.position, dt);
        self.prev_listener = self.listener.position;
        self.listener_velocity
    }
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_listener_is_still() {
        let mut mixer = Mixer::new();
        assert_eq!(mixer.advance_listener(0.016), Vec3::ZERO);
    }

    #[test]
    fn moving_the_listener_gives_it_a_velocity() {
        let mut mixer = Mixer::new();
        mixer.set_listener(Listener::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::NEG_Z,
            Vec3::Y,
        ));
        // The default listener starts at the origin, so one metre in 0.5s is 2 m/s.
        assert_eq!(mixer.advance_listener(0.5), Vec3::X * 2.0);
        // Stay put and it reads as still again.
        assert_eq!(mixer.advance_listener(0.5), Vec3::ZERO);
    }

    #[test]
    fn gains_default_to_unity_and_are_settable() {
        let mut mixer = Mixer::new();
        assert_eq!(mixer.master_gain(), 1.0);
        assert_eq!(mixer.bus_gain(Bus::Sfx), 1.0);
        mixer.set_master_gain(0.5);
        mixer.set_bus_gain(Bus::Music, 0.25);
        assert_eq!(mixer.master_gain(), 0.5);
        assert_eq!(mixer.bus_gain(Bus::Music), 0.25);
        assert_eq!(mixer.bus_gain(Bus::Sfx), 1.0, "buses are independent");
    }
}
