//! The audio engine.
//!
//! One owner holds the voices and the clip table, and the device is an `Option`, so
//! the bookkeeping runs with the `Option` empty and is testable without a sound card.

use glam::Vec3;

use adam::{AudioBackend, Listener, VoiceId};
use meats::hecs;
use meats::{AudioSource, Scene};
use vault::{AssetId, Assets};

use crate::attenuation::{attenuation_modeled, AttenuationModel};
use crate::backend::Backend;
use crate::doppler;
use crate::mixer::Mixer;
use crate::sound::{Bus, SoundId, SoundTable};
use crate::spatial::pan;
use crate::spec::VoiceSpec;
use crate::voice::VoicePool;

/// Voices an engine runs at once when nothing asks for another number.
pub const DEFAULT_VOICES: usize = 64;

/// Playback: the voice table, the clips, the output settings, and the device if
/// there is one.
///
/// With no output device this is a silent engine: it answers every call, counts no
/// voices and plays nothing. That is what makes registering it safe on a build
/// machine, and it is the only reason `WirePlugin::build` cannot fail.
pub struct AudioEngine {
    mixer: Mixer,
    pool: VoicePool,
    sounds: SoundTable,
    device: Option<Backend>,
}

impl AudioEngine {
    /// Open the default output device; silent if there is none.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_VOICES)
    }

    /// As [`AudioEngine::new`], with a cap on simultaneous voices. A capacity of
    /// zero refuses every voice, which is a valid way to switch audio off.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            mixer: Mixer::new(),
            pool: VoicePool::new(capacity),
            sounds: SoundTable::new(),
            device: Backend::open(),
        }
    }

    /// True when there was no output device to open.
    pub fn is_silent(&self) -> bool {
        self.device.is_none()
    }

    pub fn listener(&self) -> Listener {
        self.mixer.listener()
    }

    pub fn set_listener(&mut self, listener: Listener) {
        self.mixer.set_listener(listener);
    }

    /// Voices playing right now. Zero in silent mode, because nothing was started.
    pub fn active_voices(&self) -> usize {
        self.pool.len()
    }

    pub fn bus_gain(&self, bus: Bus) -> f32 {
        self.mixer.bus_gain(bus)
    }

    pub fn set_bus_gain(&mut self, bus: Bus, gain: f32) {
        self.mixer.set_bus_gain(bus, gain);
    }

    pub fn attenuation_model(&self) -> AttenuationModel {
        self.mixer.attenuation_model()
    }

    /// Choose the falloff curve for every spatial voice.
    pub fn set_attenuation_model(&mut self, model: AttenuationModel) {
        self.mixer.set_attenuation_model(model);
    }

    pub fn rolloff(&self) -> f32 {
        self.mixer.rolloff()
    }

    /// Shape the inverse and exponential curves; 1.0 is neutral.
    pub fn set_rolloff(&mut self, rolloff: f32) {
        self.mixer.set_rolloff(rolloff);
    }

    pub fn doppler_factor(&self) -> f32 {
        self.mixer.doppler_factor()
    }

    /// Strength of the Doppler shift; 0.0 switches it off.
    pub fn set_doppler_factor(&mut self, doppler: f32) {
        self.mixer.set_doppler_factor(doppler);
    }

    /// Speed the listener is moving at, from the last `update`'s listener pose.
    pub fn listener_velocity(&self) -> Vec3 {
        self.mixer.listener_velocity()
    }

    /// The clip id for an asset, registering it the first time `vault` has its
    /// bytes. `None` while the load is in flight.
    pub fn register(&mut self, assets: &Assets, asset: AssetId) -> Option<SoundId> {
        self.sounds.resolve(assets, asset)
    }

    /// Play a clip with no position, so no falloff or panning.
    pub fn play_2d(
        &mut self,
        assets: &Assets,
        clip: AssetId,
        gain: f32,
        looping: bool,
    ) -> Option<VoiceId> {
        let sound = self.register(assets, clip)?;
        self.play_sound(
            sound,
            VoiceSpec {
                gain,
                looping,
                ..VoiceSpec::default()
            },
        )
    }

    /// Play a clip at a world position, spatialised against the listener.
    pub fn play_3d(
        &mut self,
        assets: &Assets,
        clip: AssetId,
        position: Vec3,
        gain: f32,
        looping: bool,
        range: f32,
    ) -> Option<VoiceId> {
        let sound = self.register(assets, clip)?;
        self.play_sound(
            sound,
            VoiceSpec {
                gain,
                looping,
                spatial: true,
                range,
                position,
                ..VoiceSpec::default()
            },
        )
    }

    /// Start an already-registered clip. `None` when the device refused it or the
    /// pool had no room.
    pub fn play_sound(&mut self, sound: SoundId, spec: VoiceSpec) -> Option<VoiceId> {
        self.device.as_ref()?;
        let id = self.pool.spawn(None, sound, spec)?;
        if self.start_player(id) {
            Some(id)
        } else {
            self.pool.remove(id);
            None
        }
    }

    /// Stop a voice now, whether the scene started it or a direct play did.
    pub fn stop(&mut self, voice: VoiceId) -> bool {
        if let Some(backend) = self.device.as_mut() {
            backend.stop(voice);
        }
        self.pool.remove(voice)
    }

    /// One frame: the voices are brought in line with the scene's playing sources,
    /// their velocities are advanced by `dt`, then the device is driven to match.
    /// A no-op with no device.
    pub fn update(&mut self, scene: &Scene, assets: &Assets, dt: f32) {
        if self.device.is_none() {
            return;
        }
        self.reconcile(scene, assets);
        self.advance(dt);
        self.drive();
    }

    /// Listener and voice velocities for this frame. The listener's previous
    /// position starts level with it, so the first frame has no phantom speed.
    fn advance(&mut self, dt: f32) {
        self.mixer.advance_listener(dt);
        for voice in self.pool.voices_mut() {
            voice.advance(dt);
        }
    }

    /// Start the player of a voice that has none, failing when its clip cannot be
    /// decoded.
    fn start_player(&mut self, id: VoiceId) -> bool {
        let Some(voice) = self.pool.get(id) else {
            return false;
        };
        let Some(sound) = self.sounds.get(voice.sound) else {
            return false;
        };
        let Some(backend) = self.device.as_mut() else {
            return false;
        };
        backend.start(voice, sound)
    }

    /// Bring the voice pool in line with the scene: a source that started playing
    /// gets a voice at its entity's world position, a source that stopped loses it,
    /// and a running voice picks up whatever the component now says.
    fn reconcile(&mut self, scene: &Scene, assets: &Assets) {
        let world = scene.world();
        let mut query = world.query::<(hecs::Entity, &AudioSource)>();
        let sources = query.iter().filter_map(|(hecs_entity, source)| {
            let entity = scene.entity_of(hecs_entity)?;
            let position = scene.world_transform(entity).w_axis.truncate();
            Some((entity, source, position))
        });

        let pool = &mut self.pool;
        let sounds = &mut self.sounds;
        pool.reconcile(sources, |source| match source.clip {
            Some(clip) => sounds.resolve(assets, clip),
            None => None,
        });
    }

    /// Bring the device in line with the pool: start what has no player yet, apply
    /// gain, pitch and pan, and reap what has finished.
    fn drive(&mut self) {
        let mixer = self.mixer;
        let listener = mixer.listener();

        // Voices the scene added this frame have no player yet.
        let pending: Vec<VoiceId> = match &self.device {
            Some(backend) => self
                .pool
                .voices()
                .iter()
                .filter(|voice| !backend.has(voice.id))
                .map(|voice| voice.id)
                .collect(),
            None => return,
        };
        for id in pending {
            // A clip that cannot be decoded is dropped rather than retried, and
            // logged, every frame.
            if !self.start_player(id) {
                self.pool.remove(id);
            }
        }

        let Some(backend) = self.device.as_mut() else {
            return;
        };

        for id in backend.finished() {
            backend.stop(id);
            self.pool.remove(id);
        }

        for voice in self.pool.voices() {
            let distance = (voice.spec.position - listener.position).length();
            if voice.spec.spatial {
                voice.pan.set_pan(pan(&listener, voice.spec.position));
            } else {
                voice.pan.set_gains(1.0, 1.0);
            }
            let falloff = if voice.spec.spatial {
                attenuation_modeled(
                    mixer.attenuation_model(),
                    distance,
                    voice.spec.range,
                    mixer.rolloff(),
                )
            } else {
                1.0
            };
            let speed = if voice.spec.spatial {
                voice.spec.pitch
                    * doppler::doppler_factor(
                        doppler::SPEED_OF_SOUND,
                        mixer.doppler_factor(),
                        mixer.listener_velocity(),
                        voice.velocity(),
                        listener.position,
                        voice.spec.position,
                    )
            } else {
                voice.spec.pitch
            };
            let volume =
                voice.spec.gain * falloff * mixer.bus_gain(voice.spec.bus) * mixer.master_gain();
            backend.set_gain(voice.id, volume, speed);
        }
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBackend for AudioEngine {
    fn name(&self) -> &'static str {
        "wire"
    }

    fn listener(&self) -> Listener {
        self.mixer.listener()
    }

    fn set_listener(&mut self, listener: Listener) {
        self.mixer.set_listener(listener);
    }

    /// `dt` drives the Doppler shift: the velocity is the listener's and each
    /// voice's change in position over this frame. rodio's clock still runs the
    /// mixer; `dt` is only for the shift.
    fn update(&mut self, scene: &Scene, assets: &Assets, dt: f32) {
        AudioEngine::update(self, scene, assets, dt);
    }

    fn play(&mut self, assets: &Assets, clip: AssetId, gain: f32) -> Option<VoiceId> {
        self.play_2d(assets, clip, gain, false)
    }

    fn stop(&mut self, voice: VoiceId) {
        AudioEngine::stop(self, voice);
    }

    fn active_voices(&self) -> usize {
        self.pool.len()
    }

    fn master_gain(&self) -> f32 {
        self.mixer.master_gain()
    }

    fn set_master_gain(&mut self, gain: f32) {
        self.mixer.set_master_gain(gain);
    }
}
