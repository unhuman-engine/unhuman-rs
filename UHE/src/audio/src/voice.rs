//! The active voices: one per entity, in a table that cannot grow without bound.

use adam::VoiceId;
use glam::Vec3;
use meats::{AudioSource, Entity};

use crate::sound::SoundId;
use crate::spatial::PanHandle;
use crate::spec::VoiceSpec;

/// One playing sound.
#[derive(Debug)]
pub struct Voice {
    pub id: VoiceId,
    pub sound: SoundId,
    /// The entity this voice follows, if it came from the scene rather than from a
    /// direct `play`.
    pub entity: Option<Entity>,
    pub spec: VoiceSpec,
    /// Written by the engine once a frame, read by the audio thread.
    pub pan: PanHandle,
    /// Where the voice was last frame, for [`Voice::advance`].
    prev_position: Vec3,
    /// World velocity in metres per second, from the last advance.
    velocity: Vec3,
    /// Set while [`VoicePool::reconcile`] walks the scene, sweeping the voices whose
    /// entity it did not visit.
    seen: bool,
}

impl Voice {
    /// World velocity in metres per second, written by [`Voice::advance`].
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Fold this frame's position into a velocity, for the engine's Doppler. A
    /// non-positive `dt` leaves the velocity at zero.
    pub fn advance(&mut self, dt: f32) {
        self.velocity = crate::doppler::velocity(self.prev_position, self.spec.position, dt);
        self.prev_position = self.spec.position;
    }
}

/// The active voices.
///
/// Bounded on purpose. When a spawn meets a full pool it displaces the voice with
/// the lowest [`VoiceSpec::priority`], oldest first among equals; a spawn whose
/// priority is below that lowest one is refused instead. That keeps a burst of
/// equal-priority one-shots cycling through the oldest voice while music at a
/// higher priority survives them. A capacity of zero still refuses everything.
#[derive(Debug)]
pub struct VoicePool {
    capacity: usize,
    next_id: VoiceId,
    voices: Vec<Voice>,
}

impl VoicePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_id: 1,
            voices: Vec::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.voices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }

    pub fn voices(&self) -> &[Voice] {
        &self.voices
    }

    pub fn voices_mut(&mut self) -> &mut [Voice] {
        &mut self.voices
    }

    pub fn get(&self, id: VoiceId) -> Option<&Voice> {
        self.voices.iter().find(|voice| voice.id == id)
    }

    pub fn get_mut(&mut self, id: VoiceId) -> Option<&mut Voice> {
        self.voices.iter_mut().find(|voice| voice.id == id)
    }

    pub fn voice_of(&self, entity: Entity) -> Option<&Voice> {
        self.voices
            .iter()
            .find(|voice| voice.entity == Some(entity))
    }

    /// Add a voice, stealing when the pool is full. `None` when the pool has no
    /// room at all, or when the new voice would not outrank the weakest one.
    ///
    /// An entity already sounding is replaced rather than shadowed, so a re-used
    /// entity never inherits a stale voice.
    pub fn spawn(
        &mut self,
        entity: Option<Entity>,
        sound: SoundId,
        spec: VoiceSpec,
    ) -> Option<VoiceId> {
        if self.capacity == 0 {
            return None;
        }
        if let Some(entity) = entity {
            if let Some(index) = self
                .voices
                .iter()
                .position(|voice| voice.entity == Some(entity))
            {
                self.voices.remove(index);
            }
        }
        if self.voices.len() >= self.capacity {
            let victim = self.weakest();
            if spec.priority < self.voices[victim].spec.priority {
                return None;
            }
            self.voices.remove(victim);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.voices.push(Voice {
            id,
            sound,
            entity,
            spec,
            pan: PanHandle::default(),
            prev_position: spec.position,
            velocity: Vec3::ZERO,
            seen: true,
        });
        Some(id)
    }

    /// Drop one voice, if it is still there.
    pub fn remove(&mut self, id: VoiceId) -> bool {
        match self.voices.iter().position(|voice| voice.id == id) {
            Some(index) => {
                self.voices.remove(index);
                true
            }
            None => false,
        }
    }

    /// The index of the voice to evict: lowest priority, oldest first among
    /// equals. Voice ids only increase and the table keeps insertion order, so the
    /// earliest index with the minimum priority is also the oldest.
    fn weakest(&self) -> usize {
        let mut index = 0;
        for (candidate, voice) in self.voices.iter().enumerate() {
            if voice.spec.priority < self.voices[index].spec.priority {
                index = candidate;
            }
        }
        index
    }

    /// One frame of bookkeeping against the scene's `AudioSource` components: a
    /// playing source that has no voice gets one, a live voice picks up the
    /// component's gain, pitch, looping, spatial flag and range, and a source that
    /// stopped loses its voice. Voices whose entity the scene no longer has are
    /// swept.
    ///
    /// `resolve` decides whether a source can be played at all. The engine's
    /// resolver looks the clip up in `vault` and answers `None` while its bytes are
    /// still loading, which leaves the source for a later frame.
    pub fn reconcile<'a, I, F>(&mut self, sources: I, mut resolve: F)
    where
        I: IntoIterator<Item = (Entity, &'a AudioSource, Vec3)>,
        F: FnMut(&AudioSource) -> Option<SoundId>,
    {
        for voice in self.voices.iter_mut() {
            voice.seen = false;
        }
        for (entity, source, position) in sources {
            let spec = VoiceSpec::from_source(source, position);
            match self
                .voices
                .iter()
                .position(|voice| voice.entity == Some(entity))
            {
                Some(index) => {
                    if source.playing {
                        self.voices[index].spec = spec;
                        self.voices[index].seen = true;
                    } else {
                        self.voices.remove(index);
                    }
                }
                None => {
                    if source.playing {
                        if let Some(sound) = resolve(source) {
                            self.spawn(Some(entity), sound, spec);
                        }
                    }
                }
            }
        }
        // Voices started directly, with no entity behind them, are not the scene's
        // to sweep.
        self.voices
            .retain(|voice| voice.seen || voice.entity.is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::DEFAULT_RANGE;
    use meats::Scene;

    fn source(playing: bool) -> AudioSource {
        AudioSource {
            clip: None,
            clip_path: String::new(),
            playing,
            looping: false,
            gain: 1.0,
            pitch: 1.0,
            spatial: true,
            range: DEFAULT_RANGE,
        }
    }

    /// Entities are `meats`', not wire's: the only way to make one is a scene.
    fn entity(name: &str) -> Entity {
        Scene::new().spawn(name)
    }

    fn spec(priority: i32) -> VoiceSpec {
        VoiceSpec {
            priority,
            ..VoiceSpec::default()
        }
    }

    #[test]
    fn spawn_replaces_the_voice_of_a_re_used_entity() {
        let mut pool = VoicePool::new(4);
        let entity = entity("source");
        let first = pool
            .spawn(Some(entity), SoundId::new(1), VoiceSpec::default())
            .expect("room");
        let second = pool
            .spawn(Some(entity), SoundId::new(2), VoiceSpec::default())
            .expect("room");
        assert_ne!(first, second);
        assert_eq!(pool.len(), 1);
        assert_eq!(
            pool.voice_of(entity).expect("one voice").sound,
            SoundId::new(2)
        );
    }

    #[test]
    fn a_zero_capacity_pool_refuses_every_voice() {
        let mut pool = VoicePool::new(0);
        assert!(pool
            .spawn(None, SoundId::new(1), VoiceSpec::default())
            .is_none());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn a_full_pool_steals_the_lowest_priority_and_refuses_below_it() {
        let mut pool = VoicePool::new(2);
        let quiet = pool
            .spawn(None, SoundId::new(1), spec(1))
            .expect("room for one");
        let loud = pool
            .spawn(None, SoundId::new(2), spec(5))
            .expect("room for two");

        // Lower than the weakest (1): refused, nothing is stolen.
        assert!(pool.spawn(None, SoundId::new(3), spec(0)).is_none());
        assert_eq!(pool.len(), 2);
        assert!(pool.get(quiet).is_some());
        assert!(pool.get(loud).is_some());

        // Equal to the weakest: the weakest is displaced.
        let equal = pool
            .spawn(None, SoundId::new(4), spec(1))
            .expect("displaces the weakest");
        assert!(pool.get(quiet).is_none());
        assert!(pool.get(loud).is_some());
        assert!(pool.get(equal).is_some());
    }

    #[test]
    fn equal_priority_evicts_the_oldest() {
        let mut pool = VoicePool::new(2);
        let first = pool.spawn(None, SoundId::new(1), spec(0)).expect("room");
        let second = pool.spawn(None, SoundId::new(2), spec(0)).expect("room");
        let third = pool.spawn(None, SoundId::new(3), spec(0)).expect("room");
        assert!(pool.get(first).is_none());
        assert!(pool.get(second).is_some());
        assert!(pool.get(third).is_some());
    }

    #[test]
    fn advance_turns_a_moved_voice_into_a_velocity() {
        let mut pool = VoicePool::new(1);
        let id = pool
            .spawn(None, SoundId::new(1), VoiceSpec::default())
            .expect("room");
        pool.get_mut(id).expect("voice").spec.position = Vec3::new(2.0, 0.0, 0.0);
        pool.get_mut(id).expect("voice").advance(0.5);
        assert_eq!(pool.get(id).expect("voice").velocity(), Vec3::X * 4.0);
        // A second tick with no movement is still.
        pool.get_mut(id).expect("voice").advance(0.5);
        assert_eq!(pool.get(id).expect("voice").velocity(), Vec3::ZERO);
    }

    #[test]
    fn reconcile_updates_a_running_voice_in_place() {
        let mut pool = VoicePool::new(4);
        let entity = entity("source");
        let quiet = source(true);
        pool.reconcile([(entity, &quiet, Vec3::ZERO)], |_| Some(SoundId::new(1)));
        let id = pool.voice_of(entity).expect("spawned").id;

        let mut loud = source(true);
        loud.gain = 0.25;
        loud.looping = true;
        let position = Vec3::new(1.0, 0.0, 2.0);
        pool.reconcile([(entity, &loud, position)], |_| Some(SoundId::new(1)));

        assert_eq!(pool.len(), 1);
        let voice = pool.get(id).expect("same voice");
        assert_eq!(voice.spec.gain, 0.25);
        assert!(voice.spec.looping);
        assert_eq!(voice.spec.position, position);
    }

    #[test]
    fn reconcile_sweeps_a_voice_whose_entity_is_gone() {
        let mut pool = VoicePool::new(4);
        let entity = entity("source");
        let playing = source(true);
        pool.reconcile([(entity, &playing, Vec3::ZERO)], |_| Some(SoundId::new(1)));
        assert_eq!(pool.len(), 1);

        pool.reconcile(Vec::new(), |_| Some(SoundId::new(1)));
        assert_eq!(pool.len(), 0);
    }
}
