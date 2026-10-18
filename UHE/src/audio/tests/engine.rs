//! `wire`'s acceptance tests: attenuation, panning, the voice pool, and the
//! reconciliation `AudioEngine::update` runs.
//!
//! Everything here is device-free. `AudioEngine::update` is `VoicePool::reconcile`
//! plus a device drive, and with no device it is a no-op by design, so the tests
//! drive the pool exactly as `update` does on a machine that can play. What needs a
//! sound card is at the bottom and `#[ignore]`d.

use glam::Vec3;
use meats::{AudioSource, Scene};
use wire::{
    attenuation, attenuation_modeled, channel_gains, pan, AttenuationModel, Listener, SoundId,
    VoicePool, VoiceSpec,
};

/// A scene component as `meats` stores it. `clip` is `None` -- the resolver in these
/// tests answers for any source, which is what the engine's resolver does once
/// `vault` has the bytes.
fn source(playing: bool) -> AudioSource {
    AudioSource {
        clip: None,
        clip_path: String::new(),
        playing,
        looping: false,
        gain: 1.0,
        pitch: 1.0,
        spatial: true,
        range: 20.0,
    }
}

#[test]
fn attenuation_is_one_at_the_source_and_zero_past_the_range() {
    assert_eq!(attenuation(0.0, 12.0), 1.0);
    assert_eq!(attenuation(6.0, 12.0), 0.5);
    assert_eq!(attenuation(12.0, 12.0), 0.0);
    assert_eq!(attenuation(40.0, 12.0), 0.0);
    // A range of zero is "no falloff", which is what a 2D source wants.
    assert_eq!(attenuation(0.0, 0.0), 1.0);
}

#[test]
fn attenuation_never_rises_with_distance() {
    let range = 8.0;
    let mut previous = attenuation(0.0, range);
    let mut distance = 0.25;
    while distance <= range * 2.0 {
        let gain = attenuation(distance, range);
        assert!(
            gain <= previous,
            "gain rose from {previous} to {gain} at {distance}"
        );
        assert!((0.0..=1.0).contains(&gain), "gain {gain} out of range");
        previous = gain;
        distance += 0.25;
    }
    assert_eq!(previous, 0.0);
}

#[test]
fn distance_model_selection_changes_the_curve_but_not_the_bound() {
    let range = 8.0;
    let linear = attenuation(4.0, range);
    let inverse = attenuation_modeled(AttenuationModel::Inverse, 4.0, range, 1.0);
    let exponential = attenuation_modeled(AttenuationModel::Exponential, 4.0, range, 1.0);
    assert_eq!(linear, 0.5);
    assert!(inverse < linear, "inverse {inverse} vs linear {linear}");
    assert!(
        exponential < linear,
        "exponential {exponential} vs linear {linear}"
    );
    for model in [
        AttenuationModel::Linear,
        AttenuationModel::Inverse,
        AttenuationModel::Exponential,
    ] {
        assert_eq!(attenuation_modeled(model, range, range, 1.0), 0.0);
        assert_eq!(attenuation_modeled(model, 0.0, range, 1.0), 1.0);
    }
}

#[test]
fn doppler_shifts_pitch_with_relative_motion_and_clamps() {
    use wire::doppler::doppler_factor;
    use wire::{MAX_DOPPLER, MIN_DOPPLER, SPEED_OF_SOUND};

    let listener = Listener::default();
    let source = Vec3::new(0.0, 0.0, 10.0);
    let shift = |source_velocity| {
        doppler_factor(
            SPEED_OF_SOUND,
            1.0,
            Vec3::ZERO,
            source_velocity,
            listener.position,
            source,
        )
    };
    assert_eq!(shift(Vec3::ZERO), 1.0);
    assert!(shift(Vec3::new(0.0, 0.0, -5.0)) > 1.0, "approaching rises");
    assert!(shift(Vec3::new(0.0, 0.0, 5.0)) < 1.0, "receding falls");
    // At the speed of sound the denominator is singular, so the shift is capped.
    assert_eq!(shift(Vec3::new(0.0, 0.0, -SPEED_OF_SOUND)), MAX_DOPPLER);
    assert_eq!(shift(Vec3::new(0.0, 0.0, SPEED_OF_SOUND)), MIN_DOPPLER);
}

#[test]
fn a_full_pool_keeps_the_higher_priority_voice() {
    let mut pool = VoicePool::new(1);
    let music = pool
        .spawn(
            None,
            SoundId::new(1),
            VoiceSpec {
                priority: 10,
                ..VoiceSpec::default()
            },
        )
        .expect("room for one");
    // A one-shot below the music is refused rather than silencing it.
    assert!(pool
        .spawn(
            None,
            SoundId::new(2),
            VoiceSpec {
                priority: 0,
                ..VoiceSpec::default()
            },
        )
        .is_none());
    assert!(pool.get(music).is_some());
    // An equal-priority one-shot displaces the oldest.
    let shot = pool
        .spawn(
            None,
            SoundId::new(3),
            VoiceSpec {
                priority: 10,
                ..VoiceSpec::default()
            },
        )
        .expect("steals the oldest");
    assert!(pool.get(music).is_none());
    assert!(pool.get(shot).is_some());
}

#[test]
fn panning_is_hard_left_hard_right_and_centred_in_front() {
    let listener = Listener::default();
    assert_eq!(pan(&listener, Vec3::new(-3.0, 0.0, 0.0)), -1.0);
    assert_eq!(pan(&listener, Vec3::new(3.0, 0.0, 0.0)), 1.0);
    assert_eq!(pan(&listener, Vec3::new(0.0, 0.0, -3.0)), 0.0);
    // A listener with no forward vector has no left or right to place a source on.
    assert_eq!(
        pan(&Listener::new(Vec3::ZERO, Vec3::ZERO, Vec3::Y), Vec3::X),
        0.0
    );

    // Hard left still reaches the right speaker: miniaudio's floor, because panning
    // that silenced a speaker outright "doesn't even remotely represent the real world".
    assert_eq!(channel_gains(-1.0), (1.0, wire::MIN_CHANNEL_GAIN));
    assert_eq!(channel_gains(1.0), (wire::MIN_CHANNEL_GAIN, 1.0));
}

#[test]
fn a_voice_pool_refuses_more_than_its_capacity_and_frees_the_oldest() {
    let mut pool = VoicePool::new(2);
    let spec = VoiceSpec::default();
    let first = pool
        .spawn(None, SoundId::new(1), spec)
        .expect("room for one");
    let second = pool
        .spawn(None, SoundId::new(2), spec)
        .expect("room for two");
    assert_eq!(pool.len(), 2);
    assert_eq!(pool.capacity(), 2);

    let third = pool
        .spawn(None, SoundId::new(3), spec)
        .expect("the oldest is freed to make room");
    assert_eq!(pool.len(), 2);
    assert!(pool.get(first).is_none(), "the oldest voice was dropped");
    assert!(pool.get(second).is_some());
    assert!(pool.get(third).is_some());

    // Switching audio off entirely: a pool with no capacity refuses every voice
    // rather than evicting one per spawn.
    let mut off = VoicePool::new(0);
    assert!(off.spawn(None, SoundId::new(1), spec).is_none());
    assert!(off.is_empty());
}

#[test]
fn update_spawns_exactly_one_voice_for_one_playing_source() {
    let mut scene = Scene::new();
    let entity = scene.spawn("shot");
    let mut pool = VoicePool::new(8);
    let sound = |_: &AudioSource| Some(SoundId::new(1));

    let playing = source(true);
    let position = Vec3::new(1.0, 2.0, -3.0);
    pool.reconcile([(entity, &playing, position)], sound);

    assert_eq!(pool.len(), 1, "one playing source, one voice");
    let voice = pool.voice_of(entity).expect("a voice for the source");
    assert_eq!(voice.sound, SoundId::new(1));
    assert_eq!(voice.spec.position, position);
    let id = voice.id;

    pool.reconcile([(entity, &playing, position)], sound);
    assert_eq!(pool.len(), 1, "a second frame spawns no second voice");
    assert_eq!(pool.voice_of(entity).expect("still sounding").id, id);

    let mut louder = source(true);
    louder.gain = 0.25;
    louder.looping = true;
    pool.reconcile([(entity, &louder, position)], sound);
    let voice = pool.voice_of(entity).expect("still sounding");
    assert_eq!(pool.len(), 1);
    assert_eq!(voice.id, id, "the voice is the same one");
    assert_eq!(voice.spec.gain, 0.25, "a changed gain is applied");
    assert!(voice.spec.looping);

    let stopped = source(false);
    pool.reconcile([(entity, &stopped, position)], sound);
    assert_eq!(pool.len(), 0, "a stopped source drops its voice");
}

#[test]
fn update_leaves_a_voice_whose_clip_has_not_loaded_yet_alone() {
    let mut scene = Scene::new();
    let entity = scene.spawn("music");
    let mut pool = VoicePool::new(4);
    let playing = source(true);

    pool.reconcile([(entity, &playing, Vec3::ZERO)], |_| None);
    assert_eq!(pool.len(), 0, "nothing to play yet");

    pool.reconcile([(entity, &playing, Vec3::ZERO)], |_| Some(SoundId::new(1)));
    assert_eq!(pool.len(), 1, "the next frame plays it");
}

/// Needs a sound card: `AudioEngine::new` opens the default output device.
#[test]
#[ignore = "needs an audio output device"]
fn the_default_device_opens_and_the_engine_reports_it() {
    use adam::AudioBackend;

    let engine = wire::AudioEngine::new();
    assert!(
        !engine.is_silent(),
        "this machine is expected to have a default output device"
    );
    assert_eq!(engine.name(), "wire");
    assert_eq!(engine.active_voices(), 0);
}

/// Needs a sound card: building the plugin opens the default output device.
#[test]
#[ignore = "needs an audio output device"]
fn the_plugin_installs_the_audio_backend() {
    use adam::{Engine, Plugin};

    let mut eng = Engine::new();
    let mut plugin = wire::WirePlugin::new();
    assert_eq!(plugin.name(), "wire");
    assert!(plugin.build(&mut eng).is_ok());

    let audio = eng.audio.as_ref().expect("the plugin installed a backend");
    assert_eq!(audio.name(), "wire");
    assert_eq!(audio.active_voices(), 0);
}
