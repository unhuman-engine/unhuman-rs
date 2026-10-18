//! Audio: the mixer, the backend, spatialisation.
//!
//! [`AudioEngine`] owns the voices and the clip table, [`backend::Backend`] owns the
//! rodio device, and [`WirePlugin`] installs the engine into `Engine::audio`.
//!
//! Three behaviours are chosen rather than left to a library default. Distance
//! falloff reaches silence at the source's `range` under every [`AttenuationModel`],
//! where the classic inverse curve only reaches it at infinity. Pitch shifts from
//! relative motion (Doppler) are clamped to a documented range. And the number of
//! simultaneous voices is bounded, with a priority policy for what a full table
//! gives up.

pub mod attenuation;
pub mod backend;
pub mod doppler;
pub mod engine;
pub mod mixer;
pub mod plugin;
pub mod sound;
pub mod spatial;
pub mod spec;
pub mod voice;

/// The listener pose and the voice handle are `adam`'s: they cross the
/// `AudioBackend` seam, and a second definition here would only be a second thing
/// to convert between.
pub use adam::{Listener, VoiceId};

pub use attenuation::{attenuation, attenuation_modeled, AttenuationModel, DEFAULT_ROLLOFF};
pub use doppler::{DEFAULT_DOPPLER_FACTOR, MAX_DOPPLER, MIN_DOPPLER, SPEED_OF_SOUND};
pub use engine::{AudioEngine, DEFAULT_VOICES};
pub use mixer::Mixer;
pub use plugin::{listener_from, WirePlugin};
pub use sound::{Bus, Sound, SoundId, SoundTable};
pub use spatial::{channel_gains, pan, Pan, PanHandle, MIN_CHANNEL_GAIN};
pub use spec::{VoiceSpec, DEFAULT_PRIORITY, DEFAULT_RANGE};
pub use voice::{Voice, VoicePool};
