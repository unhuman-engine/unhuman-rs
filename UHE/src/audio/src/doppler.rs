//! Doppler pitch from the relative motion of a listener and a source.
//!
//! The shift is the classic `(c + vl) / (c + vs)`, where `vl` and `vs` are the
//! listener's and source's speeds along the line from the listener to the source.
//! `doppler` scales both and the result is clamped to [`MIN_DOPPLER`],
//! [`MAX_DOPPLER`]: a source moving at the speed of sound makes the denominator
//! singular, and a clangour of shifted, inverted one-shots is worse than a
//! capped, plausible one. A negative `dt` yields zero velocity, not a backwards one.

use glam::Vec3;

/// Speed of sound in air at 20 degrees Celsius, in metres per second.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Doppler strength when the engine was given none: 1.0 is fully on.
pub const DEFAULT_DOPPLER_FACTOR: f32 = 1.0;

/// The lowest pitch multiplier the shift can produce.
pub const MIN_DOPPLER: f32 = 0.5;

/// The highest pitch multiplier the shift can produce.
pub const MAX_DOPPLER: f32 = 2.0;

/// The denominator floor, in metres per second, that keeps the singularity finite.
const MIN_DENOMINATOR: f32 = 1e-3;

/// Average velocity from a previous and current position. Zero when `dt` is not a
/// positive, finite step: the first frame after a spawn must not read as a jump.
pub fn velocity(previous: Vec3, current: Vec3, dt: f32) -> Vec3 {
    if !dt.is_finite() || dt <= 0.0 {
        return Vec3::ZERO;
    }
    (current - previous) / dt
}

/// The pitch multiplier for a source and listener in relative motion.
///
/// `listener_velocity` is the listener's velocity, `source_velocity` the source's,
/// and `listener_position`/`source_position` place the line between them. Returns
/// 1.0 when the two coincide, when `speed_of_sound` is not positive, or when the
/// velocities are not finite.
pub fn doppler_factor(
    speed_of_sound: f32,
    doppler: f32,
    listener_velocity: Vec3,
    source_velocity: Vec3,
    listener_position: Vec3,
    source_position: Vec3,
) -> f32 {
    if !speed_of_sound.is_finite() || speed_of_sound <= 0.0 || !doppler.is_finite() {
        return 1.0;
    }
    let offset = source_position - listener_position;
    let distance = offset.length();
    if !distance.is_finite() || distance <= f32::EPSILON {
        return 1.0;
    }
    // Listener to source.
    let axis = offset / distance;
    let numerator = speed_of_sound + doppler * listener_velocity.dot(axis);
    let denominator = (speed_of_sound + doppler * source_velocity.dot(axis)).max(MIN_DENOMINATOR);
    if !numerator.is_finite() {
        return 1.0;
    }
    (numerator / denominator).clamp(MIN_DOPPLER, MAX_DOPPLER)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factor(listener_velocity: Vec3, source_velocity: Vec3) -> f32 {
        doppler_factor(
            SPEED_OF_SOUND,
            DEFAULT_DOPPLER_FACTOR,
            listener_velocity,
            source_velocity,
            Vec3::ZERO,
            Vec3::Z,
        )
    }

    #[test]
    fn a_velocity_that_did_not_move_is_zero() {
        assert_eq!(velocity(Vec3::ZERO, Vec3::ZERO, 0.016), Vec3::ZERO);
        assert_eq!(velocity(Vec3::ZERO, Vec3::X, 0.0), Vec3::ZERO);
        assert_eq!(velocity(Vec3::ZERO, Vec3::X, -0.1), Vec3::ZERO);
        assert_eq!(velocity(Vec3::ZERO, Vec3::X, f32::NAN), Vec3::ZERO);
        assert_eq!(
            velocity(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            Vec3::X * 4.0
        );
    }

    #[test]
    fn a_source_approaching_raises_the_pitch_and_receding_lowers_it() {
        // The source sits on +Z, so moving toward the listener is -Z.
        assert!(factor(Vec3::ZERO, Vec3::NEG_Z) > 1.0);
        assert!(factor(Vec3::ZERO, Vec3::Z) < 1.0);
        // Stillness is the identity.
        assert_eq!(factor(Vec3::ZERO, Vec3::ZERO), 1.0);
    }

    #[test]
    fn listener_motion_shifts_the_pitch_the_other_way() {
        // The listener moving toward the source (on +Z) is +Z.
        assert!(factor(Vec3::Z, Vec3::ZERO) > 1.0);
        assert!(factor(Vec3::NEG_Z, Vec3::ZERO) < 1.0);
    }

    #[test]
    fn the_shift_is_clamped_and_survives_the_singularity() {
        // The source receding at the speed of sound would divide to infinity.
        let extreme = factor(Vec3::ZERO, Vec3::Z * (SPEED_OF_SOUND * 4.0));
        assert_eq!(extreme, MIN_DOPPLER);
        let toward = factor(Vec3::ZERO, Vec3::Z * -(SPEED_OF_SOUND * 4.0));
        assert_eq!(toward, MAX_DOPPLER);
    }

    #[test]
    fn coincidence_and_bad_input_are_the_identity() {
        let coincident = doppler_factor(
            SPEED_OF_SOUND,
            1.0,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        );
        assert_eq!(coincident, 1.0);
        assert!(factor(Vec3::ZERO, Vec3::ZERO).is_finite());
        assert_eq!(
            doppler_factor(0.0, 1.0, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::Z),
            1.0
        );
    }
}
