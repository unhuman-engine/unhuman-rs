//! Distance falloff: linear, inverse and exponential, all bounded to `range`.
//!
//! Gain is 1.0 at the listener and 0.0 at `range`, so a source always has a
//! distance it is silent past. The classic inverse and exponential curves only
//! approach zero, which is why linear was the original choice; here the other two
//! are renormalised onto [0, 1] so the bound holds. `rolloff` shapes inverse and
//! exponential, and 1.0 is neutral.

/// The curve a spatial voice's gain follows with distance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttenuationModel {
    #[default]
    Linear,
    Inverse,
    Exponential,
}

/// The rolloff used when the engine was given none.
pub const DEFAULT_ROLLOFF: f32 = 1.0;

/// The smallest rolloff the shaped curves accept, so their normalisation never
/// divides by zero: below it they limit to the linear curve anyway.
const MIN_ROLLOFF: f32 = 1e-3;

/// Linear falloff: 1.0 at the listener, 0.0 at `range` and beyond. A range of
/// zero or less means no falloff at all.
pub fn attenuation(distance: f32, range: f32) -> f32 {
    attenuation_modeled(AttenuationModel::Linear, distance, range, DEFAULT_ROLLOFF)
}

/// Gain under `model`. Every model is continuous, monotone non-increasing, 1.0
/// at the listener and 0.0 at `range`; a range of zero or less disables falloff.
///
/// Inverse and exponential are the classic curves normalised between their value
/// at the listener and their value at `range`, so both still reach silence there.
pub fn attenuation_modeled(
    model: AttenuationModel,
    distance: f32,
    range: f32,
    rolloff: f32,
) -> f32 {
    let Some(x) = normalized(distance, range) else {
        return 1.0;
    };
    match model {
        AttenuationModel::Linear => 1.0 - x,
        AttenuationModel::Inverse => inverse(x, rolloff),
        AttenuationModel::Exponential => exponential(x, rolloff),
    }
}

/// `distance` as a fraction of `range`, clamped to [0, 1]. `None` is "no falloff".
fn normalized(distance: f32, range: f32) -> Option<f32> {
    if !range.is_finite() || range <= 0.0 {
        return None;
    }
    Some((distance / range).clamp(0.0, 1.0))
}

/// `1 / (1 + k*x)` renormalised to hit 0 at x = 1.
fn inverse(x: f32, rolloff: f32) -> f32 {
    let k = rolloff.max(MIN_ROLLOFF);
    let at_range = 1.0 / (1.0 + k);
    (1.0 / (1.0 + k * x) - at_range) / (1.0 - at_range)
}

/// `exp(-k*x)` renormalised to hit 0 at x = 1.
fn exponential(x: f32, rolloff: f32) -> f32 {
    let k = rolloff.max(MIN_ROLLOFF);
    let at_range = (-k).exp();
    ((-k * x).exp() - at_range) / (1.0 - at_range)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS: [AttenuationModel; 3] = [
        AttenuationModel::Linear,
        AttenuationModel::Inverse,
        AttenuationModel::Exponential,
    ];

    #[test]
    fn linear_is_one_at_the_listener_and_zero_at_the_range() {
        assert_eq!(attenuation(0.0, 10.0), 1.0);
        assert_eq!(attenuation(5.0, 10.0), 0.5);
        assert_eq!(attenuation(10.0, 10.0), 0.0);
        assert_eq!(attenuation(99.0, 10.0), 0.0);
        assert_eq!(attenuation(0.0, 0.0), 1.0);
    }

    #[test]
    fn every_model_starts_at_one_and_reaches_silence() {
        for model in MODELS {
            assert_eq!(attenuation_modeled(model, 0.0, 8.0, 1.0), 1.0);
            assert_eq!(attenuation_modeled(model, 8.0, 8.0, 1.0), 0.0);
            assert_eq!(attenuation_modeled(model, 80.0, 8.0, 1.0), 0.0);
            assert_eq!(attenuation_modeled(model, 0.0, 0.0, 1.0), 1.0);
        }
    }

    #[test]
    fn every_model_is_monotone_and_in_range() {
        for model in MODELS {
            let range = 12.0;
            let mut previous = 1.0;
            let mut distance = 0.0;
            while distance <= range * 1.5 {
                let gain = attenuation_modeled(model, distance, range, 2.0);
                assert!(
                    (0.0..=1.0).contains(&gain),
                    "{model:?} gain {gain} out of range"
                );
                assert!(gain <= previous + 1e-6, "{model:?} gain rose with distance");
                previous = gain;
                distance += 0.25;
            }
            assert_eq!(previous, 0.0);
        }
    }

    #[test]
    fn inverse_and_exponential_fall_faster_than_linear_near_the_source() {
        let linear = attenuation(4.0, 8.0);
        let inverse = attenuation_modeled(AttenuationModel::Inverse, 4.0, 8.0, 1.0);
        let exponential = attenuation_modeled(AttenuationModel::Exponential, 4.0, 8.0, 1.0);
        assert!(inverse < linear, "inverse {inverse} vs linear {linear}");
        assert!(
            exponential < linear,
            "exponential {exponential} vs linear {linear}"
        );
    }
}
