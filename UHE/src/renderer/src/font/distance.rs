//! Signed distance to the coloured outline.
//!
//! For each sample the shape is tested once for inside/outside, then each
//! channel is the distance to the nearest edge carrying that channel. The sign
//! is shared by all three channels, which is the point: near a corner the two
//! distances to the corner's own edges stay small while the third channel is
//! measured from somewhere else, so `median3` can pick a different channel than
//! a plain distance field would.

use glam::Vec2;

use super::edge_color::EdgeColor;
use super::shape::{point_segment_distance, Segment};

/// Nonzero winding: a point is inside when a ray crosses the outline more times
/// one way than the other. Handles nested contours (a counter has opposite
/// winding) the way the fill rule does.
pub fn winding_number(segments: &[Segment], p: Vec2) -> i32 {
    let mut winding = 0;
    for segment in segments {
        let (a, b) = (segment.a, segment.b);
        let left = (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
        if a.y <= p.y {
            if b.y > p.y && left > 0.0 {
                winding += 1;
            }
        } else if b.y <= p.y && left < 0.0 {
            winding -= 1;
        }
    }
    winding
}

/// Signed distance to each channel in pixels, positive inside. A shape with no
/// edges yields `-range`, so the encoder never sees a non-finite value.
pub fn channel_distances(segments: &[Segment], p: Vec2, range: f32) -> [f32; 3] {
    if segments.is_empty() {
        return [-range; 3];
    }
    let sign = if winding_number(segments, p) != 0 {
        1.0
    } else {
        -1.0
    };

    let mut nearest = [f32::INFINITY; 3];
    for segment in segments {
        let distance = point_segment_distance(p, segment.a, segment.b);
        if segment.color.contains(EdgeColor::RED) {
            nearest[0] = nearest[0].min(distance);
        }
        if segment.color.contains(EdgeColor::GREEN) {
            nearest[1] = nearest[1].min(distance);
        }
        if segment.color.contains(EdgeColor::BLUE) {
            nearest[2] = nearest[2].min(distance);
        }
    }

    for value in &mut nearest {
        // A channel with no edge is off the shape; saturate it at the range.
        if !value.is_finite() {
            *value = range;
        }
        *value *= sign;
    }
    nearest
}

/// The field's encoding: 0.5 on the edge, above 0.5 inside, one range unit from
/// 0 to 1. `reconstruct::coverage` is the inverse.
pub fn encode(distance: f32, range: f32) -> u8 {
    debug_assert!(range > 0.0);
    let value = (0.5 + 0.5 * distance / range).clamp(0.0, 1.0);
    (value * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::reconstruct::median3;
    use crate::font::shape::EdgeKind;

    const RANGE: f32 = 2.0;

    fn segment(a: (f32, f32), b: (f32, f32), color: EdgeColor) -> Segment {
        Segment {
            a: Vec2::new(a.0, a.1),
            b: Vec2::new(b.0, b.1),
            color,
        }
    }

    /// A 2x2 square, wound so the interior is `(0,0)..(2,2)`.
    fn square() -> Vec<Segment> {
        vec![
            segment((0.0, 0.0), (2.0, 0.0), EdgeColor::RED),
            segment((2.0, 0.0), (2.0, 2.0), EdgeColor::GREEN),
            segment((2.0, 2.0), (0.0, 2.0), EdgeColor::BLUE),
            segment((0.0, 2.0), (0.0, 0.0), EdgeColor::RED),
        ]
    }

    #[test]
    fn a_straight_edge_is_positive_inside_and_negative_outside() {
        let square = square();
        let inside = channel_distances(&square, Vec2::new(1.0, 1.0), RANGE);
        let outside = channel_distances(&square, Vec2::new(3.0, 1.0), RANGE);

        assert!(inside.iter().all(|value| *value > 0.0), "{inside:?}");
        assert!(outside.iter().all(|value| *value < 0.0), "{outside:?}");
        // The nearest edge is one unit away on each side.
        assert!((inside.iter().copied().fold(f32::INFINITY, f32::min) - 1.0).abs() < 1e-4);
        assert!((outside.iter().copied().fold(f32::NEG_INFINITY, f32::max) + 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_right_angle_corner_keeps_the_channels_distinct() {
        // Two edges meet at the origin; the third edge is far away. Distinct
        // channels are what lets the median keep the corner instead of rounding
        // it: if all three distances were equal there would be no corner to keep.
        let corner = vec![
            segment((0.0, 0.0), (4.0, 0.0), EdgeColor::RED),
            segment((0.0, 0.0), (0.0, 4.0), EdgeColor::GREEN),
            segment((4.0, 0.0), (0.0, 4.0), EdgeColor::BLUE),
        ];
        let distances = channel_distances(&corner, Vec2::new(0.5, 2.0), RANGE);

        assert!(distances.iter().all(|value| value.is_finite()));
        // Nearer the left edge than the bottom, nearer the bottom than the
        // hypotenuse: strictly ordered, so none of the channels is degenerate.
        assert!(distances[1] < distances[2], "{distances:?}");
        assert!(distances[2] < distances[0], "{distances:?}");
        // The reconstruction takes the middle, not the nearest. A plain field
        // would have all three equal and collapse to the minimum; the spread is
        // what keeps the corner instead of rounding it.
        let median = median3(distances[0], distances[1], distances[2]);
        assert_eq!(median, distances[2]);
        assert!(median > distances[1] && median < distances[0]);
    }

    #[test]
    fn a_flattened_quadratic_is_within_tolerance_of_the_curve() {
        let curve = EdgeKind::Quad {
            a: Vec2::new(0.0, 0.0),
            c: Vec2::new(1.0, 3.0),
            b: Vec2::new(4.0, 0.0),
        };
        let mut points = Vec::new();
        curve.polyline(|p| p, 0.01, &mut points);
        let segments: Vec<Segment> = points
            .windows(2)
            .map(|pair| {
                segment(
                    (pair[0].x, pair[0].y),
                    (pair[1].x, pair[1].y),
                    EdgeColor::RED,
                )
            })
            .collect();

        let evaluate = |t: f32| {
            let inv = 1.0 - t;
            Vec2::new(0.0, 0.0) * inv * inv
                + Vec2::new(1.0, 3.0) * (2.0 * inv * t)
                + Vec2::new(4.0, 0.0) * t * t
        };

        for probe in [
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(3.0, 3.0),
            Vec2::new(2.0, -1.0),
        ] {
            let flattened = segments
                .iter()
                .map(|s| point_segment_distance(probe, s.a, s.b))
                .fold(f32::INFINITY, f32::min);
            let exact = (0..=4000)
                .map(|i| (probe - evaluate(i as f32 / 4000.0)).length())
                .fold(f32::INFINITY, f32::min);
            assert!(
                (flattened - exact).abs() < 0.02,
                "probe {probe:?}: flattened {flattened}, exact {exact}"
            );
        }
    }

    #[test]
    fn an_empty_shape_is_finite_and_encoded() {
        let distances = channel_distances(&[], Vec2::ZERO, RANGE);
        assert!(distances.iter().all(|value| value.is_finite()));
        for value in distances {
            let _ = encode(value, RANGE);
        }
    }
}
