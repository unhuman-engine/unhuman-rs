//! Glyph outlines as MSDF edges.
//!
//! ab_glyph hands back a flat list of curves with no contour separators, so the
//! first job is to regroup them: a contour starts wherever a curve does not
//! begin where the previous one ended, and every real contour is closed. Curves
//! stay curves, because edge colouring needs their true end tangents; only the
//! distance pass sees flattened polylines.
//!
//! Coordinates are the face's own font units, y up. Scaling to pixels is a
//! closure the caller supplies, so this module never assumes a sign convention.

use ab_glyph::{OutlineCurve, Point};
use glam::Vec2;

use crate::font::edge_color::EdgeColor;

/// How far the flattened polyline may leave the true curve, in pixels. A tenth
/// of a pixel is below the space between two texels at text sizes, and it keeps
/// the segment count small enough that sampling stays cheap.
pub const FLATTEN_TOLERANCE: f32 = 0.1;

/// Subdivision guard. A curve that is still not flat after this many halvings is
/// pathological; stopping is better than recursing forever.
const MAX_SUBDIVISION: u32 = 16;

/// One outline segment. The colour is which channels its distance feeds.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub kind: EdgeKind,
    pub color: EdgeColor,
}

/// A line, quadratic or cubic Bezier, in font units.
#[derive(Clone, Copy, Debug)]
pub enum EdgeKind {
    Line {
        a: Vec2,
        b: Vec2,
    },
    Quad {
        a: Vec2,
        c: Vec2,
        b: Vec2,
    },
    Cubic {
        a: Vec2,
        c1: Vec2,
        c2: Vec2,
        b: Vec2,
    },
}

/// A closed run of edges. The last edge ends where the first begins.
#[derive(Clone, Debug, Default)]
pub struct Contour {
    pub edges: Vec<Edge>,
}

/// A flattened straight piece, carrying the colour of the curve it came from.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub a: Vec2,
    pub b: Vec2,
    pub color: EdgeColor,
}

fn point(p: Point) -> Vec2 {
    Vec2::new(p.x, p.y)
}

/// Distance from `p` to the segment `a`-`b`.
pub fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let length_squared = ab.length_squared();
    if length_squared <= 0.0 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / length_squared).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

impl EdgeKind {
    pub fn start(self) -> Vec2 {
        match self {
            EdgeKind::Line { a, .. } | EdgeKind::Quad { a, .. } | EdgeKind::Cubic { a, .. } => a,
        }
    }

    pub fn end(self) -> Vec2 {
        match self {
            EdgeKind::Line { b, .. } | EdgeKind::Quad { b, .. } | EdgeKind::Cubic { b, .. } => b,
        }
    }

    /// Tangent at the start, scaled arbitrarily. Corners only compare its angle.
    pub fn direction_start(self) -> Vec2 {
        match self {
            EdgeKind::Line { a, b } => b - a,
            EdgeKind::Quad { a, c, .. } => 2.0 * (c - a),
            EdgeKind::Cubic { a, c1, .. } => 3.0 * (c1 - a),
        }
    }

    /// Tangent at the end, scaled arbitrarily.
    pub fn direction_end(self) -> Vec2 {
        match self {
            EdgeKind::Line { a, b } => b - a,
            EdgeKind::Quad { c, b, .. } => 2.0 * (b - c),
            EdgeKind::Cubic { c2, b, .. } => 3.0 * (b - c2),
        }
    }

    /// Append this curve to `out` as a polyline of pixel-space points.
    pub(crate) fn polyline(
        self,
        to_pixels: impl Fn(Vec2) -> Vec2 + Copy,
        tolerance: f32,
        out: &mut Vec<Vec2>,
    ) {
        match self {
            EdgeKind::Line { a, b } => {
                out.push(to_pixels(a));
                out.push(to_pixels(b));
            }
            EdgeKind::Quad { a, c, b } => {
                out.push(to_pixels(a));
                flatten_quad(to_pixels(a), to_pixels(c), to_pixels(b), tolerance, 0, out);
            }
            EdgeKind::Cubic { a, c1, c2, b } => {
                out.push(to_pixels(a));
                flatten_cubic(
                    to_pixels(a),
                    to_pixels(c1),
                    to_pixels(c2),
                    to_pixels(b),
                    tolerance,
                    0,
                    out,
                );
            }
        }
    }

    /// Cut the curve at `t` into two curves of the same kind.
    fn split_at(self, t: f32) -> (EdgeKind, EdgeKind) {
        match self {
            EdgeKind::Line { a, b } => {
                let m = a.lerp(b, t);
                (EdgeKind::Line { a, b: m }, EdgeKind::Line { a: m, b })
            }
            EdgeKind::Quad { a, c, b } => {
                let p01 = a.lerp(c, t);
                let p12 = c.lerp(b, t);
                let p012 = p01.lerp(p12, t);
                (
                    EdgeKind::Quad { a, c: p01, b: p012 },
                    EdgeKind::Quad { a: p012, c: p12, b },
                )
            }
            EdgeKind::Cubic { a, c1, c2, b } => {
                let p01 = a.lerp(c1, t);
                let p12 = c1.lerp(c2, t);
                let p23 = c2.lerp(b, t);
                let p012 = p01.lerp(p12, t);
                let p123 = p12.lerp(p23, t);
                let p0123 = p012.lerp(p123, t);
                (
                    EdgeKind::Cubic {
                        a,
                        c1: p01,
                        c2: p012,
                        b: p0123,
                    },
                    EdgeKind::Cubic {
                        a: p0123,
                        c1: p123,
                        c2: p23,
                        b,
                    },
                )
            }
        }
    }

    /// Three equal-parameter pieces, in order. Only teardrop colouring needs it.
    pub(crate) fn split_in_thirds(self) -> [EdgeKind; 3] {
        let (first, rest) = self.split_at(1.0 / 3.0);
        let (second, third) = rest.split_at(0.5);
        [first, second, third]
    }
}

fn flatten_quad(p0: Vec2, p1: Vec2, p2: Vec2, tolerance: f32, depth: u32, out: &mut Vec<Vec2>) {
    if depth >= MAX_SUBDIVISION || point_segment_distance(p1, p0, p2) <= tolerance {
        out.push(p2);
        return;
    }
    let p01 = (p0 + p1) * 0.5;
    let p12 = (p1 + p2) * 0.5;
    let p012 = (p01 + p12) * 0.5;
    flatten_quad(p0, p01, p012, tolerance, depth + 1, out);
    flatten_quad(p012, p12, p2, tolerance, depth + 1, out);
}

fn flatten_cubic(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    tolerance: f32,
    depth: u32,
    out: &mut Vec<Vec2>,
) {
    let deviation = point_segment_distance(p1, p0, p3).max(point_segment_distance(p2, p0, p3));
    if depth >= MAX_SUBDIVISION || deviation <= tolerance {
        out.push(p3);
        return;
    }
    let p01 = (p0 + p1) * 0.5;
    let p12 = (p1 + p2) * 0.5;
    let p23 = (p2 + p3) * 0.5;
    let p012 = (p01 + p12) * 0.5;
    let p123 = (p12 + p23) * 0.5;
    let p0123 = (p012 + p123) * 0.5;
    flatten_cubic(p0, p01, p012, p0123, tolerance, depth + 1, out);
    flatten_cubic(p0123, p123, p23, p3, tolerance, depth + 1, out);
}

/// Regroup ab_glyph's flat curve list into closed contours, dropping the
/// zero-length edges a face emits to close a contour it has already closed.
pub fn contours(curves: &[OutlineCurve]) -> Vec<Contour> {
    let mut kinds = Vec::with_capacity(curves.len());
    for curve in curves {
        let kind = match *curve {
            OutlineCurve::Line(p0, p1) => EdgeKind::Line {
                a: point(p0),
                b: point(p1),
            },
            OutlineCurve::Quad(p0, p1, p2) => EdgeKind::Quad {
                a: point(p0),
                c: point(p1),
                b: point(p2),
            },
            OutlineCurve::Cubic(p0, p1, p2, p3) => EdgeKind::Cubic {
                a: point(p0),
                c1: point(p1),
                c2: point(p2),
                b: point(p3),
            },
        };
        if (kind.end() - kind.start()).length_squared() > 0.0 {
            kinds.push(kind);
        }
    }

    let mut contours = Vec::new();
    let mut current: Vec<Edge> = Vec::new();
    for kind in kinds {
        // A curve whose start is not the previous end belongs to a new contour.
        if let Some(last) = current.last() {
            if last.kind.end() != kind.start() {
                close_contour(&mut contours, &mut current);
            }
        }
        current.push(Edge {
            kind,
            color: EdgeColor::BLACK,
        });
    }
    close_contour(&mut contours, &mut current);
    contours
}

fn close_contour(contours: &mut Vec<Contour>, current: &mut Vec<Edge>) {
    if current.is_empty() {
        return;
    }
    let first = current[0].kind.start();
    let last = current.last().expect("seen a first edge").kind.end();
    if last != first {
        current.push(Edge {
            kind: EdgeKind::Line { a: last, b: first },
            color: EdgeColor::BLACK,
        });
    }
    contours.push(Contour {
        edges: std::mem::take(current),
    });
}

/// Flatten every contour to pixel-space segments, each tagged with its curve's
/// colour. `to_pixels` is where the caller flips y and applies the em scale.
pub fn flatten(
    contours: &[Contour],
    to_pixels: impl Fn(Vec2) -> Vec2 + Copy,
    tolerance: f32,
    out: &mut Vec<Segment>,
) {
    let mut polyline = Vec::new();
    for contour in contours {
        for edge in &contour.edges {
            polyline.clear();
            edge.kind.polyline(to_pixels, tolerance, &mut polyline);
            for pair in polyline.windows(2) {
                out.push(Segment {
                    a: pair[0],
                    b: pair[1],
                    color: edge.color,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: fn(Vec2) -> Vec2 = |p| p;

    #[test]
    fn a_straight_line_flattens_to_one_segment() {
        let line = EdgeKind::Line {
            a: Vec2::new(0.0, 0.0),
            b: Vec2::new(3.0, 4.0),
        };
        let mut polyline = Vec::new();
        line.polyline(IDENTITY, FLATTEN_TOLERANCE, &mut polyline);
        assert_eq!(polyline, vec![Vec2::new(0.0, 0.0), Vec2::new(3.0, 4.0)]);
    }

    #[test]
    fn a_curve_flattens_to_more_than_one_segment() {
        // A cubic that is not a line cannot be one segment within tolerance.
        let curve = EdgeKind::Cubic {
            a: Vec2::new(0.0, 0.0),
            c1: Vec2::new(0.0, 4.0),
            c2: Vec2::new(4.0, 4.0),
            b: Vec2::new(4.0, 0.0),
        };
        let mut polyline = Vec::new();
        curve.polyline(IDENTITY, FLATTEN_TOLERANCE, &mut polyline);
        assert!(polyline.len() > 2);
        // Every emitted point lies on or near the curve, and the ends are exact.
        assert_eq!(polyline.first().copied(), Some(Vec2::new(0.0, 0.0)));
        assert_eq!(polyline.last().copied(), Some(Vec2::new(4.0, 0.0)));
    }

    #[test]
    fn a_disjoint_curve_starts_a_second_contour() {
        // Two curves that do not share an endpoint are two contours; a face
        // emits them back to back, so the split has to be by endpoint.
        let curves = [
            OutlineCurve::Line(ab_glyph::point(0.0, 0.0), ab_glyph::point(1.0, 0.0)),
            OutlineCurve::Line(ab_glyph::point(5.0, 5.0), ab_glyph::point(6.0, 5.0)),
        ];
        let contours = contours(&curves);
        assert_eq!(contours.len(), 2);
        // Each was implicitly closed with the line back to its start.
        assert_eq!(contours[0].edges.len(), 2);
        assert_eq!(contours[1].edges.len(), 2);
    }
}
