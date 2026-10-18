//! Assigning the three MSDF channels to outline edges.
//!
//! A corner survives median reconstruction only when the edges meeting there
//! feed different channels; if every edge fed every channel the three distances
//! would agree and the median would be an ordinary distance field, rounding the
//! corner off. A smooth run of edges is one spline and gets a single colour; a
//! run between two corners is one spline and each spline is cycled through the
//! channels. The scheme is msdfgen's simple colouring: 3 radian corner
//! threshold, fixed seed 0.

use glam::Vec2;

use super::shape::{Contour, EdgeKind};

/// A corner is a turn past this many radians. Near-straight joins stay smooth.
const CORNER_ANGLE: f32 = 3.0;

/// Which channels an edge feeds. Bit 0 is red, bit 1 green, bit 2 blue.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EdgeColor(u8);

impl EdgeColor {
    pub const BLACK: Self = Self(0);
    pub const RED: Self = Self(1);
    pub const GREEN: Self = Self(2);
    pub const BLUE: Self = Self(4);
    pub const WHITE: Self = Self(7);

    /// Two-channel colours the seed starts from and cycles through.
    const CYAN: Self = Self(6);
    const MAGENTA: Self = Self(5);
    const YELLOW: Self = Self(3);

    /// True when `channel` (a single channel) is one of this edge's channels.
    pub fn contains(self, channel: Self) -> bool {
        self.0 & channel.0 != 0
    }

    fn from_bits(bits: u8) -> Self {
        Self(bits & 7)
    }

    fn bits(self) -> u8 {
        self.0
    }
}

/// The low bits of a running counter, so the choice of channels is deterministic
/// without storing state the caller cares about. msdfgen draws the same way.
struct Seed(u64);

impl Seed {
    fn bit(&mut self) -> u8 {
        let bit = (self.0 & 1) as u8;
        self.0 >>= 1;
        bit
    }

    fn ternary(&mut self) -> u8 {
        let value = (self.0 % 3) as u8;
        self.0 /= 3;
        value
    }
}

fn is_corner(before: Vec2, after: Vec2) -> bool {
    let before = before.normalize_or_zero();
    let after = after.normalize_or_zero();
    before.dot(after) <= 0.0 || (before.x * after.y - before.y * after.x).abs() > CORNER_ANGLE.sin()
}

/// Rotate a colour to a different one. Never returns the same colour back.
fn switch(color: &mut EdgeColor, seed: &mut Seed) {
    let shifted = color.bits() << (1 + seed.bit());
    *color = EdgeColor::from_bits(shifted | (shifted >> 3));
}

/// Like [`switch`], but avoids landing on `banned` when `color` is a single
/// channel that would otherwise be forced there.
fn switch_avoiding(color: &mut EdgeColor, seed: &mut Seed, banned: EdgeColor) {
    let combined = EdgeColor::from_bits(color.bits() & banned.bits());
    // A single shared channel: take the other two, the only way not to clash.
    if combined.bits().count_ones() == 1 {
        *color = EdgeColor::from_bits(combined.bits() ^ EdgeColor::WHITE.bits());
    } else {
        switch(color, seed);
    }
}

/// -1, 0 or 1, balanced over `0..n`, so a teardrop's three colours are used
/// equally around the contour.
fn symmetrical_trichotomy(position: usize, count: usize) -> i32 {
    ((3 * position as i32 + 1) / count as i32) - 1
}

fn find_corners(contour: &Contour) -> Vec<usize> {
    let count = contour.edges.len();
    let mut corners = Vec::new();
    let mut before = contour.edges[count - 1].kind.direction_end();
    for (index, edge) in contour.edges.iter().enumerate() {
        if is_corner(before, edge.kind.direction_start()) {
            corners.push(index);
        }
        before = edge.kind.direction_end();
    }
    corners
}

/// Colour every edge of every contour in place.
pub fn color_contours(contours: &mut [Contour]) {
    let mut seed = Seed(0);
    let mut color =
        [EdgeColor::CYAN, EdgeColor::MAGENTA, EdgeColor::YELLOW][seed.ternary() as usize];

    for contour in contours {
        if contour.edges.is_empty() {
            continue;
        }
        let corners = find_corners(contour);
        match corners.len() {
            0 => {
                switch(&mut color, &mut seed);
                for edge in &mut contour.edges {
                    edge.color = color;
                }
            }
            1 => color_teardrop(contour, corners[0], &mut color, &mut seed),
            _ => color_multi_corner(contour, &corners, &mut color, &mut seed),
        }
    }
}

/// One corner means the contour is a teardrop. The middle of the long side is
/// white so the two channels coming into the point stay separable.
fn color_teardrop(contour: &mut Contour, corner: usize, color: &mut EdgeColor, seed: &mut Seed) {
    switch(color, seed);
    let first = *color;
    let middle = EdgeColor::WHITE;
    switch(color, seed);
    let last = *color;
    let colors = [first, middle, last];

    let count = contour.edges.len();
    if count >= 3 {
        for index in 0..count {
            let scheme = 1 + symmetrical_trichotomy(index, count);
            contour.edges[(corner + index) % count].color = colors[scheme as usize];
        }
        return;
    }

    // Fewer than three edges for three colours: split them and retag the pieces.
    // The pieces are ordered so the corner starts the run, then coloured in
    // pairs, which is what keeps the two long sides on separate channels.
    let first_third = contour.edges[0].kind.split_in_thirds();
    let parts: Vec<EdgeKind> = if count == 1 {
        first_third.to_vec()
    } else {
        let second_third = contour.edges[1].kind.split_in_thirds();
        let [f0, f1, f2] = first_third;
        let [s0, s1, s2] = second_third;
        if corner == 0 {
            vec![f0, f1, f2, s0, s1, s2]
        } else {
            vec![s0, s1, s2, f0, f1, f2]
        }
    };

    let assignments: &[usize] = if count == 1 {
        &[0, 1, 2]
    } else {
        &[0, 0, 1, 1, 2, 2]
    };
    contour.edges = parts
        .into_iter()
        .zip(assignments.iter())
        .map(|(kind, &scheme)| super::shape::Edge {
            kind,
            color: colors[scheme],
        })
        .collect();
}

/// Several corners: cut the contour into splines at each corner and hand out
/// channels so no two splines entering the same corner share one.
fn color_multi_corner(
    contour: &mut Contour,
    corners: &[usize],
    color: &mut EdgeColor,
    seed: &mut Seed,
) {
    let corner_count = corners.len();
    let start = corners[0];
    let count = contour.edges.len();

    switch(color, seed);
    let initial = *color;
    let mut spline = 0;
    for step in 0..count {
        let index = (start + step) % count;
        if spline + 1 < corner_count && corners[spline + 1] == index {
            spline += 1;
            let banned = if spline == corner_count - 1 {
                initial
            } else {
                EdgeColor::BLACK
            };
            switch_avoiding(color, seed, banned);
        }
        contour.edges[index].color = *color;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::shape::Edge;

    fn line(a: Vec2, b: Vec2) -> Edge {
        Edge {
            kind: EdgeKind::Line { a, b },
            color: EdgeColor::BLACK,
        }
    }

    fn square() -> Contour {
        Contour {
            edges: vec![
                line(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
                line(Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)),
                line(Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0)),
                line(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
            ],
        }
    }

    #[test]
    fn a_square_gets_at_least_two_distinct_colours() {
        let mut contours = vec![square()];
        color_contours(&mut contours);

        let colors: Vec<EdgeColor> = contours[0].edges.iter().map(|e| e.color).collect();
        assert!(colors.iter().all(|c| *c != EdgeColor::BLACK));
        assert!(
            colors.windows(2).any(|pair| pair[0] != pair[1]),
            "one colour for every edge is the degenerate case: {colors:?}"
        );
    }

    #[test]
    fn a_smooth_contour_gets_one_colour() {
        // A 72-gon: every joint turns 5 degrees, well under the corner
        // threshold, so the whole contour is one spline.
        let segments = 72;
        let mut edges = Vec::new();
        for i in 0..segments {
            let angle = std::f32::consts::TAU * i as f32 / segments as f32;
            let next = std::f32::consts::TAU * (i + 1) as f32 / segments as f32;
            edges.push(line(
                Vec2::new(angle.cos(), angle.sin()),
                Vec2::new(next.cos(), next.sin()),
            ));
        }
        let mut contours = vec![Contour { edges }];
        color_contours(&mut contours);

        let first = contours[0].edges[0].color;
        assert!(contours[0].edges.iter().all(|e| e.color == first));
    }

    #[test]
    fn a_right_turn_is_a_corner() {
        assert!(is_corner(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)));
        assert!(!is_corner(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.01)));
    }
}
