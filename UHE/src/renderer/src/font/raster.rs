//! Sampling a coloured outline into an MSDF bitmap.
//!
//! The distance range decides the encoding and the border: a texel further than
//! `range` from the outline cannot be told from the one beyond it, so the field
//! is padded past the rounded range and the border saturates. Sampling is at
//! texel centres, or every glyph gains a half-pixel bias and drifts off its quad.

use ab_glyph::GlyphId;
use glam::Vec2;

use crate::font::distance;
use crate::font::shape::Segment;

pub(crate) struct Raster {
    pub(crate) codepoint: u32,
    pub(crate) glyph_id: GlyphId,
    pub(crate) advance: f32,
    pub(crate) plane_min: Vec2,
    pub(crate) plane_max: Vec2,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) rgba: Vec<u8>,
}

impl Raster {
    /// A glyph with no field yet: whitespace, or an empty outline. The advance
    /// and the font metrics still come through; the quad stays zero.
    pub(crate) fn empty(codepoint: u32, glyph_id: GlyphId, advance: f32) -> Self {
        Self {
            codepoint,
            glyph_id,
            advance,
            plane_min: Vec2::ZERO,
            plane_max: Vec2::ZERO,
            width: 0,
            height: 0,
            rgba: Vec::new(),
        }
    }

    /// Sample `segments` into an RGBA bitmap and set the plane bounds its quad
    /// maps to. The padding is one more than the rounded range so the field
    /// reaches its saturated limits before the bitmap border.
    pub(crate) fn fill_msdf(&mut self, segments: &[Segment], em: f32, range: f32) {
        let padding = range.ceil() as i32 + 1;
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for segment in segments {
            min = min.min(segment.a).min(segment.b);
            max = max.max(segment.a).max(segment.b);
        }

        let x0 = min.x.floor() as i32 - padding;
        let y0 = min.y.floor() as i32 - padding;
        let x1 = max.x.ceil() as i32 + padding;
        let y1 = max.y.ceil() as i32 + padding;
        let width = (x1 - x0).max(1) as u32;
        let height = (y1 - y0).max(1) as u32;

        let mut rgba = vec![0u8; width as usize * height as usize * 4];
        for row in 0..height {
            for column in 0..width {
                let sample = Vec2::new(
                    x0 as f32 + column as f32 + 0.5,
                    y0 as f32 + row as f32 + 0.5,
                );
                let distances = distance::channel_distances(segments, sample, range);
                let texel = (row as usize * width as usize + column as usize) * 4;
                rgba[texel] = distance::encode(distances[0], range);
                rgba[texel + 1] = distance::encode(distances[1], range);
                rgba[texel + 2] = distance::encode(distances[2], range);
                rgba[texel + 3] = 255;
            }
        }

        self.plane_min = Vec2::new(x0 as f32 / em, -(y1 as f32) / em);
        self.plane_max = Vec2::new(x1 as f32 / em, -(y0 as f32) / em);
        self.width = width;
        self.height = height;
        self.rgba = rgba;
    }
}
