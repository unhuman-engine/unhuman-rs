//! Runtime MSDF atlas generation from a TTF.
//!
//! Load the face, take a glyph set, outline each glyph, colour and flatten its
//! curves, sample a signed distance per channel, pack, and hand back the same
//! record the JSON parser produces. It exists because `fonts/Inter.ttf` ships
//! with the engine and the default face is rasterized at startup rather than
//! loaded from a baked atlas.
//!
//! ## The field and its range
//!
//! Every texel stores, per colour channel, the signed distance to the nearest
//! outline edge carrying that channel, encoded as `0.5 + 0.5 * distance / range`
//! ([`crate::font::distance`]); positive is inside. The glyph is padded by one
//! more than the rounded range so the field saturates before the bitmap border,
//! which leaves the border texels all inside or all outside and safe to clamp.
//! The range is [`DISTANCE_RANGE`] pixels and travels to the shader as
//! [`Metrics::distance_range`], where [`crate::font::reconstruct`] widens it back
//! to screen pixels. See [`crate::font::edge_color`] for why the channels differ.
//!
//! The coordinates are the atlas JSON's, not a second convention: plane bounds in
//! em units with y up from the baseline, UVs normalized and y down from the top
//! of the image, `advance` in em units, `line_height` the font's own. `layout`
//! therefore cannot tell which of the two producers made the atlas.

use std::collections::HashMap;

use ab_glyph::{Font, FontRef, ScaleFont};
use glam::Vec2;

use crate::font::atlas::{Atlas, Glyph, Metrics};
use crate::font::edge_color;
use crate::font::raster::Raster;
use crate::font::shape;

/// Rasterization size. The field is sampled at this resolution.
pub const DEFAULT_PIXEL_SIZE: u32 = 48;

/// Default charset: Latin-1.
pub const DEFAULT_FIRST_CODEPOINT: u32 = 0x20;
pub const DEFAULT_LAST_CODEPOINT: u32 = 0xFF;

/// How far the signed distance spreads from an edge, in pixels. Past this the
/// encoding saturates, so it also sets the padding the field needs to be read
/// correctly: about one screen pixel at the size text is drawn.
pub const DISTANCE_RANGE: f32 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeneratorOptions {
    /// Em size in pixels: the resolution the field is sampled at.
    pub pixel_size: u32,
    pub first_codepoint: u32,
    pub last_codepoint: u32,
    /// The packed atlas is exactly this wide; the height is whatever it takes.
    pub atlas_width: u32,
    pub padding: u32,
}

impl Default for GeneratorOptions {
    fn default() -> Self {
        Self {
            pixel_size: DEFAULT_PIXEL_SIZE,
            first_codepoint: DEFAULT_FIRST_CODEPOINT,
            last_codepoint: DEFAULT_LAST_CODEPOINT,
            atlas_width: 512,
            padding: 1,
        }
    }
}

#[derive(Debug)]
pub enum GenerateError {
    Font(ab_glyph::InvalidFont),
    /// `last_codepoint` before `first_codepoint`, or nothing in the range mapped
    /// to a glyph. An atlas with no glyphs has no fallback and cannot be laid out.
    NoGlyphs,
    /// One glyph is wider than the whole atlas: no packing can help.
    GlyphTooLarge {
        codepoint: u32,
        width: u32,
        atlas_width: u32,
    },
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Font(error) => write!(f, "{error}"),
            GenerateError::NoGlyphs => write!(f, "the font has no glyph in the requested range"),
            GenerateError::GlyphTooLarge {
                codepoint,
                width,
                atlas_width,
            } => write!(
                f,
                "glyph {codepoint:#x} is {width}px wide, atlas is {atlas_width}px"
            ),
        }
    }
}

impl std::error::Error for GenerateError {}

impl From<ab_glyph::InvalidFont> for GenerateError {
    fn from(error: ab_glyph::InvalidFont) -> Self {
        GenerateError::Font(error)
    }
}

pub fn generate(ttf: &[u8], options: &GeneratorOptions) -> Result<Atlas, GenerateError> {
    let font = FontRef::try_from_slice(ttf)?;
    let em = options.pixel_size as f32;
    let scaled = font.as_scaled(em);
    // Outline points are in the face's font units, not pixels; the flip to y
    // down happens with the scale so the field and the plane bounds agree.
    let units_to_pixels = em / font.units_per_em().unwrap_or(em);

    let mut rasters = Vec::new();
    if options.last_codepoint >= options.first_codepoint {
        for codepoint in options.first_codepoint..=options.last_codepoint {
            let Some(character) = char::from_u32(codepoint) else {
                continue;
            };
            let glyph_id = font.glyph_id(character);
            // Glyph 0 is `.notdef`: the character is not in this face, so the
            // atlas holds only the characters the font actually has.
            if glyph_id.0 == 0 {
                continue;
            }

            let advance = scaled.h_advance(glyph_id) / em;
            let mut raster = Raster::empty(codepoint, glyph_id, advance);

            // A whitespace glyph has an advance and no outline; it keeps the
            // advance and a zero quad, and the field skips it.
            if let Some(outline) = font.outline(glyph_id) {
                let mut contours = shape::contours(&outline.curves);
                edge_color::color_contours(&mut contours);
                let mut segments = Vec::new();
                shape::flatten(
                    &contours,
                    |p| Vec2::new(p.x * units_to_pixels, -p.y * units_to_pixels),
                    shape::FLATTEN_TOLERANCE,
                    &mut segments,
                );
                if !segments.is_empty() {
                    raster.fill_msdf(&segments, em, DISTANCE_RANGE);
                }
            }

            rasters.push(raster);
        }
    }
    if rasters.is_empty() {
        return Err(GenerateError::NoGlyphs);
    }

    let (width, height, placements) = pack(&rasters, options)?;

    let mut rgba8 = vec![0u8; width as usize * height as usize * 4];
    let mut glyphs = Vec::with_capacity(rasters.len());
    for (raster, placement) in rasters.iter().zip(placements.iter()) {
        let Some((left, top)) = *placement else {
            // No bitmap: whitespace, or a glyph with an empty outline.
            glyphs.push(Glyph {
                unicode: raster.codepoint,
                advance: raster.advance,
                plane_bounds_min: raster.plane_min,
                plane_bounds_max: raster.plane_max,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ZERO,
            });
            continue;
        };

        for row in 0..raster.height {
            for column in 0..raster.width {
                let source = (row * raster.width + column) as usize * 4;
                let texel = ((top + row) as usize * width as usize + (left + column) as usize) * 4;
                rgba8[texel..texel + 4].copy_from_slice(&raster.rgba[source..source + 4]);
            }
        }

        let u0 = left as f32 / width as f32;
        let v0 = top as f32 / height as f32;
        let u1 = (left + raster.width) as f32 / width as f32;
        let v1 = (top + raster.height) as f32 / height as f32;
        glyphs.push(Glyph {
            unicode: raster.codepoint,
            advance: raster.advance,
            plane_bounds_min: raster.plane_min,
            plane_bounds_max: raster.plane_max,
            uv_min: Vec2::new(u0, v0),
            uv_max: Vec2::new(u1, v1),
        });
    }

    let mut kerning: HashMap<(u32, u32), f32> = HashMap::new();
    for first in &rasters {
        for second in &rasters {
            // The kern table is sparse, so the zero pairs cost a lookup and no
            // entry: the map ends up the size of the pairs that move anything.
            let pair = scaled.kern(first.glyph_id, second.glyph_id);
            if pair != 0.0 {
                kerning.insert((first.codepoint, second.codepoint), pair / em);
            }
        }
    }

    let metrics = Metrics {
        em_size: 1.0,
        line_height: (scaled.ascent() - scaled.descent() + scaled.line_gap()) / em,
        ascender: scaled.ascent() / em,
        descender: scaled.descent() / em,
        distance_range: DISTANCE_RANGE,
    };

    Ok(Atlas::assemble(
        glyphs, kerning, metrics, width, height, rgba8,
    ))
}

/// Atlas width and height, and where each raster landed in it.
type PackedSheet = (u32, u32, Vec<Option<(u32, u32)>>);

/// Shelf packing: tallest first, left to right, a new row when the current one
/// is full. No rotation and no repacking; the difference is atlas area, not
/// correctness.
fn pack(rasters: &[Raster], options: &GeneratorOptions) -> Result<PackedSheet, GenerateError> {
    let width = options.atlas_width.max(1);
    let padding = options.padding;
    let mut placements = vec![None; rasters.len()];

    let mut order: Vec<usize> = (0..rasters.len())
        .filter(|&i| rasters[i].height > 0)
        .collect();
    order.sort_by(|&a, &b| {
        rasters[b]
            .height
            .cmp(&rasters[a].height)
            .then(rasters[a].codepoint.cmp(&rasters[b].codepoint))
    });

    let mut left = padding;
    let mut top = padding;
    let mut shelf_height = 0;
    for index in order {
        let (glyph_width, glyph_height) = (rasters[index].width, rasters[index].height);
        if glyph_width + 2 * padding > width {
            return Err(GenerateError::GlyphTooLarge {
                codepoint: rasters[index].codepoint,
                width: glyph_width,
                atlas_width: width,
            });
        }
        if left + glyph_width + padding > width {
            left = padding;
            top += shelf_height + padding;
            shelf_height = 0;
        }
        placements[index] = Some((left, top));
        left += glyph_width + padding;
        shelf_height = shelf_height.max(glyph_height);
    }

    let used_height = top + shelf_height + padding;
    Ok((width, used_height.max(1).next_power_of_two(), placements))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped_face() -> Vec<u8> {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for ancestor in manifest.ancestors() {
            let candidate = ancestor.join("assets/fonts/Inter.ttf");
            if candidate.is_file() {
                return std::fs::read(candidate).expect("the face is readable");
            }
        }
        panic!("no assets/fonts/Inter.ttf above {}", manifest.display());
    }

    fn only(codepoint: u32) -> GeneratorOptions {
        GeneratorOptions {
            first_codepoint: codepoint,
            last_codepoint: codepoint,
            ..GeneratorOptions::default()
        }
    }

    #[test]
    fn a_whitespace_glyph_has_a_zero_quad_and_no_field() {
        let atlas = generate(&shipped_face(), &only(0x20)).expect("space");
        assert_eq!(atlas.len(), 1);
        assert!(atlas.is_msdf());
        assert_eq!(atlas.metrics.distance_range, DISTANCE_RANGE);

        let space = atlas.glyph(0x20).expect("a space");
        assert!(space.advance > 0.0);
        assert_eq!(space.plane_bounds_min, Vec2::ZERO);
        assert_eq!(space.plane_bounds_max, Vec2::ZERO);
        assert_eq!(space.uv_min, Vec2::ZERO);
        assert_eq!(
            atlas.rgba8.len(),
            atlas.width as usize * atlas.height as usize * 4
        );
    }

    #[test]
    fn a_letter_gets_a_signed_distance_field() {
        let atlas = generate(&shipped_face(), &only(b'A' as u32)).expect("A");
        let a = atlas.glyph(b'A' as u32).expect("an A");
        assert!(a.plane_bounds_max.y > 0.0, "A sits above the baseline");
        assert!(a.uv_max.x > a.uv_min.x && a.uv_max.y > a.uv_min.y);

        let x0 = (a.uv_min.x * atlas.width as f32) as usize;
        let x1 = (a.uv_max.x * atlas.width as f32) as usize;
        let y0 = (a.uv_min.y * atlas.height as f32) as usize;
        let y1 = (a.uv_max.y * atlas.height as f32) as usize;

        let mut inside = 0;
        let mut outside = 0;
        let mut distinct = false;
        for y in y0..y1 {
            for x in x0..x1 {
                let texel = (y * atlas.width as usize + x) * 4;
                let (r, g, b) = (
                    atlas.rgba8[texel],
                    atlas.rgba8[texel + 1],
                    atlas.rgba8[texel + 2],
                );
                if r > 140 {
                    inside += 1;
                }
                if r < 115 {
                    outside += 1;
                }
                if r != g || g != b {
                    distinct = true;
                }
            }
        }
        assert!(
            inside > 0 && outside > 0,
            "inside {inside}, outside {outside}"
        );
        assert!(
            distinct,
            "every texel had equal channels: no edge colouring"
        );
    }
}
