//! Reading a distance field back into coverage.
//!
//! This is the CPU twin of the median in `assets/shaders/text.wgsl`. The text
//! pass reconstructs in the shader; the generator and its tests use this. Both
//! sides have to agree, so both do the same steps: take the median of the three
//! channels, subtract the edge's 0.5, then widen one texture unit back to screen
//! pixels with the atlas's own distance range.

use glam::Vec2;

/// The middle of three values. `f32` has no `Ord`, and a mean would blur the
/// two distances the edge colouring worked to keep apart.
pub fn median3(r: f32, g: f32, b: f32) -> f32 {
    r.min(g).max(r.max(g).min(b))
}

/// How many distance units one screen pixel spans at this sampling rate. The
/// shader computes `screen_tex_size` from `fwidth(uv)`; a caller with no
/// derivatives passes the texel size directly.
pub fn screen_px_range(distance_range: f32, atlas_size: Vec2, screen_tex_size: Vec2) -> f32 {
    let unit_range = Vec2::new(distance_range / atlas_size.x, distance_range / atlas_size.y);
    (0.5 * unit_range.dot(screen_tex_size)).max(1.0)
}

/// Coverage in `0..=1` from the three reconstructed channels. Same expression
/// the shader runs, clamped the same way.
pub fn coverage(r: f32, g: f32, b: f32, screen_px_range: f32) -> f32 {
    ((median3(r, g, b) - 0.5) * screen_px_range + 0.5).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::distance::encode;

    #[test]
    fn the_median_is_the_middle_of_the_three() {
        assert_eq!(median3(0.1, 0.5, 0.9), 0.5);
        assert_eq!(median3(0.9, 0.1, 0.5), 0.5);
        assert_eq!(median3(0.5, 0.5, 0.5), 0.5);
        assert_eq!(median3(0.2, 0.2, 0.8), 0.2);
    }

    #[test]
    fn a_texel_per_pixel_reconstructs_the_distance() {
        // When one texture unit maps to one pixel, `screen_px_range` equals the
        // atlas range, so the decode is the inverse of `encode`.
        let range = 2.0;
        for distance in [-1.0_f32, 0.0, 1.0] {
            let texel = encode(distance, range) as f32 / 255.0;
            let reconstructed = coverage(texel, texel, texel, range);
            let expected = 0.5 + 0.5 * distance;
            assert!(
                (reconstructed - expected).abs() < 0.01,
                "distance {distance}: {reconstructed} != {expected}"
            );
        }
    }

    #[test]
    fn the_edge_channel_is_a_half() {
        // 0.5 in the field is the surface, so coverage there is a half.
        assert!((coverage(0.5, 0.5, 0.5, 4.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_full_size_screen_texel_gives_the_atlas_range() {
        let atlas = Vec2::new(512.0, 256.0);
        let range = screen_px_range(2.0, atlas, atlas);
        assert!((range - 2.0).abs() < 1e-4);
    }
}
