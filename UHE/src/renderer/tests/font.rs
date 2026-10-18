//! The atlas format, the layout arithmetic and the runtime rasterizer.
//!
//! Pure CPU. Nothing here touches a device, and the one fixture that is not
//! inline is the engine's own shipped face, resolved by walking up from this
//! crate rather than a fixed `../../`; a path that silently resolves to nothing
//! is how a suite reports green having tested nothing.

use std::path::{Path, PathBuf};

use bifrost::font::atlas::{Atlas, AtlasError};
use bifrost::font::generate::{generate, GeneratorOptions};
use bifrost::font::layout::{layout, Align, LayoutOptions};
use glam::Vec2;
use vault::Image;

const SPACE: u32 = 0x20;
const A: u32 = b'A' as u32;
const V: u32 = b'V' as u32;
const W: u32 = b'W' as u32;
const Z: u32 = b'z' as u32;

fn asset(relative: &str) -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        let candidate = ancestor.join(relative);
        if candidate.is_file() {
            return candidate;
        }
    }
    panic!("could not find {relative} above {}", manifest.display());
}

/// The hand-written atlas: the smallest file with the `msdf-atlas-gen` schema.
fn fixture() -> Atlas {
    let path = asset("assets/fonts/test-atlas.json");
    let json = std::fs::read_to_string(&path).expect("the atlas fixture is readable");
    Atlas::parse(&json).expect("the atlas fixture parses")
}

fn near(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() < 1e-5
}

#[test]
fn parses_the_atlas_json() {
    let atlas = fixture();

    assert_eq!(atlas.width, 64);
    assert_eq!(atlas.height, 64);
    assert_eq!(atlas.len(), 4);
    assert_eq!(atlas.metrics.line_height, 2.0);
    assert_eq!(atlas.metrics.ascender, 1.5);
    assert_eq!(atlas.metrics.descender, -0.5);
    assert_eq!(atlas.metrics.distance_range, 2.0);
    assert!(atlas.is_msdf());
    assert!(near(atlas.scale(), 0.5));

    let a = atlas.glyph(A).expect("the fixture has an A");
    assert_eq!(a.advance, 0.6);
    assert_eq!(a.plane_bounds_min, Vec2::new(0.0, 0.0));
    assert_eq!(a.plane_bounds_max, Vec2::new(0.6, 1.0));
    // `atlasBounds` is y down from the top of the image, and the UV axis follows
    // it: min is the top edge, max the bottom.
    assert_eq!(a.uv_min, Vec2::new(0.0, 0.0));
    assert_eq!(a.uv_max, Vec2::new(12.0 / 64.0, 16.0 / 64.0));

    assert_eq!(atlas.kerning(A, V), -0.2);
    assert_eq!(atlas.kerning(V, A), 0.0);
    assert_eq!(atlas.kerning(A, W), 0.0);
}

#[test]
fn an_atlas_without_kerning_or_a_distance_range_still_parses() {
    let atlas = Atlas::parse(
        r#"{
            "atlas": { "width": 4, "height": 4 },
            "metrics": { "lineHeight": 1 },
            "glyphs": [ { "unicode": 65, "advance": 1 } ]
        }"#,
    )
    .expect("a minimal atlas");

    assert_eq!(atlas.len(), 1);
    assert_eq!(atlas.kerning(A, A), 0.0);
    // The distance range a minimal atlas defaults to.
    assert_eq!(atlas.metrics.distance_range, 2.0);
    assert!(atlas.is_msdf());
    assert!(near(atlas.scale(), 1.0));
}

#[test]
fn an_atlas_that_is_not_json_is_an_error() {
    let error = Atlas::parse("not an atlas").expect_err("a parse failure");
    assert!(matches!(error, AtlasError::Json(_)));
    assert!(Atlas::parse(r#"{"atlas":{"width":1,"height":1},"metrics":{},"glyphs":[]}"#).is_err());
}

#[test]
fn attaches_the_decoded_image() {
    let mut atlas = fixture();
    let image = Image {
        width: 64,
        height: 64,
        rgba8: vec![7u8; 64 * 64 * 4],
        srgb: false,
    };
    atlas.set_image(&image).expect("a matching image");
    assert_eq!(atlas.rgba8.len(), 64 * 64 * 4);
    assert_eq!(atlas.rgba8[0], 7);

    let mismatched = Image {
        width: 32,
        height: 32,
        rgba8: vec![0u8; 32 * 32 * 4],
        srgb: false,
    };
    assert!(atlas.set_image(&mismatched).is_err());

    Atlas::from_json_and_image(
        &std::fs::read_to_string(asset("assets/fonts/test-atlas.json")).expect("readable"),
        &image,
    )
    .expect("json and image together");
}

#[test]
fn layout_advances_by_the_glyph_and_the_kerning_pair() {
    let atlas = fixture();
    let out = layout("AV", &atlas, &LayoutOptions::default());

    assert_eq!(out.placements.len(), 2);
    assert_eq!(out.placements[0].pen_x, 0.0);
    // advance(A) * scale, then the (A, V) pair, then V's own 0.5 * scale.
    assert!(near(
        out.placements[1].pen_x,
        atlas.glyph(A).unwrap().advance * atlas.scale() + atlas.kerning(A, V) * atlas.scale()
    ));
    assert!(near(out.placements[1].pen_x, 0.2));
    assert!(near(out.width, 0.45));
    assert!(near(out.height, 1.0));
}

#[test]
fn layout_adds_the_tracking_to_every_glyph() {
    let atlas = fixture();
    let options = LayoutOptions {
        kerning: 0.1,
        ..LayoutOptions::default()
    };
    let out = layout("AV", &atlas, &options);

    // A's advance plus one tracking, less the (A, V) pair.
    assert!(near(out.placements[1].pen_x, 0.3));
    assert!(near(out.width, 0.65));
}

#[test]
fn a_missing_glyph_uses_the_fallback() {
    let atlas = fixture();
    assert!(atlas.glyph(Z).is_none());
    assert_eq!(atlas.fallback().unicode, A);
    assert_eq!(atlas.glyph_or_fallback(Z).unicode, A);

    let out = layout("Az", &atlas, &LayoutOptions::default());
    assert_eq!(out.placements.len(), 2);
    // The codepoint is kept, since it is what the caller asked for, but the quad
    // is the fallback glyph's, so the character is not dropped from the run.
    assert_eq!(out.placements[1].codepoint, Z);
    assert_eq!(out.placements[1].quad.uv, out.placements[0].quad.uv);
    assert!(near(out.placements[1].pen_x, 0.3));
}

#[test]
fn wrapping_breaks_at_the_last_fitting_space() {
    let atlas = fixture();
    let options = LayoutOptions {
        wrap_width: Some(1.7),
        ..LayoutOptions::default()
    };
    let out = layout("AW AW AWW", &atlas, &options);

    // "AW AW" is 1.65 and fits; the later space would leave "AW AWW" at 2.05.
    // So the break is the second space, not the first, and not the hard break.
    assert_eq!(out.placements.len(), 8);
    assert_eq!(out.placements[2].codepoint, SPACE);
    assert_eq!(
        out.placements
            .iter()
            .filter(|placement| placement.codepoint == SPACE)
            .count(),
        1
    );
    // The space the break happened at is dropped, not rendered at the line start.
    assert_eq!(out.placements[5].codepoint, A);
    assert_eq!(out.placements[5].pen_x, 0.0);
    assert_eq!(out.placements[5].pen_y, -1.0);
    assert!(near(out.width, 1.7));
    assert!(near(out.height, 2.0));
}

#[test]
fn wrapping_hard_breaks_a_word_wider_than_the_line() {
    let atlas = fixture();
    let options = LayoutOptions {
        wrap_width: Some(0.5),
        ..LayoutOptions::default()
    };
    let out = layout("AWWW", &atlas, &options);

    assert_eq!(out.placements.len(), 4);
    for (index, placement) in out.placements.iter().enumerate() {
        assert_eq!(placement.pen_y, -(index as f32));
        assert_eq!(placement.pen_x, 0.0);
    }
}

#[test]
fn alignment_moves_a_line_within_the_box() {
    let atlas = fixture();
    let options = LayoutOptions {
        wrap_width: Some(1.7),
        align: Align::Right,
        ..LayoutOptions::default()
    };
    let out = layout("AW AW AWW", &atlas, &options);

    // Line one is 1.65 wide in a 1.7 box, line two 1.1.
    assert!(near(out.placements[0].pen_x, 0.05));
    assert!(near(out.placements[5].pen_x, 0.6));
}

#[test]
fn newlines_start_a_line_one_line_height_down() {
    let atlas = fixture();
    let options = LayoutOptions {
        line_spacing: 0.5,
        ..LayoutOptions::default()
    };
    let out = layout("A\nV", &atlas, &options);

    assert_eq!(out.placements.len(), 2);
    assert_eq!(out.placements[1].pen_x, 0.0);
    assert!(near(out.placements[1].pen_y, -1.5));
    assert!(near(out.height, 3.0));
}

#[test]
fn generates_an_msdf_atlas_from_the_shipped_face() {
    let bytes = std::fs::read(asset("assets/fonts/Inter.ttf")).expect("the shipped face");
    let atlas = generate(&bytes, &GeneratorOptions::default()).expect("Inter rasterizes");

    // 190 of the 224 codepoints in 0x20..0xFF are in the face's cmap.
    assert!(atlas.len() >= 150, "only {} glyphs", atlas.len());
    assert_eq!(atlas.width, 512);
    assert_eq!(
        atlas.rgba8.len(),
        atlas.width as usize * atlas.height as usize * 4
    );
    assert!(atlas.metrics.line_height > 0.0);
    // A distance field now, with the range the shader widens by.
    assert!(atlas.is_msdf());
    assert!(atlas.metrics.distance_range > 0.0);

    let space = atlas.glyph(SPACE).expect("a space");
    assert!(space.advance > 0.0);
    assert_eq!(space.plane_bounds_min, Vec2::ZERO);
    assert_eq!(space.plane_bounds_max, Vec2::ZERO);

    let a = atlas.glyph(A).expect("an A");
    assert!(a.plane_bounds_max.y > 0.0, "A sits above the baseline");
    assert!(a.plane_bounds_min.y <= 0.0);
    assert!(a.uv_max.x > a.uv_min.x && a.uv_max.y > a.uv_min.y);
    assert!(a.uv_max.x <= 1.0 && a.uv_max.y <= 1.0);

    let out = layout("Hello A", &atlas, &LayoutOptions::default());
    assert_eq!(out.placements.len(), 7);
    assert!(out.width > 1.0);
    assert!(near(out.height, 1.0));
}
