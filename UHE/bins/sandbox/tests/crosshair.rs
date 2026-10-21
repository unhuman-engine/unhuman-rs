//! Crosshair geometry, which is pixels and needs no device.

#[allow(dead_code)]
#[path = "../src/crosshair.rs"]
mod crosshair;

use aimlab::Report;
use crosshair::{CrosshairRects, PixelRect};
use glam::Vec2;

fn rect(min: (f32, f32), max: (f32, f32)) -> PixelRect {
    PixelRect {
        min: Vec2::new(min.0, min.1),
        max: Vec2::new(max.0, max.1),
    }
}

#[test]
fn bars_and_dot_surround_the_centre() {
    let rects = CrosshairRects::around(Vec2::new(100.0, 50.0), Report::CROSSHAIR);
    let [left, right, top, bottom, dot] = rects.fill;

    assert_eq!(left, rect((88.0, 49.0), (96.0, 51.0)));
    assert_eq!(right, rect((104.0, 49.0), (112.0, 51.0)));
    assert_eq!(top, rect((99.0, 38.0), (101.0, 46.0)));
    assert_eq!(bottom, rect((99.0, 54.0), (101.0, 62.0)));
    assert_eq!(dot, rect((98.0, 48.0), (102.0, 52.0)));
}

#[test]
fn every_outline_surrounds_its_fill() {
    let rects = CrosshairRects::around(Vec2::new(100.0, 50.0), Report::CROSSHAIR);
    for (outline, fill) in rects.outline.into_iter().zip(rects.fill) {
        assert!(outline.min.x < fill.min.x && outline.min.y < fill.min.y);
        assert!(outline.max.x > fill.max.x && outline.max.y > fill.max.y);
    }
    assert_eq!(rects.outline[4], rect((97.0, 47.0), (103.0, 53.0)));
}
