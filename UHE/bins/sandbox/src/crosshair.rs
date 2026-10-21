//! The aim lab's crosshair, as HUD quads.
//!
//! `Report::CROSSHAIR` is in screen pixels and the overlay draws unit quads, so
//! this turns the description into pixel rects. Kept apart from `hud` so the
//! geometry is testable without a device.

use aimlab::Crosshair;
use bifrost::batch2d::quad_transform;
use bifrost::passes::OverlayFrame;
use glam::{Mat4, Vec2, Vec4};

/// How far an outline extends past the shape it borders, in pixels.
const OUTLINE: f32 = 1.0;

/// An axis-aligned rect in HUD pixels, origin top-left and y down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelRect {
    pub min: Vec2,
    pub max: Vec2,
}

impl PixelRect {
    fn center(self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    fn size(self) -> Vec2 {
        self.max - self.min
    }

    fn inflate(self, amount: f32) -> Self {
        Self {
            min: self.min - Vec2::splat(amount),
            max: self.max + Vec2::splat(amount),
        }
    }

    /// The sprite pass places a unit quad by its centre, not its top-left.
    fn transform(self) -> Mat4 {
        quad_transform(self.center().extend(0.0), self.size())
    }
}

/// The four bars and the dot, twice: outlines, then the fills drawn over them.
pub struct CrosshairRects {
    pub outline: [PixelRect; 5],
    pub fill: [PixelRect; 5],
}

impl CrosshairRects {
    /// Lays the crosshair out about `center`, with sizes left in screen pixels.
    pub fn around(center: Vec2, crosshair: Crosshair) -> Self {
        let half_thickness = crosshair.thickness * 0.5;
        let arm = crosshair.half_length;
        let gap = crosshair.gap;
        let dot = crosshair.dot_radius;
        let (x, y) = (center.x, center.y);

        let fill = [
            PixelRect {
                min: Vec2::new(x - arm, y - half_thickness),
                max: Vec2::new(x - gap, y + half_thickness),
            },
            PixelRect {
                min: Vec2::new(x + gap, y - half_thickness),
                max: Vec2::new(x + arm, y + half_thickness),
            },
            PixelRect {
                min: Vec2::new(x - half_thickness, y - arm),
                max: Vec2::new(x + half_thickness, y - gap),
            },
            PixelRect {
                min: Vec2::new(x - half_thickness, y + gap),
                max: Vec2::new(x + half_thickness, y + arm),
            },
            PixelRect {
                min: Vec2::new(x - dot, y - dot),
                max: Vec2::new(x + dot, y + dot),
            },
        ];
        Self {
            outline: fill.map(|rect| rect.inflate(OUTLINE)),
            fill,
        }
    }
}

/// Queues the crosshair at the middle of `viewport`.
pub fn push(hud: &OverlayFrame, viewport: Vec2, crosshair: Crosshair) {
    let rects = CrosshairRects::around(viewport * 0.5, crosshair);
    for rect in rects.outline {
        hud.push_quad(rect.transform(), color(crosshair.outline));
    }
    for rect in rects.fill {
        hud.push_quad(rect.transform(), color(crosshair.color));
    }
}

fn color([red, green, blue, alpha]: [u8; 4]) -> Vec4 {
    Vec4::new(
        red as f32 / 255.0,
        green as f32 / 255.0,
        blue as f32 / 255.0,
        alpha as f32 / 255.0,
    )
}
