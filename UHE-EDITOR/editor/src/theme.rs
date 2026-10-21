//! `mirror`'s palette, mapped into gpui colors.
//!
//! `mirror::theme` owns the palette and the theme list. This is only the
//! conversion, plus the derived tones the panels need: panel background, border,
//! row hover.

use gpui::Hsla;
use mirror::theme::{Palette, Rgba, Theme};

/// Every color one frame of the editor draws with.
///
/// Only what the panels draw: the palette is `mirror`'s, mapped once per frame.
#[derive(Clone, Copy, Debug)]
pub struct EditorTheme {
    pub bg: Hsla,
    pub panel: Hsla,
    pub border: Hsla,
    pub text: Hsla,
    pub text_dim: Hsla,
    pub accent: Hsla,
    pub accent_hover: Hsla,
    pub accent_muted: Hsla,
    pub accent_foreground: Hsla,
    pub row_hover: Hsla,
    pub toolbar: Hsla,
    pub status: Hsla,
    pub console_info: Hsla,
    pub console_warn: Hsla,
    pub console_error: Hsla,
}

impl EditorTheme {
    pub fn from_palette(palette: &Palette) -> Self {
        let panel_t = if palette.light { 0.08 } else { 0.045 };
        let border_t = if palette.light { 0.22 } else { 0.14 };
        let status_t = if palette.light { 0.05 } else { 0.025 };
        let panel = mix(palette.bg, palette.text, panel_t);

        Self {
            bg: hsla(palette.bg),
            panel: hsla(panel),
            border: hsla(mix(palette.bg, palette.text, border_t)),
            text: hsla(palette.text),
            text_dim: hsla(palette.text_dim),
            accent: hsla(palette.accent),
            accent_hover: hsla(palette.accent_hover),
            accent_muted: hsla(palette.accent_muted),
            accent_foreground: hsla(palette.accent_foreground),
            row_hover: hsla(mix(panel, palette.text, 0.05)),
            toolbar: hsla(palette.toolbar_bg),
            status: hsla(mix(palette.bg, palette.text, status_t)),
            console_info: hsla(palette.console_info),
            console_warn: hsla(palette.console_warn),
            console_error: hsla(palette.console_error),
        }
    }
}

impl From<&Theme> for EditorTheme {
    fn from(theme: &Theme) -> Self {
        Self::from_palette(&theme.palette())
    }
}

/// gpui wants its own color type. The values are `mirror`'s, clamped: a palette
/// is data, and a channel outside `0..=1` is a bug that surfaces as an entirely
/// black panel rather than where it was written.
fn hsla(color: Rgba) -> Hsla {
    Hsla::from(gpui::Rgba {
        r: color.r.clamp(0.0, 1.0),
        g: color.g.clamp(0.0, 1.0),
        b: color.b.clamp(0.0, 1.0),
        a: color.a.clamp(0.0, 1.0),
    })
}

/// `t == 0` is `a`, `t == 1` is `b`. Alpha comes from `a`.
fn mix(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    Rgba::new(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirror::theme::ThemeId;

    fn close(a: Rgba, b: Rgba) {
        for (x, y) in [(a.r, b.r), (a.g, b.g), (a.b, b.b), (a.a, b.a)] {
            assert!((x - y).abs() < 1e-3, "{a:?} vs {b:?}");
        }
    }

    /// The round-trip comparison is `mirror` versus gpui's colour type; both are
    /// four `f32` channels, so the test compares them directly.
    fn from_gpui(color: gpui::Rgba) -> Rgba {
        Rgba::new(color.r, color.g, color.b, color.a)
    }

    #[test]
    fn every_theme_survives_the_round_trip() {
        let mut theme = Theme::new();
        for id in ThemeId::all() {
            theme.set_selected(*id);
            let palette = theme.palette();
            let mapped = EditorTheme::from(&theme);

            close(palette.accent, from_gpui(mapped.accent.to_rgb()));
            close(palette.text, from_gpui(mapped.text.to_rgb()));
            close(palette.bg, from_gpui(mapped.bg.to_rgb()));
        }
    }

    #[test]
    fn a_custom_accent_reaches_the_panel_colors() {
        let mut theme = Theme::new();
        theme.set_custom_accent(Rgba::rgb(1.0, 0.0, 0.0));
        let palette = theme.palette();
        let mapped = EditorTheme::from(&theme);

        close(palette.accent, from_gpui(mapped.accent.to_rgb()));
        let accent = mapped.accent.to_rgb();
        assert!(accent.r > accent.g && accent.r > accent.b);
    }

    #[test]
    fn derived_tones_move_toward_the_text_color() {
        let dark = Palette {
            accent: Rgba::rgb(0.42, 0.39, 1.0),
            accent_hover: Rgba::rgb(0.52, 0.49, 1.0),
            accent_active: Rgba::rgb(0.62, 0.59, 1.0),
            accent_muted: Rgba::rgb(0.2, 0.2, 0.4),
            accent_foreground: Rgba::rgb(1.0, 1.0, 1.0),
            bg: Rgba::rgb(0.0, 0.0, 0.0),
            text: Rgba::rgb(0.92, 0.92, 0.92),
            text_dim: Rgba::rgb(0.42, 0.42, 0.42),
            title_bg: Rgba::rgb(0.02, 0.02, 0.02),
            title_tint: 0.0,
            toolbar_bg: Rgba::rgb(0.03, 0.03, 0.03),
            console_info: Rgba::rgb(0.5, 0.8, 1.0),
            console_warn: Rgba::rgb(1.0, 0.8, 0.3),
            console_error: Rgba::rgb(1.0, 0.4, 0.4),
            console_critical: Rgba::rgb(1.0, 0.2, 0.2),
            light: false,
        };
        let mapped = EditorTheme::from_palette(&dark);
        assert!(mapped.panel.to_rgb().r > mapped.bg.to_rgb().r);
        assert!(mapped.row_hover.to_rgb().r > mapped.panel.to_rgb().r);
        assert!(mapped.border.to_rgb().r > mapped.bg.to_rgb().r);

        let light = Palette {
            bg: Rgba::rgb(1.0, 1.0, 1.0),
            text: Rgba::rgb(0.05, 0.05, 0.05),
            light: true,
            ..dark
        };
        let mapped = EditorTheme::from_palette(&light);
        assert!(mapped.panel.to_rgb().r < mapped.bg.to_rgb().r);
        assert!(mapped.border.to_rgb().r < mapped.bg.to_rgb().r);
    }

    #[test]
    fn out_of_range_channels_are_clamped() {
        let color = hsla(Rgba::new(-1.0, 2.0, 0.5, 1.5));
        let back = color.to_rgb();
        assert!(back.r <= 1.0 && back.g <= 1.0 && back.b <= 1.0);
        assert_eq!(color.a, 1.0);
    }

    #[test]
    fn mix_hits_its_endpoints() {
        let a = Rgba::new(0.0, 0.0, 0.0, 0.5);
        let b = Rgba::new(1.0, 0.5, 0.25, 0.5);
        close(mix(a, b, 0.0), a);
        close(mix(a, b, 1.0), b);
        close(mix(a, b, 0.5), Rgba::new(0.5, 0.25, 0.125, 0.5));
        close(mix(a, b, 4.0), b);
    }
}
