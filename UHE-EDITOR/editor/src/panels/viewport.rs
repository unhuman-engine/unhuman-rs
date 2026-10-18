//! The center panel: the engine's frame, or the reason there is not one.
//!
//! `mirror`'s gizmo is not drawn here. It picks against the mouse position in
//! viewport space, and this panel shows the readback image without an overlay
//! yet, so the pick and the drag (`mirror::gizmo::TransformGizmo`,
//! `Camera { viewport_size }`) still wait. See `crate::viewport`.

use std::sync::Arc;

use gpui::{div, img, prelude::*, px, IntoElement, ObjectFit, RenderImage};

use crate::theme::EditorTheme;
use crate::viewport::MISSING_FRAME_REASON;

pub fn panel(
    theme: &EditorTheme,
    frame: Option<&Arc<RenderImage>>,
    measure: impl IntoElement,
) -> impl IntoElement {
    let body = match frame {
        Some(image) => div()
            .flex_1()
            .min_h_0()
            .child(
                img(image.clone())
                    .size_full()
                    .object_fit(ObjectFit::Contain),
            )
            .into_any_element(),
        None => div()
            .max_w(px(420.0))
            .text_sm()
            .text_center()
            .text_color(theme.text_dim)
            .child(MISSING_FRAME_REASON)
            .into_any_element(),
    };

    div()
        .size_full()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap_2()
        .bg(theme.bg)
        .child(div().text_xs().text_color(theme.text_dim).child("VIEWPORT"))
        // The frame's own box, and the only element whose size is the render
        // size. The measure element paints over it and reports that size, so the
        // engine renders at the resolution the image is shown at rather than a
        // fixed one scaled into it.
        .child(
            div()
                .relative()
                .flex_1()
                .min_h_0()
                .w_full()
                .flex()
                .flex_col()
                .child(body)
                .child(measure),
        )
}
