//! The root view and the frame loop.
//!
//! One gpui frame is one engine frame: `next_frame` advances the engine, asks
//! for the frame after it, and marks the view dirty. `render` lays the docks out
//! and paints the dropdown last, so it sits over the panels.

use gpui::{
    div, prelude::*, px, Context, ExternalPaths, IntoElement, KeyDownEvent, Render, Window,
};

use crate::app::EditorApp;
use crate::dock as docking;
use crate::layout::{Dock, Splitter};
use crate::menu;
use crate::panels;

impl Render for EditorApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = self.theme;

        let left = !self.layout.is_empty(Dock::Left);
        let right = !self.layout.is_empty(Dock::Right);
        let bottom = !self.layout.is_empty(Dock::Bottom);
        let left_width = self.layout.left_width;
        let right_width = self.layout.right_width;
        let bottom_height = self.layout.bottom_height;

        let mut main = div().flex().flex_1().min_h_0();
        if left {
            main = main
                .child(
                    div()
                        .flex_none()
                        .h_full()
                        .w(px(left_width))
                        .child(docking::region(Dock::Left, self, &theme, cx)),
                )
                .child(self.splitter(Splitter::Hierarchy, &theme, cx));
        }
        main = main.child(
            div()
                .flex_1()
                .min_w_0()
                .h_full()
                .child(panels::viewport::panel(
                    &theme,
                    self.frame.as_ref(),
                    self.viewport_surface(cx),
                )),
        );
        if right {
            main = main
                .child(self.splitter(Splitter::Inspector, &theme, cx))
                .child(
                    div()
                        .flex_none()
                        .h_full()
                        .w(px(right_width))
                        .child(docking::region(Dock::Right, self, &theme, cx)),
                );
        }

        let mut column = div()
            .size_full()
            .flex()
            .flex_col()
            .bg(theme.bg)
            .text_color(theme.text)
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _window, cx| {
                let paths = paths.paths().to_vec();
                this.drop_files(&paths, cx);
            }))
            .on_key_down(
                cx.listener(|this, event: &KeyDownEvent, _, cx| this.handle_key(event, cx)),
            )
            .child(menu::bar(self, &theme, cx))
            .child(self.toolbar(cx))
            .child(main);

        if bottom {
            column = column
                .child(self.splitter(Splitter::Content, &theme, cx))
                .child(
                    div()
                        .flex_none()
                        .w_full()
                        .h(px(bottom_height))
                        .child(docking::region(Dock::Bottom, self, &theme, cx)),
                );
        }

        // The dropdown is painted last so it sits over the panels.
        column
            .child(self.status_bar())
            .child(menu::overlay(self, &theme, cx))
    }
}

/// One gpui frame is one engine frame: advance the engine, mark the view dirty,
/// and ask for the frame after this one.
pub(crate) fn next_frame(this: &mut EditorApp, window: &mut Window, cx: &mut Context<EditorApp>) {
    // Before the engine advances: the size is the panel's, as the previous paint
    // measured it, so a window that changed size renders at the new one now
    // rather than a frame later.
    this.sync_viewport_extent(window);
    this.save_layout();

    if let Some(replaced) = this.advance_frame() {
        // gpui's sprite atlas holds the tile until the image is dropped, and a
        // readback frame is a new image every time. Dropping the previous one
        // here keeps the atlas at one frame instead of growing it by one each
        // frame.
        if let Err(error) = window.drop_image(replaced) {
            log::warn!("editor: could not drop the previous viewport image: {error}");
        }
    }

    cx.notify();
    cx.on_next_frame(window, next_frame);
}
