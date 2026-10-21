//! The editor binary: gpui over `mirror`'s model.
//!
//! The panels read `mirror`'s model; the engine loop they drive is `adam`'s,
//! advanced from gpui's frame callback (`view::next_frame`). The dock layout,
//! the command registry and the frame source live in their own modules:
//! `layout` persists which panel sits on which edge, `commands` is the one list
//! the keys, the menu and the shortcut panel read, and `frame_source` is the
//! seam the renderer's readback is consumed through.

mod actions;
mod app;
mod chrome;
mod commands;
mod dock;
mod frame_source;
mod gizmo;
mod input;
mod layout;
mod menu;
mod panels;
mod profiler;
mod shortcuts;
mod theme;
mod view;
mod viewport;

use gpui::{px, size, App, AppContext, Bounds, TitlebarOptions, WindowBounds, WindowOptions};

fn main() {
    adam::logging::init();

    gpui_platform::application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1440.0), px(900.0)), cx);

        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("unhuman".into()),
                    appears_transparent: false,
                    traffic_light_position: None,
                }),
                ..Default::default()
            },
            |window, cx| {
                let view = cx.new(app::EditorApp::new);
                view.update(cx, |this, cx| this.start_frame_loop(window, cx));
                view
            },
        )
        .expect("open the editor window");

        cx.activate(true);
    });
}
