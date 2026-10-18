//! The editor binary: gpui over `mirror`'s model.
//!
//! The panels read `mirror`'s model; the engine loop they drive is `adam`'s,
//! advanced from gpui's frame callback (`app::next_frame`). The layout is fixed:
//! a resizable hierarchy on the left, viewport in the middle, inspector on the
//! right, content browser below, toolbar above and status bar under it. Docking
//! and floating windows are not implemented.

mod app;
mod panels;
mod theme;
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
