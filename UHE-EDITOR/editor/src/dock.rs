//! A dock: its tabs, the active panel under them, and the close button on each
//! tab. One function so every edge renders the same way.

use gpui::{div, prelude::*, px, ClickEvent, Context, IntoElement, SharedString};

use crate::app::EditorApp;
use crate::layout::{Dock, PanelId};
use crate::panels;
use crate::theme::EditorTheme;

const TAB_HEIGHT: f32 = 26.0;

pub fn region(
    dock: Dock,
    app: &EditorApp,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let panels = app.layout.panels(dock).to_vec();
    let active = app.layout.active(dock);

    let mut tabs = Vec::with_capacity(panels.len());
    for panel in panels {
        tabs.push(tab(panel, active == Some(panel), theme, cx));
    }

    let body = match active {
        Some(panel) => panels::body_for(panel, app, theme, cx),
        None => div().into_any_element(),
    };

    div()
        .flex()
        .flex_col()
        .size_full()
        .min_h_0()
        .child(
            div()
                .flex_none()
                .h(px(TAB_HEIGHT))
                .flex()
                .items_stretch()
                .bg(theme.toolbar)
                .border_b_1()
                .border_color(theme.border)
                .children(tabs)
                .child(div().flex_1())
                .when_some(active, |bar, panel| {
                    bar.child(
                        div()
                            .flex_none()
                            .flex()
                            .items_center()
                            .pr_1()
                            .child(move_controls(panel, theme, cx)),
                    )
                }),
        )
        .child(div().flex_1().min_h_0().overflow_hidden().child(body))
}

fn tab(
    panel: PanelId,
    active: bool,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let title = panel.title();
    div()
        .id(SharedString::from(title))
        .flex_none()
        .h_full()
        .px_2()
        .flex()
        .items_center()
        .gap_2()
        .text_xs()
        .cursor_pointer()
        .bg(if active { theme.panel } else { theme.toolbar })
        .text_color(if active { theme.text } else { theme.text_dim })
        .hover(move |style| style.bg(theme.row_hover))
        .on_click(cx.listener(move |this, _event: &ClickEvent, _, cx| {
            this.layout.activate(panel);
            this.save_layout();
            cx.notify();
        }))
        .child(SharedString::from(title))
        .child(
            div()
                .id(SharedString::from(format!("dock-close-{title}")))
                .flex_none()
                .px_1()
                .rounded_sm()
                .text_color(theme.text_dim)
                .hover(|style| style.text_color(theme.text))
                .child("x")
                .on_click(cx.listener(move |this, _event: &ClickEvent, _, cx| {
                    // The tab's own click would activate the panel on its way up.
                    cx.stop_propagation();
                    this.close_panel(panel, cx);
                })),
        )
}

/// A "move to" control for a panel, used by the View menu and the panel header:
/// three one-letter targets for the three edges.
pub fn move_controls(
    panel: PanelId,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let mut row = div().flex().items_center().gap_1();
    for dock in Dock::ALL {
        row = row.child(
            div()
                .id(SharedString::from(format!(
                    "move-{}-{}",
                    panel.title(),
                    dock.title()
                )))
                .flex_none()
                .px_1()
                .rounded_sm()
                .text_xs()
                .cursor_pointer()
                .text_color(theme.text_dim)
                .hover(move |style| style.bg(theme.row_hover))
                .child(match dock {
                    Dock::Left => "<",
                    Dock::Right => ">",
                    Dock::Bottom => "v",
                })
                .on_click(cx.listener(move |this, _event: &ClickEvent, _, cx| {
                    this.move_panel(panel, dock, cx);
                })),
        );
    }
    row
}
