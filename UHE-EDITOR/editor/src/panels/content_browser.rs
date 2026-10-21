//! The bottom panel: the asset tree.
//!
//! `mirror::content_browser` owns the root, the current directory, the filter and
//! the selection; this draws the breadcrumbs, the entries and the navigation.
//!
//! The thumbnails are names, not pictures: gpui's image elements load through
//! gpui's own asset cache, not `vault`. The filter box is a label for the same
//! reason: gpui has no text input widget in this crate, only the text renderer.

use gpui::{
    div, prelude::*, px, ClickEvent, Context, IntoElement, Pixels, Point, Render, SharedString,
    Window,
};
use mirror::content_browser::{DragPayload, Entry};
use mirror::Editor;

use crate::app::EditorApp;
use crate::panels::{body, dim_label, with_alpha};
use crate::theme::EditorTheme;

const CELL_WIDTH: f32 = 132.0;
const CELL_HEIGHT: f32 = 28.0;

pub fn panel(
    editor: &Editor,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let browser = &editor.content_browser;
    let entries = browser.entries();
    let breadcrumbs = browser.breadcrumbs();
    let selected = browser.selected.clone();

    let mut crumbs: Vec<gpui::AnyElement> = Vec::with_capacity(breadcrumbs.len() + 1);
    crumbs.push(crumb(0, "Assets", None, theme, cx).into_any_element());
    for (index, (name, path)) in breadcrumbs.iter().enumerate() {
        crumbs.push(crumb(index + 1, name, Some(path), theme, cx).into_any_element());
    }

    let mut cells: Vec<gpui::AnyElement> = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        // The drag carries the registry's virtual path, not the filesystem path:
        // the drop side resolves it against the mounted root.
        let payload = browser.payload(entry);
        cells.push(cell(index, entry, payload, selected.as_deref(), theme, cx).into_any_element());
    }

    body()
        .bg(theme.panel)
        .child(
            div()
                .flex_none()
                .h(px(24.0))
                .px_1()
                .flex()
                .items_center()
                .gap_1()
                .border_b_1()
                .border_color(theme.border)
                .text_xs()
                .text_color(theme.text_dim)
                .child("CONTENT BROWSER")
                .children(crumbs),
        )
        .child(
            div()
                .id("content-browser-entries")
                .flex()
                .flex_wrap()
                .content_start()
                .gap_1()
                .p_1()
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .when(entries.is_empty(), |grid| {
                    grid.child(dim_label("Empty directory", theme))
                })
                .children(cells),
        )
}

fn crumb(
    index: usize,
    name: &str,
    path: Option<&std::path::PathBuf>,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let owned = path.cloned();
    let label = SharedString::from(name.to_string());
    let element = div()
        .id(("crumb", index))
        .flex_none()
        .px_1()
        .rounded_sm()
        .cursor_pointer()
        .hover(move |style| style.bg(theme.row_hover))
        .child(label);

    let Some(path) = owned else {
        return element;
    };

    element.on_click(cx.listener(move |this, _event: &ClickEvent, _window, cx| {
        this.editor.content_browser.navigate(&path);
        cx.notify();
    }))
}

fn cell(
    index: usize,
    entry: &Entry,
    payload: DragPayload,
    selected: Option<&str>,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let is_selected = selected == Some(entry.path.as_str());
    let path = entry.path.clone();
    let is_dir = entry.is_dir;
    let hover = theme.row_hover;

    let (bg, text) = if is_selected {
        (theme.accent_muted, theme.accent_foreground)
    } else {
        (with_alpha(theme.panel, 0.0), theme.text)
    };

    let label = if is_dir {
        format!("▸ {}", entry.name)
    } else if entry.extension.is_empty() {
        entry.name.clone()
    } else {
        format!("{} .{}", entry.name, entry.extension)
    };

    div()
        .id(("content-browser-cell", index))
        .flex_none()
        .w(px(CELL_WIDTH))
        .h(px(CELL_HEIGHT))
        .px_1()
        .flex()
        .items_center()
        .rounded_sm()
        .text_sm()
        .cursor_pointer()
        .bg(bg)
        .text_color(text)
        .hover(move |style| style.bg(hover))
        // A file drags into the viewport, where the drop spawns it. A directory
        // has nothing to spawn, so it only navigates on click.
        .when(!is_dir, |cell| {
            cell.on_drag(payload, |payload, position, _window, cx| {
                let label: SharedString = payload.path.clone().into();
                cx.new(|_| DragPreview { label, position })
            })
        })
        .on_click(cx.listener(move |this, _event: &ClickEvent, _window, cx| {
            let typed_path = std::path::PathBuf::from(&path);
            if is_dir {
                this.editor.content_browser.navigate(&typed_path);
            } else {
                this.editor.content_browser.selected = Some(path.clone());
            }
            cx.notify();
        }))
        .child(div().min_w_0().truncate().child(SharedString::from(label)))
}

/// The tile that follows the cursor while a file is dragged out of the browser.
struct DragPreview {
    label: SharedString,
    position: Point<Pixels>,
}

impl Render for DragPreview {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().pl(self.position.x).pt(self.position.y).child(
            div()
                .px_2()
                .py_1()
                .rounded_sm()
                .bg(gpui::rgba(0x1e1e1ecc))
                .border_1()
                .border_color(gpui::rgba(0xffffff22))
                .text_xs()
                .text_color(gpui::white())
                .child(self.label.clone()),
        )
    }
}
