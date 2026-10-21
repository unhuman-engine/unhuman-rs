//! The menu bar. Every item is a [`crate::commands::Command`] or a panel move,
//! read from the same registry the shortcut handler and the Help > Shortcuts
//! panel use, so a shortcut and a menu item are one entry, not two.
//!
//! gpui has no menu widget at this rev, so this is a manual dropdown: the titles
//! are a row in the top bar and the open dropdown is an absolute element painted
//! after the content (`bar` first, `overlay` last).

use gpui::{div, prelude::*, px, AnyElement, ClickEvent, Context, IntoElement, SharedString};

use crate::app::EditorApp;
use crate::commands::Command;
use crate::dock;
use crate::layout::PanelId;
use crate::theme::EditorTheme;

pub const MENU_HEIGHT: f32 = 26.0;
const ITEM_HEIGHT: f32 = 22.0;
const MENU_WIDTH: f32 = 210.0;

/// The menu bar's titles.
pub const TITLES: [&str; 6] = ["File", "Edit", "Gizmo", "Play", "View", "Help"];

/// One row of an open menu.
#[derive(Clone)]
pub enum Item {
    /// A command, labelled from its spec.
    Command(Command),
    /// A command whose label is only known at paint time, such as Undo's
    /// `mirror::commands` entry.
    Labelled {
        command: Command,
        label: String,
    },
    /// Show or hide a panel.
    Panel(PanelId),
    /// A row of move-to-edge controls for a panel.
    Move(PanelId),
    Separator,
}

/// The open menu's items, built from the editor's state.
pub fn items(app: &EditorApp, index: usize) -> Vec<Item> {
    match TITLES.get(index).copied() {
        Some("File") => commands_of("File"),
        Some("Edit") => vec![
            labelled(Command::Undo, app.editor.history.undo_label(), "Undo"),
            labelled(Command::Redo, app.editor.history.redo_label(), "Redo"),
            Item::Separator,
            Item::Command(Command::Rename),
            Item::Command(Command::Delete),
        ],
        Some("Gizmo") => commands_of("Gizmo"),
        Some("Play") => commands_of("Play"),
        Some("View") => {
            let mut items: Vec<Item> = PanelId::ALL.into_iter().map(Item::Panel).collect();
            items.push(Item::Separator);
            for panel in PanelId::ALL {
                if app.layout.is_open(panel) {
                    items.push(Item::Move(panel));
                }
            }
            items
        }
        Some("Help") => vec![Item::Command(Command::ToggleShortcuts)],
        _ => Vec::new(),
    }
}

fn commands_of(category: &str) -> Vec<Item> {
    Command::in_category(category).map(Item::Command).collect()
}

/// A dynamic-label command, or the plain command when the history has no entry.
fn labelled(command: Command, label: Option<String>, fallback: &str) -> Item {
    match label {
        Some(label) => Item::Labelled {
            command,
            label: format!("{fallback}: {label}"),
        },
        None => Item::Command(command),
    }
}

/// The title row. Painted as the first child of the root.
pub fn bar(app: &EditorApp, theme: &EditorTheme, cx: &mut Context<EditorApp>) -> impl IntoElement {
    let mut row = div()
        .flex_none()
        .h(px(MENU_HEIGHT))
        .px_1()
        .flex()
        .items_center()
        .gap_1()
        .bg(theme.toolbar)
        .border_b_1()
        .border_color(theme.border);

    for (index, title) in TITLES.iter().enumerate() {
        let open = app.menu == Some(index);
        row = row.child(menu_button(index, title, open, theme, cx));
    }
    row
}

fn menu_button(
    index: usize,
    title: &'static str,
    open: bool,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    div()
        .id(SharedString::from(format!("menu-{title}")))
        .flex_none()
        .px_2()
        .py_1()
        .rounded_sm()
        .text_xs()
        .cursor_pointer()
        .text_color(if open {
            theme.accent_foreground
        } else {
            theme.text
        })
        .bg(if open {
            theme.accent_muted
        } else {
            theme.toolbar
        })
        .hover(move |style| style.bg(theme.row_hover))
        .on_click(cx.listener(move |this, _event: &ClickEvent, _, cx| {
            this.menu = if this.menu == Some(index) {
                None
            } else {
                Some(index)
            };
            cx.notify();
        }))
        .child(SharedString::from(title))
}

/// The open dropdown, painted after the content so it sits on top. Empty when no
/// menu is open.
pub fn overlay(app: &EditorApp, theme: &EditorTheme, cx: &mut Context<EditorApp>) -> AnyElement {
    let Some(index) = app.menu else {
        return div().into_any_element();
    };

    let rows: Vec<AnyElement> = items(app, index)
        .into_iter()
        .map(|item| row_element(item, theme, cx))
        .collect();

    div()
        .absolute()
        .top(px(MENU_HEIGHT))
        .left(px(x_offset(index)))
        .w(px(MENU_WIDTH))
        .flex()
        .flex_col()
        .py_1()
        .rounded_sm()
        .bg(theme.panel)
        .border_1()
        .border_color(theme.border)
        .shadow_md()
        .children(rows)
        .into_any_element()
}

fn row_element(item: Item, theme: &EditorTheme, cx: &mut Context<EditorApp>) -> AnyElement {
    match item {
        Item::Separator => div()
            .flex_none()
            .my_1()
            .h(px(1.0))
            .w_full()
            .bg(theme.border)
            .into_any_element(),
        Item::Move(panel) => div()
            .flex_none()
            .h(px(ITEM_HEIGHT))
            .px_2()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .text_sm()
            .text_color(theme.text)
            .child(SharedString::from(format!("Move {}", panel.title())))
            .child(dock::move_controls(panel, theme, cx))
            .into_any_element(),
        Item::Command(command) => clickable(
            command.label().to_string(),
            command.binding().unwrap_or(""),
            Item::Command(command),
            theme,
            cx,
        ),
        Item::Labelled { command, label } => clickable(
            label,
            command.binding().unwrap_or(""),
            Item::Command(command),
            theme,
            cx,
        ),
        Item::Panel(panel) => {
            clickable(panel.title().to_string(), "", Item::Panel(panel), theme, cx)
        }
    }
}

fn clickable(
    label: String,
    binding: &str,
    action: Item,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> AnyElement {
    let hint = binding.to_uppercase().replace('-', " ");
    let command = match action {
        Item::Command(command) => Some(command),
        _ => None,
    };
    let panel = match action {
        Item::Panel(panel) => Some(panel),
        _ => None,
    };

    let element = div()
        .id(SharedString::from(format!("menu-item-{label}")))
        .flex_none()
        .h(px(ITEM_HEIGHT))
        .px_2()
        .flex()
        .items_center()
        .justify_between()
        .gap_2()
        .text_sm()
        .cursor_pointer()
        .text_color(theme.text)
        .hover(move |style| style.bg(theme.row_hover));

    let element = match (command, panel) {
        (Some(command), _) => {
            element.on_click(cx.listener(move |this, _event: &ClickEvent, _, cx| {
                this.menu = None;
                this.run_command(command, cx);
            }))
        }
        (None, Some(panel)) => {
            element.on_click(cx.listener(move |this, _event: &ClickEvent, _, cx| {
                this.menu = None;
                this.toggle_panel(panel, cx);
            }))
        }
        (None, None) => element,
    };

    element
        .child(SharedString::from(label))
        .child(
            div()
                .flex_none()
                .text_xs()
                .text_color(theme.text_dim)
                .child(SharedString::from(hint)),
        )
        .into_any_element()
}

/// An estimate of a title's width, so its dropdown lines up under it. gpui has
/// no "read a laid-out element's bounds" call, so this is a character-count
/// approximation rather than a measurement.
fn x_offset(index: usize) -> f32 {
    const CHAR: f32 = 7.5;
    const PAD: f32 = 16.0;
    TITLES
        .iter()
        .take(index)
        .map(|title| title.chars().count() as f32 * CHAR + PAD)
        .sum::<f32>()
        + 4.0
}
