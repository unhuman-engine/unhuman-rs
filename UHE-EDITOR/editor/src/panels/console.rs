//! The console panel: `mirror::Console`'s buffer, painted as one row per record.
//!
//! The buffer is shared with `adam::logging`, so engine records arrive here on
//! whichever thread emitted them. The panel paints a snapshot; the log file is
//! the pinned first line because a launch from a desktop entry has no terminal.
//!
//! The filter controls and the command line live here because this is the only
//! view of the console. The filter is written into `Console`, so the logger is
//! the one that drops a record, not a second check made while painting.

use gpui::{
    div, prelude::*, px, Context, Div, FocusHandle, IntoElement, KeyDownEvent, SharedString,
};

use mirror::{Console, Level, LogEntry};

use super::button;
use crate::app::EditorApp;
use crate::panels::ROW_HEIGHT;
use crate::theme::EditorTheme;

const LEVELS: [Level; 5] = [
    Level::Trace,
    Level::Debug,
    Level::Info,
    Level::Warn,
    Level::Error,
];

/// The two text fields need a handle that outlives one paint, or clicking to
/// focus would be undone by the next render dropping the handle.
struct ConsoleFocus {
    target: FocusHandle,
    command: FocusHandle,
}

impl gpui::Global for ConsoleFocus {}

fn console_focus(cx: &mut Context<EditorApp>) -> ConsoleFocus {
    if let Some(focus) = cx.try_global::<ConsoleFocus>() {
        return ConsoleFocus {
            target: focus.target.clone(),
            command: focus.command.clone(),
        };
    }
    let focus = ConsoleFocus {
        target: cx.focus_handle(),
        command: cx.focus_handle(),
    };
    cx.set_global(ConsoleFocus {
        target: focus.target.clone(),
        command: focus.command.clone(),
    });
    focus
}

pub fn panel(
    app: &EditorApp,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let console = &app.editor.console;
    let focus = console_focus(cx);
    let path = adam::logging::log_path();
    let path_line = match &path {
        Some(path) => format!("log file: {}", path.display()),
        None => "log file: not configured".to_string(),
    };

    let mut rows: Vec<gpui::AnyElement> = Vec::with_capacity(console.len() + 1);
    rows.push(
        div()
            .flex_none()
            .h(px(ROW_HEIGHT))
            .px_2()
            .flex()
            .items_center()
            .text_xs()
            .text_color(theme.text_dim)
            .child(SharedString::from(path_line.clone()))
            .into_any_element(),
    );

    for entry in console.lines() {
        // `mirror::Editor::new` seeds the same path; the header above owns it.
        if entry.target == "editor" && entry.message == path_line {
            continue;
        }
        rows.push(entry_row(entry, theme));
    }

    div()
        .flex()
        .flex_col()
        .size_full()
        .bg(theme.panel)
        .child(filter_bar(console, theme, &focus, cx))
        .child(
            div()
                .id("console-list")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .children(rows),
        )
        .child(command_bar(console, theme, &focus.command, cx))
}

fn level_label(level: Level) -> &'static str {
    match level {
        Level::Trace => "TRACE",
        Level::Debug => "DEBUG",
        Level::Info => "INFO",
        Level::Warn => "WARN",
        Level::Error => "ERROR",
    }
}

fn level_id(level: Level) -> &'static str {
    match level {
        Level::Trace => "console-level-trace",
        Level::Debug => "console-level-debug",
        Level::Info => "console-level-info",
        Level::Warn => "console-level-warn",
        Level::Error => "console-level-error",
    }
}

fn level_color(level: Level, theme: &EditorTheme) -> gpui::Hsla {
    match level {
        Level::Trace | Level::Debug => theme.text_dim,
        Level::Info => theme.console_info,
        Level::Warn => theme.console_warn,
        Level::Error => theme.console_error,
    }
}

fn filter_bar(
    console: &Console,
    theme: &EditorTheme,
    focus: &ConsoleFocus,
    cx: &mut Context<EditorApp>,
) -> Div {
    let active = console.level();
    let mut bar = div()
        .flex_none()
        .h(px(ROW_HEIGHT))
        .px_1()
        .flex()
        .items_center()
        .gap_1()
        .border_b_1()
        .border_color(theme.border);

    for level in LEVELS {
        bar = bar.child(button(
            level_id(level),
            level_label(level),
            level == active,
            theme,
            cx,
            move |this, _, cx| {
                this.editor.console.set_level(level);
                cx.notify();
            },
        ));
    }

    let target = console.target();
    let placeholder = target.is_empty();
    let handle = focus.target.clone();
    let click = handle.clone();
    bar.child(
        div()
            .id("console-target")
            .flex_1()
            .min_w_0()
            .h(px(20.0))
            .px_2()
            .flex()
            .items_center()
            .rounded_sm()
            .border_1()
            .border_color(theme.border)
            .text_xs()
            .truncate()
            .cursor_pointer()
            .text_color(if placeholder {
                theme.text_dim
            } else {
                theme.text
            })
            .track_focus(&handle)
            .on_click(cx.listener(move |_, _, window, cx| {
                click.focus(window, cx);
            }))
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
                let console = &this.editor.console;
                if event.keystroke.key == "escape" {
                    console.set_target("");
                } else if let Some(text) = edited_line(&console.target(), event) {
                    console.set_target(text);
                }
                cx.stop_propagation();
                cx.notify();
            }))
            .child(if placeholder {
                SharedString::from("all targets")
            } else {
                SharedString::from(target)
            }),
    )
}

fn command_bar(
    console: &Console,
    theme: &EditorTheme,
    focus: &FocusHandle,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let line = console.command_line();
    let placeholder = line.is_empty();
    let handle = focus.clone();
    let click = handle.clone();

    div()
        .id("console-command")
        .flex_none()
        .h(px(ROW_HEIGHT + 4.0))
        .px_2()
        .flex()
        .items_center()
        .gap_1()
        .border_t_1()
        .border_color(theme.border)
        .text_xs()
        .cursor_pointer()
        .track_focus(&handle)
        .on_click(cx.listener(move |_, _, window, cx| {
            click.focus(window, cx);
        }))
        .on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
            if event.keystroke.key == "enter" {
                let line = this.editor.console.command_line();
                this.run_console_command(&line, cx);
            } else {
                let console = &this.editor.console;
                match event.keystroke.key.as_str() {
                    "up" => console.history_up(),
                    "down" => console.history_down(),
                    "backspace" => console.command_backspace(),
                    "escape" => console.command_clear(),
                    _ => {
                        if let Some(text) = typed_text(event) {
                            console.command_insert(&text);
                        }
                    }
                }
            }
            cx.stop_propagation();
            cx.notify();
        }))
        .child(
            div()
                .flex_none()
                .text_color(theme.console_info)
                .child(SharedString::from(">")),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .truncate()
                .text_color(if placeholder {
                    theme.text_dim
                } else {
                    theme.text
                })
                .child(if placeholder {
                    SharedString::from("type a command: undo, redo, save, play...")
                } else {
                    SharedString::from(line)
                }),
        )
}

/// A key that inserts text, or a backspace that removes the last character.
fn edited_line(current: &str, event: &KeyDownEvent) -> Option<String> {
    if event.keystroke.key == "backspace" {
        let mut line = current.to_string();
        line.pop();
        return Some(line);
    }
    typed_text(event).map(|text| format!("{current}{text}"))
}

fn typed_text(event: &KeyDownEvent) -> Option<String> {
    let text = event.keystroke.key_char.as_deref()?;
    let modifiers = event.keystroke.modifiers;
    if modifiers.control || modifiers.alt || modifiers.platform {
        return None;
    }
    Some(text.to_string())
}

fn entry_row(entry: LogEntry, theme: &EditorTheme) -> gpui::AnyElement {
    let mut row = div()
        .flex_none()
        .min_h(px(ROW_HEIGHT))
        .px_2()
        .flex()
        .items_center()
        .gap_2()
        .text_xs()
        .child(
            div()
                .flex_none()
                .w(px(44.0))
                .text_color(level_color(entry.level, theme))
                .child(SharedString::from(level_label(entry.level))),
        )
        .child(
            div()
                .flex_none()
                .w(px(104.0))
                .truncate()
                .text_color(theme.text_dim)
                .child(SharedString::from(entry.target)),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .truncate()
                .text_color(theme.text)
                .child(SharedString::from(entry.message)),
        );

    if entry.count > 1 {
        row = row.child(
            div()
                .flex_none()
                .text_color(theme.console_warn)
                .child(SharedString::from(format!("x{}", entry.count))),
        );
    }

    row.into_any_element()
}
