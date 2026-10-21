//! The Help > Shortcuts panel: the command registry, printed. The keys here are
//! the keys the handler actually matches, because both read [`crate::commands`].

use gpui::{div, prelude::*, px, AnyElement, IntoElement, SharedString};

use crate::commands::{Command, CATEGORIES};
use crate::panels::ROW_HEIGHT;
use crate::theme::EditorTheme;

pub fn panel(theme: &EditorTheme) -> impl IntoElement {
    let mut rows: Vec<AnyElement> = Vec::new();
    for category in CATEGORIES {
        let commands: Vec<Command> = Command::in_category(category).collect();
        if commands.is_empty() {
            continue;
        }
        rows.push(
            div()
                .flex_none()
                .px_2()
                .pt_2()
                .pb_1()
                .text_xs()
                .text_color(theme.accent)
                .child(SharedString::from((*category).to_string()))
                .into_any_element(),
        );
        for command in commands {
            let keys = command
                .binding()
                .unwrap_or("")
                .to_uppercase()
                .replace('-', " ");
            rows.push(
                div()
                    .flex_none()
                    .h(px(ROW_HEIGHT))
                    .px_2()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .text_sm()
                    .text_color(theme.text)
                    .child(SharedString::from(command.label()))
                    .child(
                        div()
                            .flex_none()
                            .text_xs()
                            .text_color(theme.text_dim)
                            .child(SharedString::from(keys)),
                    )
                    .into_any_element(),
            );
        }
    }

    div()
        .id("shortcuts-list")
        .flex()
        .flex_col()
        .size_full()
        .overflow_y_scroll()
        .bg(theme.panel)
        .children(rows)
}
