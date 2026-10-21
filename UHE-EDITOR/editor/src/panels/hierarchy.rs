//! The left panel: the entity tree.
//!
//! The tree, the selection and the expansion are `mirror::hierarchy`'s. A row
//! drags onto another row to reparent, and F2 (or a double click) starts the
//! inline rename. Both go through `Editor`, which records them for undo.

use gpui::{div, prelude::*, px, AnyElement, ClickEvent, Context, Empty, IntoElement};
use meats::{Entity, Scene};
use mirror::hierarchy::{Hierarchy, Row};
use mirror::Editor;

use crate::app::EditorApp;
use crate::panels::{body, button, dim_label, with_alpha, ROW_HEIGHT};
use crate::theme::EditorTheme;

pub fn panel(
    editor: &Editor,
    scene: &Scene,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let selected = editor.selection.selected;
    let renaming = editor.hierarchy.renaming;
    let rename_buffer = editor.hierarchy.rename_buffer.as_str();
    let rows = editor.rows(scene);

    let mut list: Vec<AnyElement> = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        list.push(
            row_element(index, row, selected, renaming, rename_buffer, theme, cx)
                .into_any_element(),
        );
    }

    body()
        .bg(theme.panel)
        .child(
            div()
                .flex_none()
                .h(px(26.0))
                .px_1()
                .flex()
                .items_center()
                .justify_between()
                .border_b_1()
                .border_color(theme.border)
                .text_xs()
                .text_color(theme.text_dim)
                .child("ENTITIES")
                .child(button(
                    "hierarchy-create",
                    "+",
                    false,
                    theme,
                    cx,
                    |this, _window, cx| {
                        let scene = &mut this.engine.engine_mut().scene;
                        this.editor.create_entity(scene, None);
                        cx.notify();
                    },
                )),
        )
        .child(
            div()
                .id("hierarchy-list")
                .flex()
                .flex_col()
                .flex_1()
                .min_h_0()
                .py_1()
                .overflow_y_scroll()
                .when(rows.is_empty(), |list| {
                    list.child(dim_label("No entities. Use + to add one.", theme))
                })
                .children(list),
        )
}

fn row_element(
    index: usize,
    row: &Row,
    selected: Option<Entity>,
    renaming: Option<Entity>,
    rename_buffer: &str,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    let entity = row.entity;
    let is_selected = selected == Some(entity);

    let (row_bg, row_text) = if is_selected {
        (theme.accent_muted, theme.accent_foreground)
    } else {
        (with_alpha(theme.panel, 0.0), theme.text)
    };
    let hover_bg = theme.row_hover;
    let depth = 4.0 + row.depth as f32 * 12.0;
    let label = if renaming == Some(entity) {
        format!("{rename_buffer}|")
    } else {
        row.name.clone()
    };

    div()
        .id(("hierarchy-row", index))
        .flex_none()
        .h(px(ROW_HEIGHT))
        .pl(px(depth))
        .pr_1()
        .flex()
        .items_center()
        .gap_1()
        .rounded_sm()
        .text_sm()
        .cursor_pointer()
        .bg(row_bg)
        .text_color(row_text)
        .hover(move |style| style.bg(hover_bg))
        .on_drag(entity, |_, _, _, cx| cx.new(|_| Empty))
        .on_drop(cx.listener(move |this, dragged: &Entity, _window, cx| {
            let target = {
                let scene = &this.engine.engine().scene;
                drop_target(scene, &this.editor.hierarchy, *dragged, Some(entity))
            };
            let Some(target) = target else {
                return;
            };
            let scene = &mut this.engine.engine_mut().scene;
            if this.editor.reparent(scene, *dragged, Some(target)) {
                this.editor.select(Some(*dragged));
                cx.notify();
            }
        }))
        .on_click(cx.listener(move |this, event: &ClickEvent, _window, cx| {
            this.editor.select(Some(entity));
            if event.click_count() >= 2 {
                this.editor
                    .begin_rename(&this.engine.engine().scene, entity);
            }
            cx.notify();
        }))
        .child(disclosure(index, row, theme, cx))
        .child(div().flex_1().min_w_0().truncate().child(label))
}

fn disclosure(
    index: usize,
    row: &Row,
    theme: &EditorTheme,
    cx: &mut Context<EditorApp>,
) -> impl IntoElement {
    if !row.has_children {
        return div()
            .id(("hierarchy-disclosure", index))
            .w(px(10.0))
            .flex_none();
    }

    let entity = row.entity;
    let expanded = row.expanded;

    div()
        .id(("hierarchy-disclosure", index))
        .w(px(10.0))
        .flex_none()
        .flex()
        .justify_center()
        .text_color(theme.text_dim)
        .child(if expanded { "▾" } else { "▸" })
        .on_click(cx.listener(move |this, _event: &ClickEvent, _window, cx| {
            this.editor.toggle_expanded(entity);
            cx.notify();
        }))
}

/// The entity a hierarchy drop parents `dragged` under, or `None` when the
/// pointer is over no row, the row is `dragged` itself, or the row is below it.
pub fn drop_target(
    scene: &Scene,
    hierarchy: &Hierarchy,
    dragged: Entity,
    pointer_row: Option<Entity>,
) -> Option<Entity> {
    let target = pointer_row?;
    (target != dragged && hierarchy.can_drop(scene, dragged, target)).then_some(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Root -> Child -> Grandchild.
    fn tree() -> (Scene, Entity, Entity, Entity) {
        let mut scene = Scene::new();
        let root = scene.spawn("Root");
        let child = scene.spawn("Child");
        let grandchild = scene.spawn("Grandchild");
        scene.set_parent(child, Some(root));
        scene.set_parent(grandchild, Some(child));
        (scene, root, child, grandchild)
    }

    fn row_entities(scene: &Scene, editor: &Editor) -> Vec<Entity> {
        editor.rows(scene).iter().map(|row| row.entity).collect()
    }

    #[test]
    fn a_pointer_row_resolves_to_the_parent_it_is_dropped_on() {
        let (scene, root, child, grandchild) = tree();
        let mut editor = Editor::new();
        editor.set_expanded(root, true);
        editor.set_expanded(child, true);

        let rows = row_entities(&scene, &editor);
        let root_row = rows.iter().position(|entity| *entity == root);
        assert_eq!(
            drop_target(
                &scene,
                &editor.hierarchy,
                grandchild,
                root_row.map(|index| rows[index])
            ),
            Some(root)
        );
    }

    #[test]
    fn a_drop_on_self_or_into_the_subtree_is_refused() {
        let (scene, root, child, grandchild) = tree();
        let hierarchy = Hierarchy::new();

        assert_eq!(drop_target(&scene, &hierarchy, root, Some(root)), None);
        assert_eq!(drop_target(&scene, &hierarchy, root, Some(child)), None);
        assert_eq!(
            drop_target(&scene, &hierarchy, root, Some(grandchild)),
            None
        );
        assert_eq!(
            drop_target(&scene, &hierarchy, child, Some(grandchild)),
            None
        );
        assert_eq!(
            drop_target(&scene, &hierarchy, grandchild, Some(root)),
            Some(root)
        );
        assert_eq!(drop_target(&scene, &hierarchy, grandchild, None), None);
    }
}
