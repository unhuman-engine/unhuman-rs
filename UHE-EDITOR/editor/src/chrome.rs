//! The editor's own chrome: the toolbar, the status bar and the splitters.
//!
//! These read the model and the layout; they never change it themselves except
//! through [`EditorApp::run_command`](crate::actions), so the toolbar and the
//! keys stay one code path.

use gpui::{
    canvas, div, prelude::*, px, Context, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent,
};
use mirror::gizmo::GizmoMode;
use mirror::play_state::PlayStatus;

use crate::app::{entity_label, play_label, EditorApp};
use crate::commands::Command;
use crate::layout::{Splitter, SplitterAxis};
use crate::panels::{self, button};
use crate::theme::EditorTheme;
use crate::viewport;

const TOOLBAR_HEIGHT: f32 = 34.0;
const STATUS_HEIGHT: f32 = 22.0;
pub(crate) const SPLITTER: f32 = 5.0;

fn separator(theme: &EditorTheme) -> impl IntoElement {
    div().flex_none().w(px(1.0)).h(px(18.0)).bg(theme.border)
}

impl EditorApp {
    pub(crate) fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = self.theme;
        let status = self.editor.play.status;
        let mode = self.editor.gizmo.mode;

        div()
            .flex_none()
            .h(px(TOOLBAR_HEIGHT))
            .px_2()
            .flex()
            .items_center()
            .gap_1()
            .bg(theme.toolbar)
            .border_b_1()
            .border_color(theme.border)
            .child(button(
                "toolbar-play",
                "Play",
                status == PlayStatus::Playing,
                &theme,
                cx,
                |this, _, cx| this.run_command(Command::Play, cx),
            ))
            .child(button(
                "toolbar-pause",
                "Pause",
                status == PlayStatus::Paused,
                &theme,
                cx,
                |this, _, cx| this.run_command(Command::Pause, cx),
            ))
            .child(button(
                "toolbar-step",
                "Step",
                false,
                &theme,
                cx,
                |this, _, cx| this.run_command(Command::Step, cx),
            ))
            .child(button(
                "toolbar-stop",
                "Stop",
                status == PlayStatus::Stopped,
                &theme,
                cx,
                |this, _, cx| this.run_command(Command::Stop, cx),
            ))
            .child(separator(&theme))
            .child(button(
                "toolbar-move",
                GizmoMode::Translate.name(),
                mode == GizmoMode::Translate,
                &theme,
                cx,
                |this, _, cx| this.run_command(Command::Translate, cx),
            ))
            .child(button(
                "toolbar-rotate",
                GizmoMode::Rotate.name(),
                mode == GizmoMode::Rotate,
                &theme,
                cx,
                |this, _, cx| this.run_command(Command::Rotate, cx),
            ))
            .child(button(
                "toolbar-scale",
                GizmoMode::Scale.name(),
                mode == GizmoMode::Scale,
                &theme,
                cx,
                |this, _, cx| this.run_command(Command::Scale, cx),
            ))
            .child(div().flex_1())
            .child(panels::dim_label(
                viewport::backend_label(self.backend.as_deref()),
                &theme,
            ))
            .child(panels::dim_label(
                viewport::fps_label(self.editor.stats.fps),
                &theme,
            ))
    }

    pub(crate) fn status_bar(&self) -> impl IntoElement {
        let theme = self.theme;
        let entities = self.engine.engine().scene.entities().len();

        div()
            .flex_none()
            .h(px(STATUS_HEIGHT))
            .px_2()
            .flex()
            .items_center()
            .gap_3()
            .bg(theme.status)
            .border_t_1()
            .border_color(theme.border)
            .text_xs()
            .text_color(theme.text_dim)
            .child(play_label(self.editor.play.status))
            .child(self.editor.gizmo.mode.name())
            .child(entity_label(entities))
            .child(viewport::frame_status(
                self.backend.as_deref(),
                self.editor.stats.fps,
                self.editor.stats.frame_ms,
            ))
            .child(format!(
                "render {:.1} ms",
                self.editor.profiler.frame().timings.engine_ms
            ))
            .child(div().flex_1())
            .child("F1: shortcuts")
            .child(self.theme_source.name())
    }

    /// A drag handle. The bar starts the drag; the move and release listeners
    /// live in its `canvas` paint callback, the one place gpui lets a view
    /// register a window-level mouse listener.
    pub(crate) fn splitter(
        &self,
        splitter: Splitter,
        theme: &EditorTheme,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let dragging = self.layout.is_dragging(splitter);
        let entity = cx.entity();
        let width_axis = splitter.axis() == SplitterAxis::Width;

        let element = div()
            .id(splitter.id())
            .flex_none()
            .bg(if dragging { theme.accent } else { theme.border })
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                    let position = if width_axis {
                        event.position.x.as_f32()
                    } else {
                        event.position.y.as_f32()
                    };
                    this.layout.begin_drag(splitter, position);
                    cx.notify();
                }),
            )
            .child(
                canvas(
                    |_, _, _| (),
                    move |_bounds, _, window, _cx| {
                        window.on_mouse_event({
                            let entity = entity.clone();
                            move |event: &MouseMoveEvent, _phase, _window, cx| {
                                if !event.dragging() {
                                    return;
                                }
                                entity.update(cx, |this, cx| {
                                    let position = if width_axis {
                                        event.position.x.as_f32()
                                    } else {
                                        event.position.y.as_f32()
                                    };
                                    if this.layout.drag_to(position) {
                                        cx.notify();
                                    }
                                });
                            }
                        });
                        window.on_mouse_event(move |_: &MouseUpEvent, _phase, _window, cx| {
                            entity.update(cx, |this, cx| {
                                if this.layout.end_drag() {
                                    this.save_layout();
                                    cx.notify();
                                }
                            });
                        });
                    },
                )
                .size_full(),
            );

        match splitter.axis() {
            SplitterAxis::Width => element.w(px(SPLITTER)).h_full().cursor_ew_resize(),
            SplitterAxis::Height => element.h(px(SPLITTER)).w_full().cursor_ns_resize(),
        }
    }
}
