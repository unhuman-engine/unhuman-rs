//! The viewport's input surface and the key handler.
//!
//! The viewport is the only element that takes mouse input directly; the keys
//! are handled at the root (`crate::view::render`), because gpui dispatches a key
//! event from the focused element up to the root and the root always receives
//! it, which is what makes the shortcut list reachable no matter what is focused.

use glam::Vec2;
use gpui::{
    canvas, div, prelude::*, Context, IntoElement, KeyDownEvent, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Point, ScrollDelta, ScrollWheelEvent,
};
use mirror::gizmo::{Camera as GizmoCamera, GizmoTransform, TransformGizmo};
use mirror::Ray;

use crate::app::EditorApp;
use crate::commands;
use crate::gizmo::{append, box_batch, gizmo_batch, selection_bounds};

/// One wheel notch is about this many pixels on a precise device.
const PIXELS_PER_LINE: f32 = 50.0;

/// The camera dollies per wheel *line*; a trackpad sends pixels, and feeding
/// 120 pixels in as 120 lines is a camera that flies through its focal point.
fn scroll_lines(delta: ScrollDelta) -> (f32, f32) {
    match delta {
        ScrollDelta::Lines(point) => (point.x, point.y),
        ScrollDelta::Pixels(point) => (
            f32::from(point.x) / PIXELS_PER_LINE,
            f32::from(point.y) / PIXELS_PER_LINE,
        ),
    }
}

impl EditorApp {
    /// The viewport's input surface: it measures the panel and takes its mouse.
    ///
    /// gpui has no call that reads a laid-out element's bounds, so the size comes
    /// from the paint that laid it out: `canvas`'s prepaint is handed the bounds
    /// it is about to paint into, which is the frame's own box. The same element
    /// registers the viewport's listeners, and they are *window* listeners
    /// because an orbit drag and a gizmo drag both leave the panel before they
    /// end -- an element's own `on_mouse_move` stops firing at its edge.
    pub(crate) fn viewport_surface(&self, cx: &Context<Self>) -> impl IntoElement {
        let measured = cx.entity();
        let moved = cx.entity();

        // The canvas measures the panel and takes the *move* events: a camera or
        // gizmo drag leaves the panel, and an element's own `on_mouse_move` stops
        // at its edge, so that one is a window listener. It is the only handler
        // here that can be registered twice without harm -- it records where the
        // cursor is, and the same position twice is the same position.
        //
        // Everything else is an element handler: a window listener is registered
        // once per paint, and this element paints twice in a frame, so every
        // click was being handled twice.
        let surface = canvas(
            move |bounds, _, cx| {
                let size: (f32, f32) = (bounds.size.width.into(), bounds.size.height.into());
                let origin: (f32, f32) = (bounds.origin.x.into(), bounds.origin.y.into());
                measured.update(cx, |this, _| {
                    this.viewport = size;
                    this.viewport_origin = origin;
                });
            },
            move |_bounds, _, window, _cx| {
                window.on_mouse_event(move |event: &MouseMoveEvent, _phase, _window, cx| {
                    moved.update(cx, |this, _| this.viewport_mouse_move(event));
                });
            },
        )
        .absolute()
        .size_full();

        div()
            .id("viewport")
            .absolute()
            .size_full()
            .child(surface)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event, window, cx| {
                    this.focus_handle.focus(window, cx);
                    this.viewport_mouse_down(event);
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event, _, _| this.viewport_mouse_up(event)),
            )
            .on_mouse_up(
                MouseButton::Middle,
                cx.listener(|this, event, _, _| this.viewport_mouse_up(event)),
            )
            .on_mouse_up(
                MouseButton::Right,
                cx.listener(|this, event, _, _| this.viewport_mouse_up(event)),
            )
            .on_scroll_wheel(cx.listener(|this, event, _, _| this.viewport_scroll(event)))
            .on_drop(cx.listener(|this, payload: &mirror::DragPayload, _, cx| {
                this.add_asset(&payload.path, cx);
            }))
            .track_focus(&self.focus_handle)
    }

    /// The cursor in the viewport's own coordinates, inside or outside it.
    fn viewport_local(&self, position: Point<gpui::Pixels>) -> (f32, f32) {
        let (x, y) = self.viewport_origin;
        (f32::from(position.x) - x, f32::from(position.y) - y)
    }

    fn inside_viewport(&self, local: (f32, f32)) -> bool {
        let (width, height) = self.viewport;
        local.0 >= 0.0 && local.1 >= 0.0 && local.0 <= width && local.1 <= height
    }

    /// The cursor in the viewport's own coordinates, or `None` when it is outside.
    fn viewport_cursor(&self, position: Point<gpui::Pixels>) -> Option<(f32, f32)> {
        let local = self.viewport_local(position);
        self.inside_viewport(local).then_some(local)
    }

    /// The gizmo's camera: the same matrices the frame is drawn with, sized to
    /// the panel, so what is picked is what is on screen.
    fn gizmo_camera(&self) -> Option<GizmoCamera> {
        let camera = self.engine.plugin::<bifrost::BifrostPlugin>()?.camera();
        let (width, height) = self.viewport;
        Some(GizmoCamera {
            view: camera.view_matrix(),
            projection: camera.projection(),
            viewport_size: Vec2::new(width, height),
        })
    }

    /// The selected entity's transform, which is what a gizmo draws around.
    fn gizmo_target(&self) -> Option<GizmoTransform> {
        let entity = self.editor.selection.selected?;
        Some(mirror::gizmo_target(&self.engine.engine().scene, entity))
    }

    /// Hands the gizmo and the selection outline to the frame when they changed,
    /// and nothing otherwise.
    ///
    /// The geometry is world-space, so it only changes when the selection, the
    /// transform or the mode does -- not when the camera moves.
    pub(crate) fn sync_gizmo_lines(&mut self) {
        let mode = self.editor.gizmo.mode;
        let target = self.gizmo_target();
        let selection = self.editor.selection.selected;
        if self.gizmo_lines == Some((mode, target, selection)) {
            return;
        }
        self.gizmo_lines = Some((mode, target, selection));

        // Fold the outline's bounds before touching the plugin: `selection_bounds`
        // borrows the engine, and the plugin takes it mutably.
        let outline = selection.and_then(|entity| selection_bounds(self, entity));

        let Some(plugin) = self.engine.plugin_mut::<bifrost::BifrostPlugin>() else {
            return;
        };
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        if let Some(target) = target {
            let (v, i) = gizmo_batch(mode, &target);
            vertices = v;
            indices = i;
        }
        if let Some((min, max)) = outline {
            let (v, i) = box_batch(min, max, [1.0, 0.62, 0.12, 1.0]);
            append(&mut vertices, &mut indices, v, i);
        }
        plugin.set_lines(vertices, indices);
    }

    /// Writes this frame's input into the renderer's camera.
    ///
    /// The scroll is consumed here: it is a per-frame delta, and a frame that
    /// carried it twice would dolly twice as far.
    pub(crate) fn apply_viewport_input(&mut self) {
        if let Some(plugin) = self.engine.plugin_mut::<bifrost::BifrostPlugin>() {
            *plugin.input_mut() = self.input;
        }
        self.input.scroll = (0.0, 0.0);
    }

    /// Handlers do not notify: the frame loop repaints every frame, and a
    /// handler that notified as well would render the view twice in one frame --
    /// which registers these listeners twice, and then every event is handled
    /// twice.
    fn viewport_mouse_move(&mut self, event: &MouseMoveEvent) {
        self.input.alt = event.modifiers.alt;
        self.input.left = event.pressed_button == Some(MouseButton::Left);
        self.input.middle = event.pressed_button == Some(MouseButton::Middle);
        self.input.right = event.pressed_button == Some(MouseButton::Right);

        // A drag leaves the panel before it ends: keep tracking the cursor so
        // the camera does not stop at the panel's edge.
        let local = self.viewport_local(event.position);
        let dragging = self.input.left || self.input.middle || self.input.right;
        if dragging || self.inside_viewport(local) {
            self.input.cursor = local;
        }
        if !self.inside_viewport(local) {
            return;
        }

        if self.editor.gizmo.is_dragging() {
            let mouse = Vec2::new(local.0, local.1);
            if let Some(transform) = self.editor.gizmo.update(mouse) {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.apply_gizmo(scene, &transform);
            }
        }
    }

    fn viewport_mouse_down(&mut self, event: &MouseDownEvent) {
        if event.button != MouseButton::Left || event.modifiers.alt {
            return;
        }
        let Some(cursor) = self.viewport_cursor(event.position) else {
            return;
        };
        let mouse = Vec2::new(cursor.0, cursor.1);

        // A handle under the cursor starts a drag; anywhere else picks, which is
        // the same click doing the obvious thing in both cases.
        if let (Some(camera), Some(target)) = (self.gizmo_camera(), self.gizmo_target()) {
            if let Some(axis) = TransformGizmo::pick_axis(&camera, &target, mouse) {
                self.editor.gizmo.begin(&camera, &target, mouse, axis);
                return;
            }
        }

        let (width, height) = self.viewport;
        let Some(camera) = self
            .engine
            .plugin::<bifrost::BifrostPlugin>()
            .map(|plugin| plugin.camera().view_projection())
        else {
            return;
        };
        let Some(ray) = Ray::through_cursor(cursor, (width as u32, height as u32), camera) else {
            return;
        };

        let engine = self.engine.engine_mut();
        let picked = self.editor.pick(&engine.scene, &engine.assets, ray);
        self.editor.select(picked);
    }

    fn viewport_mouse_up(&mut self, event: &MouseUpEvent) {
        match event.button {
            MouseButton::Left => {
                self.input.left = false;
                if self.editor.gizmo.is_dragging() {
                    self.editor.gizmo.end();
                }
            }
            MouseButton::Middle => self.input.middle = false,
            MouseButton::Right => self.input.right = false,
            MouseButton::Navigate(_) => (),
        }
    }

    fn viewport_scroll(&mut self, event: &ScrollWheelEvent) {
        if self.viewport_cursor(event.position).is_none() {
            return;
        }
        let (dx, dy) = scroll_lines(event.delta);
        self.input.scroll = (self.input.scroll.0 + dx, self.input.scroll.1 + dy);
    }

    /// The root key handler: maps a keystroke to a command and runs it. The
    /// registry ([`crate::commands`]) is why the keys here and the list in Help >
    /// Shortcuts cannot drift.
    ///
    /// While a hierarchy row is being renamed, the keys edit the buffer instead:
    /// there is no text input widget to route them through, and every key would
    /// otherwise also fire its shortcut.
    pub(crate) fn handle_key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        if self.editor.hierarchy.renaming.is_some() && self.rename_key(event) {
            cx.notify();
            return;
        }
        if let Some(command) = commands::command_for_keystroke(&event.keystroke) {
            self.run_command(command, cx);
        }
    }

    /// Applies one key to the rename buffer. `false` leaves the key to the
    /// registry: a modified key is a shortcut, not text.
    fn rename_key(&mut self, event: &KeyDownEvent) -> bool {
        match event.keystroke.key.as_str() {
            "enter" => {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.commit_rename(scene);
            }
            "escape" => self.editor.cancel_rename(),
            "backspace" => {
                self.editor.hierarchy.rename_buffer.pop();
            }
            _ => {
                let Some(text) = event.keystroke.key_char.as_deref() else {
                    return false;
                };
                let modifiers = event.keystroke.modifiers;
                if modifiers.control || modifiers.alt || modifiers.platform {
                    return false;
                }
                self.editor.hierarchy.rename_buffer.push_str(text);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bifrost::camera::{CameraInput, EditorCamera};
    use gpui::{point, px};

    #[test]
    fn a_precise_scroll_is_converted_to_wheel_lines() {
        assert_eq!(
            scroll_lines(ScrollDelta::Pixels(point(px(100.0), px(-50.0)))),
            (2.0, -1.0)
        );
        assert_eq!(
            scroll_lines(ScrollDelta::Lines(point(1.0, -2.0))),
            (1.0, -2.0)
        );
    }

    /// The flags the viewport writes are the ones the orbit camera reads: left
    /// rotates, middle pans, and scroll dollies.
    #[test]
    fn held_buttons_and_scroll_reach_the_camera() {
        let mut camera = EditorCamera::new(45.0, 16.0 / 9.0, 0.1, 1000.0);
        camera.set_pitch(0.5);
        camera.set_yaw(0.6);

        let yaw = camera.yaw();
        camera.update(&CameraInput {
            alt: true,
            left: true,
            cursor: (120.0, 90.0),
            ..Default::default()
        });
        assert_ne!(camera.yaw(), yaw, "alt+left should orbit");

        let focal = camera.focal_point();
        camera.update(&CameraInput {
            alt: true,
            middle: true,
            cursor: (180.0, 150.0),
            ..Default::default()
        });
        assert_ne!(camera.focal_point(), focal, "alt+middle should pan");

        let distance = camera.distance();
        camera.on_scroll(1.0);
        assert_ne!(camera.distance(), distance, "scroll should dolly");
    }
}
