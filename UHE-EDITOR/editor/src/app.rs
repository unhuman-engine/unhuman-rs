//! The editor application: the gpui root view, the layout, and the single place
//! the engine loop is advanced.
//!
//! The engine frame and the gpui frame are the same frame. gpui owns the main
//! thread and calls back into this view for every drawn frame; the engine is
//! advanced from that callback (`next_frame`), so no second thread, channel, or
//! lock sits between the model the panels read and the scene the engine writes.
//! That holds because this struct is the only owner of both: `adam::App` and
//! `mirror::Editor` are fields of one struct, and `Engine`'s data is never
//! borrowed by two of them at once.
//!
//! The engine is built headless, because the window is gpui's. Each frame the
//! renderer reads its composed image back and this view turns it into a gpui
//! image for the viewport; see `crate::viewport`.

use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec2;
use gpui::{
    canvas, div, prelude::*, px, Context, ExternalPaths, IntoElement, KeyDownEvent, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, Point, Render, RenderImage, ScrollDelta,
    ScrollWheelEvent, Window,
};
use mirror::gizmo::{Camera as GizmoCamera, GizmoMode, GizmoTransform, TransformGizmo};
use mirror::play_state::PlayStatus;
use mirror::{Axis, Editor, Ray};

use crate::panels::{self, button};
use crate::theme::EditorTheme;
use crate::viewport;

const ENGINE_NAME: &CStr = c"unhuman-editor";

/// The engine's offscreen size, since the engine has no window of its own. The
/// viewport scales it to fit; a window-sized render would resize the engine with
/// the panel instead.
const ENGINE_EXTENT: chassis::Extent2d = chassis::Extent2d {
    width: 1280,
    height: 720,
};

const TOOLBAR_HEIGHT: f32 = 34.0;
const STATUS_HEIGHT: f32 = 22.0;
const SPLITTER: f32 = 5.0;

/// One side of the panel in device pixels, within what the device can allocate.
fn scaled(logical: f32, scale: f32, limit: u32) -> u32 {
    (logical * scale).round().clamp(1.0, limit as f32) as u32
}

/// The gizmo as a line batch: two vertices per segment, one index pair each.
///
/// `mirror` owns the geometry and hands back `(from, to, axis)`; the colours are
/// the editor's, because they are a convention of this UI rather than of the
/// maths.
fn gizmo_batch(
    mode: GizmoMode,
    target: &GizmoTransform,
) -> (Vec<bifrost::passes::LineVertex>, Vec<u32>) {
    let segments = mirror::gizmo::segments(mode, target);
    let mut vertices = Vec::with_capacity(segments.len() * 2);
    let mut indices = Vec::with_capacity(segments.len() * 2);

    for (segment, (from, to, axis)) in segments.into_iter().enumerate() {
        let color = axis_color(axis);
        vertices.push(bifrost::passes::LineVertex {
            position: from.to_array(),
            color,
        });
        vertices.push(bifrost::passes::LineVertex {
            position: to.to_array(),
            color,
        });
        let first = (segment * 2) as u32;
        indices.push(first);
        indices.push(first + 1);
    }

    (vertices, indices)
}

/// The gizmo's axis colours: x red, y green, z blue, as every editor has them.
fn axis_color(axis: Axis) -> [f32; 4] {
    match axis {
        Axis::X => [0.90, 0.20, 0.25, 1.0],
        Axis::Y => [0.35, 0.85, 0.30, 1.0],
        Axis::Z => [0.30, 0.45, 0.95, 1.0],
    }
}

const MIN_HIERARCHY: f32 = 150.0;
const MAX_HIERARCHY: f32 = 520.0;
const MIN_INSPECTOR: f32 = 190.0;
const MAX_INSPECTOR: f32 = 620.0;
const MIN_CONTENT: f32 = 70.0;
const MAX_CONTENT: f32 = 460.0;

pub struct EditorApp {
    pub(crate) engine: adam::App,
    pub(crate) editor: Editor,
    theme: EditorTheme,
    theme_source: mirror::theme::Theme,
    backend: Option<String>,
    layout: Layout,
    /// The renderer's last readback frame, as a gpui image the viewport draws.
    frame: Option<Arc<RenderImage>>,
    /// The viewport panel's size in logical pixels, as the last paint measured it.
    ///
    /// gpui has no "read an element's bounds" call, so this comes from a canvas
    /// painted over the panel. The engine renders at this size in device pixels,
    /// which is what makes the viewport the panel's resolution instead of a fixed
    /// one scaled to fit.
    viewport: (f32, f32),
    /// The viewport panel's top-left corner in window coordinates, in logical
    /// pixels. The cursor arrives in window coordinates, so this is what turns it
    /// into a position in the viewport.
    viewport_origin: (f32, f32),
    /// What the viewport's mouse and keyboard have accumulated for this frame.
    /// `bifrost`'s orbit camera reads it once per frame.
    input: bifrost::camera::CameraInput,
    /// The gizmo the frame's lines were last built for, so a frame that moved
    /// neither the selection nor the camera does not re-upload them.
    gizmo_lines: Option<(GizmoMode, Option<GizmoTransform>)>,
    /// What the viewport's key events arrive through: an element only sees keys
    /// while it is focused, and clicking the viewport is what focuses it.
    focus_handle: gpui::FocusHandle,
    /// Whether a refused viewport resize has been reported, so it is reported once.
    resize_warned: bool,
}

impl EditorApp {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let mut engine = adam::Engine::new();
        engine.assets.mount("assets");

        let arguments: Vec<String> = std::env::args().collect();
        let environment = std::env::var("UHE_BACKEND").ok();
        let backend = match engine::create_gpu_from_process(
            &arguments,
            environment.as_deref(),
            ENGINE_NAME,
            ENGINE_EXTENT,
        ) {
            Ok((gpu, _selection)) => {
                let name = gpu.info().name.clone();
                log::info!("editor renderer: {name}");
                engine.gpu = Some(gpu);
                Some(name)
            }
            Err(error) => {
                log::warn!("no GPU backend ({error}); the engine runs with gpu: None");
                None
            }
        };

        let mut app = adam::App::with_engine(engine).add(bifrost::BifrostPlugin::new());
        if let Err(error) = app.start() {
            log::error!("plugin build failed: {error}");
        }

        let editor = Editor::new();
        let theme_source = editor.theme;

        Self {
            engine: app,
            editor,
            theme: EditorTheme::from(&theme_source),
            theme_source,
            backend,
            layout: Layout::default(),
            frame: None,
            viewport: (ENGINE_EXTENT.width as f32, ENGINE_EXTENT.height as f32),
            viewport_origin: (0.0, 0.0),
            input: bifrost::camera::CameraInput::default(),
            gizmo_lines: None,
            focus_handle: cx.focus_handle(),
            resize_warned: false,
        }
    }

    /// Schedules the first engine frame. Every frame re-arms this one, so the
    /// engine advances at the rate the platform draws.
    pub fn start_frame_loop(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        cx.on_next_frame(window, next_frame);
    }

    /// The viewport's input surface: it measures the panel and takes its mouse.
    ///
    /// gpui has no call that reads a laid-out element's bounds, so the size comes
    /// from the paint that laid it out: `canvas`'s prepaint is handed the bounds
    /// it is about to paint into, which is the frame's own box. The same element
    /// registers the viewport's listeners, and they are *window* listeners
    /// because an orbit drag and a gizmo drag both leave the panel before they
    /// end -- an element's own `on_mouse_move` stops firing at its edge.
    fn viewport_surface(&self, cx: &Context<Self>) -> impl IntoElement {
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
            .on_scroll_wheel(cx.listener(|this, event, _, _| this.viewport_scroll(event)))
            .on_key_down(cx.listener(|this, event, _, _| this.viewport_key(event)))
            .track_focus(&self.focus_handle)
    }

    /// The cursor in the viewport's own coordinates, or `None` when it is outside.
    ///
    /// The panel's pixels, not the render's: every consumer here -- the orbit
    /// camera, the gizmo's projection, the pick ray -- works in the panel's
    /// space, and the render's extent only differs by the display scale.
    fn viewport_cursor(&self, position: Point<gpui::Pixels>) -> Option<(f32, f32)> {
        let (x, y) = self.viewport_origin;
        let (width, height) = self.viewport;
        let local = (f32::from(position.x) - x, f32::from(position.y) - y);
        let inside = local.0 >= 0.0 && local.1 >= 0.0 && local.0 <= width && local.1 <= height;
        inside.then_some(local)
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

    /// Hands the gizmo to the frame when it changed, and nothing otherwise.
    ///
    /// The geometry is world-space, so it only changes when the selection, the
    /// transform or the mode does -- not when the camera moves.
    fn sync_gizmo_lines(&mut self) {
        let mode = self.editor.gizmo.mode;
        let target = self.gizmo_target();
        if self.gizmo_lines == Some((mode, target)) {
            return;
        }
        self.gizmo_lines = Some((mode, target));

        let Some(plugin) = self.engine.plugin_mut::<bifrost::BifrostPlugin>() else {
            return;
        };
        match target {
            Some(target) => {
                let (vertices, indices) = gizmo_batch(mode, &target);
                plugin.set_lines(vertices, indices);
            }
            None => plugin.set_lines(Vec::new(), Vec::new()),
        }
    }

    /// Writes this frame's input into the renderer's camera.
    ///
    /// The scroll is consumed here: it is a per-frame delta, and a frame that
    /// carried it twice would dolly twice as far.
    fn apply_viewport_input(&mut self) {
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

        let Some(cursor) = self.viewport_cursor(event.position) else {
            return;
        };
        self.input.cursor = cursor;

        if self.editor.gizmo.is_dragging() {
            let mouse = Vec2::new(cursor.0, cursor.1);
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
        if event.button != MouseButton::Left {
            return;
        }
        self.input.left = false;
        if self.editor.gizmo.is_dragging() {
            self.editor.gizmo.end();
        }
    }

    fn viewport_scroll(&mut self, event: &ScrollWheelEvent) {
        if self.viewport_cursor(event.position).is_none() {
            return;
        }
        // The camera dollies per wheel *line*. A precise (trackpad) delta arrives
        // in pixels, and feeding 120 pixels in as 120 lines is a camera that
        // flies through its focal point on one flick, so pixels are converted
        // first: one notch of a wheel is about this many pixels.
        const PIXELS_PER_LINE: f32 = 50.0;

        let (dx, dy) = match event.delta {
            ScrollDelta::Lines(point) => (point.x, point.y),
            ScrollDelta::Pixels(point) => (
                f32::from(point.x) / PIXELS_PER_LINE,
                f32::from(point.y) / PIXELS_PER_LINE,
            ),
        };
        self.input.scroll = (self.input.scroll.0 + dx, self.input.scroll.1 + dy);
    }

    /// The viewport's keys: the gizmo modes, undo and redo, and delete.
    ///
    /// gpui's own binding table is for actions with a menu item; these are three
    /// single-key modes, so a match is shorter than three action types.
    fn viewport_key(&mut self, event: &KeyDownEvent) {
        let keystroke = &event.keystroke;
        let control = keystroke.modifiers.control || keystroke.modifiers.platform;
        let key = keystroke.key.as_str();

        match (control, keystroke.modifiers.shift, key) {
            (false, _, "w") => self.editor.gizmo.mode = GizmoMode::Translate,
            (false, _, "e") => self.editor.gizmo.mode = GizmoMode::Rotate,
            (false, _, "r") => self.editor.gizmo.mode = GizmoMode::Scale,
            (true, false, "z") => {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.undo(scene);
            }
            (true, true, "z") => {
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.redo(scene);
            }
            (false, _, "delete") | (false, _, "backspace") => {
                let Some(entity) = self.editor.selection.selected else {
                    return;
                };
                let scene = &mut self.engine.engine_mut().scene;
                self.editor.destroy_entity(scene, entity);
            }
            _ => (),
        }
    }

    /// Resizes the engine's frame to the viewport panel, when it has changed.
    ///
    /// Device pixels, so a frame on a scaled display is rendered at the
    /// resolution it is shown at. Clamped to the device's own texture limit: a
    /// frame is one texture per target, and asking a driver for a wider one is a
    /// validation error rather than a smaller image.
    fn sync_viewport_extent(&mut self, window: &Window) {
        let scale = window.scale_factor();
        let Some(gpu) = self.engine.engine_mut().gpu.as_mut() else {
            return;
        };
        let limit = gpu.info().max_texture_size;
        let extent = chassis::Extent2d {
            width: scaled(self.viewport.0, scale, limit),
            height: scaled(self.viewport.1, scale, limit),
        };

        if gpu.extent() == extent {
            return;
        }
        log::info!("editor: viewport {}x{}", extent.width, extent.height);

        if let Err(error) = gpu.resize(extent) {
            // Once: a backend that cannot resize would otherwise log every frame,
            // and the frame still draws at the old size either way.
            if !self.resize_warned {
                log::warn!(
                    "editor: viewport resize to {}x{}: {error}",
                    extent.width,
                    extent.height
                );
                self.resize_warned = true;
            }
        }
    }

    /// One engine frame: the editor's play state decides how much simulation
    /// runs, then `adam::App` runs it.
    ///
    /// The delta comes from the engine's own clock (`Time::delta`), so there is
    /// one clock and not two.
    ///
    /// `Time::scale` is what makes pause mean pause: `mirror::PlayState` reports
    /// the seconds to simulate (zero when paused or stopped, one fixed step after
    /// `step_once`), and a scale of zero keeps those seconds out of the
    /// fixed-step accumulator, so pausing freezes the simulation instead of
    /// queueing a catch-up burst while the render half keeps running.
    ///
    /// Returns the previous viewport image when this frame replaced it, for the
    /// caller to drop from the window (it has the window; this does not).
    fn advance_frame(&mut self) -> Option<Arc<RenderImage>> {
        let delta = self.engine.engine().time.delta().as_secs_f32();
        let scene = &mut self.engine.engine_mut().scene;
        let seconds = self.editor.tick(scene, delta);
        self.engine
            .engine_mut()
            .time
            .set_scale(if seconds > 0.0 { 1.0 } else { 0.0 });

        // Before the frame: what the viewport's mouse did, and the gizmo the
        // selection now needs.
        self.apply_viewport_input();
        self.sync_gizmo_lines();

        if let Err(error) = self.engine.frame() {
            log::error!("engine frame failed: {error}");
        }

        let replaced = self.capture_frame();

        if self.editor.theme != self.theme_source {
            self.theme_source = self.editor.theme;
            self.theme = EditorTheme::from(&self.theme_source);
        }

        replaced
    }

    /// Turns the renderer's last readback frame into a gpui image.
    ///
    /// Returns the image it replaced, if any: gpui's sprite atlas keeps a tile
    /// per image id until the image is dropped, and every frame is a new id, so
    /// holding the old one back would grow the atlas by a full-frame texture
    /// every frame. `None` when there was no new frame, in which case the last
    /// one stays (a device that cannot read back keeps showing the reason rather
    /// than flickering).
    fn capture_frame(&mut self) -> Option<Arc<RenderImage>> {
        let frame = self
            .engine
            .plugin_mut::<bifrost::BifrostPlugin>()
            .and_then(|plugin| plugin.frame_image())
            .map(viewport::render_image)?;
        self.frame.replace(frame)
    }

    /// Routes a dropped file by extension: a scene replaces the scene, a model
    /// or image becomes an entity. An unknown extension is logged and ignored.
    fn drop_files(&mut self, paths: &[PathBuf], cx: &mut Context<Self>) {
        for path in paths {
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase);
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "Entity".to_string());

            match extension.as_deref() {
                Some("unhuman") => {
                    let engine = self.engine.engine_mut();
                    match self
                        .editor
                        .open_scene(&mut engine.scene, &mut engine.assets, path)
                    {
                        Ok(()) => log::info!("editor: opened scene {}", path.display()),
                        Err(error) => log::warn!("editor: scene {}: {error}", path.display()),
                    }
                }
                Some("gltf") | Some("glb") => {
                    let engine = self.engine.engine_mut();
                    match mirror::import_path(&mut engine.assets, path) {
                        Some(virtual_path) => {
                            self.editor.add_model(
                                &mut engine.scene,
                                &mut engine.assets,
                                &virtual_path,
                                &name,
                            );
                            log::info!("editor: added model {virtual_path}");
                        }
                        None => log::warn!("editor: model {} is not loadable", path.display()),
                    }
                }
                Some("png") | Some("jpg") | Some("jpeg") | Some("bmp") | Some("gif")
                | Some("ktx2") => {
                    let engine = self.engine.engine_mut();
                    match mirror::import_path(&mut engine.assets, path) {
                        Some(virtual_path) => {
                            self.editor.add_image(
                                &mut engine.scene,
                                &mut engine.assets,
                                &virtual_path,
                                &name,
                            );
                            log::info!("editor: added image {virtual_path}");
                        }
                        None => log::warn!("editor: image {} is not loadable", path.display()),
                    }
                }
                other => log::warn!(
                    "editor: unsupported dropped file ({}) {}",
                    other.unwrap_or("no extension"),
                    path.display()
                ),
            }
        }
        cx.notify();
    }

    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
                |this, _window, cx| {
                    let scene = &mut this.engine.engine_mut().scene;
                    this.editor.play(scene);
                    cx.notify();
                },
            ))
            .child(button(
                "toolbar-pause",
                "Pause",
                status == PlayStatus::Paused,
                &theme,
                cx,
                |this, _window, cx| {
                    this.editor.pause();
                    cx.notify();
                },
            ))
            .child(button(
                "toolbar-step",
                "Step",
                false,
                &theme,
                cx,
                |this, _window, cx| {
                    this.editor.step();
                    cx.notify();
                },
            ))
            .child(button(
                "toolbar-stop",
                "Stop",
                status == PlayStatus::Stopped,
                &theme,
                cx,
                |this, _window, cx| {
                    let scene = &mut this.engine.engine_mut().scene;
                    this.editor.stop(scene);
                    cx.notify();
                },
            ))
            .child(separator(&theme))
            .child(button(
                "toolbar-move",
                GizmoMode::Translate.name(),
                mode == GizmoMode::Translate,
                &theme,
                cx,
                |this, _window, cx| {
                    this.editor.gizmo.mode = GizmoMode::Translate;
                    cx.notify();
                },
            ))
            .child(button(
                "toolbar-rotate",
                GizmoMode::Rotate.name(),
                mode == GizmoMode::Rotate,
                &theme,
                cx,
                |this, _window, cx| {
                    this.editor.gizmo.mode = GizmoMode::Rotate;
                    cx.notify();
                },
            ))
            .child(button(
                "toolbar-scale",
                GizmoMode::Scale.name(),
                mode == GizmoMode::Scale,
                &theme,
                cx,
                |this, _window, cx| {
                    this.editor.gizmo.mode = GizmoMode::Scale;
                    cx.notify();
                },
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

    fn status_bar(&self) -> impl IntoElement {
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
            .child(div().flex_1())
            .child(self.theme_source.name())
    }

    /// A drag handle. The bar starts the drag; the move and release listeners
    /// live in its `canvas` paint callback, the one place gpui lets a view
    /// register a window-level mouse listener. An element's own `on_mouse_move`
    /// stops firing once the pointer leaves the 5px bar.
    fn splitter(&self, splitter: Splitter, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = self.theme;
        let dragging = self
            .layout
            .drag
            .is_some_and(|drag| drag.splitter == splitter);
        let entity = cx.entity();

        let element = div()
            .id(splitter.id())
            .flex_none()
            .bg(if dragging { theme.accent } else { theme.border })
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, event: &MouseDownEvent, _window, cx| {
                    this.layout.begin_drag(
                        splitter,
                        event.position.x.as_f32(),
                        event.position.y.as_f32(),
                    );
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
                                    let resized = this.layout.drag_to(
                                        event.position.x.as_f32(),
                                        event.position.y.as_f32(),
                                    );
                                    if resized {
                                        cx.notify();
                                    }
                                });
                            }
                        });
                        window.on_mouse_event(move |_: &MouseUpEvent, _phase, _window, cx| {
                            entity.update(cx, |this, cx| {
                                if this.layout.end_drag() {
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

impl Render for EditorApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = self.theme;
        let scene = &self.engine.engine().scene;
        let hierarchy_width = self.layout.hierarchy_width;
        let inspector_width = self.layout.inspector_width;
        let content_height = self.layout.content_height;

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(theme.bg)
            .text_color(theme.text)
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _window, cx| {
                let paths = paths.paths().to_vec();
                this.drop_files(&paths, cx);
            }))
            .child(self.toolbar(cx))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        div()
                            .flex_none()
                            .h_full()
                            .w(px(hierarchy_width))
                            .child(panels::hierarchy::panel(&self.editor, scene, &theme, cx)),
                    )
                    .child(self.splitter(Splitter::Hierarchy, cx))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .child(panels::viewport::panel(
                                &theme,
                                self.frame.as_ref(),
                                self.viewport_surface(cx),
                            )),
                    )
                    .child(self.splitter(Splitter::Inspector, cx))
                    .child(
                        div()
                            .flex_none()
                            .h_full()
                            .w(px(inspector_width))
                            .child(panels::inspector::panel(&self.editor, scene, &theme, cx)),
                    ),
            )
            .child(self.splitter(Splitter::Content, cx))
            .child(
                div()
                    .flex_none()
                    .w_full()
                    .h(px(content_height))
                    .child(panels::content_browser::panel(&self.editor, &theme, cx)),
            )
            .child(self.status_bar())
    }
}

/// One gpui frame is one engine frame: advance the engine, mark the view dirty,
/// and ask for the frame after this one.
fn next_frame(this: &mut EditorApp, window: &mut Window, cx: &mut Context<EditorApp>) {
    // Before the engine advances: the size is the panel's, as the previous paint
    // measured it, so a window that changed size renders at the new one now
    // rather than a frame later.
    this.sync_viewport_extent(window);

    if let Some(replaced) = this.advance_frame() {
        // gpui's sprite atlas holds the tile until the image is dropped, and a
        // readback frame is a new image every time. Dropping the previous one
        // here keeps the atlas at one frame instead of growing it by one each
        // frame.
        if let Err(error) = window.drop_image(replaced) {
            log::warn!("editor: could not drop the previous viewport image: {error}");
        }
    }
    cx.notify();
    cx.on_next_frame(window, next_frame);
}

fn separator(theme: &EditorTheme) -> impl IntoElement {
    div().flex_none().w(px(1.0)).h(px(18.0)).bg(theme.border)
}

pub fn play_label(status: PlayStatus) -> &'static str {
    match status {
        PlayStatus::Stopped => "stopped",
        PlayStatus::Playing => "playing",
        PlayStatus::Paused => "paused",
    }
}

pub fn entity_label(count: usize) -> String {
    if count == 1 {
        "1 entity".to_string()
    } else {
        format!("{count} entities")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Splitter {
    Hierarchy,
    Inspector,
    Content,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SplitterAxis {
    Width,
    Height,
}

impl Splitter {
    fn id(self) -> &'static str {
        match self {
            Splitter::Hierarchy => "splitter-hierarchy",
            Splitter::Inspector => "splitter-inspector",
            Splitter::Content => "splitter-content",
        }
    }

    fn axis(self) -> SplitterAxis {
        match self {
            Splitter::Hierarchy | Splitter::Inspector => SplitterAxis::Width,
            Splitter::Content => SplitterAxis::Height,
        }
    }
}

#[derive(Clone, Copy)]
struct Drag {
    splitter: Splitter,
    origin: (f32, f32),
    start: f32,
}

pub struct Layout {
    hierarchy_width: f32,
    inspector_width: f32,
    content_height: f32,
    drag: Option<Drag>,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            hierarchy_width: 240.0,
            inspector_width: 300.0,
            content_height: 160.0,
            drag: None,
        }
    }
}

impl Layout {
    fn value(&self, splitter: Splitter) -> f32 {
        match splitter {
            Splitter::Hierarchy => self.hierarchy_width,
            Splitter::Inspector => self.inspector_width,
            Splitter::Content => self.content_height,
        }
    }

    fn begin_drag(&mut self, splitter: Splitter, x: f32, y: f32) -> bool {
        if self.drag.is_some() {
            return false;
        }
        self.drag = Some(Drag {
            splitter,
            origin: (x, y),
            start: self.value(splitter),
        });
        true
    }

    /// Moves the dragged edge to follow the pointer. Returns whether the layout
    /// actually changed, so a frame that moves nothing does not redraw.
    fn drag_to(&mut self, x: f32, y: f32) -> bool {
        let Some(drag) = self.drag else {
            return false;
        };

        let (dx, dy) = (x - drag.origin.0, y - drag.origin.1);
        let moved = match drag.splitter {
            Splitter::Hierarchy => (drag.start + dx).clamp(MIN_HIERARCHY, MAX_HIERARCHY),
            // The inspector's edge moves left as its width grows, and the content
            // browser's moves up as its height grows.
            Splitter::Inspector => (drag.start - dx).clamp(MIN_INSPECTOR, MAX_INSPECTOR),
            Splitter::Content => (drag.start - dy).clamp(MIN_CONTENT, MAX_CONTENT),
        };

        let slot = match drag.splitter {
            Splitter::Hierarchy => &mut self.hierarchy_width,
            Splitter::Inspector => &mut self.inspector_width,
            Splitter::Content => &mut self.content_height,
        };
        if *slot == moved {
            return false;
        }
        *slot = moved;
        true
    }

    fn end_drag(&mut self) -> bool {
        self.drag.take().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_status_reads_as_a_word() {
        assert_eq!(play_label(PlayStatus::Stopped), "stopped");
        assert_eq!(play_label(PlayStatus::Playing), "playing");
        assert_eq!(play_label(PlayStatus::Paused), "paused");
    }

    #[test]
    fn entity_count_is_plural_where_it_should_be() {
        assert_eq!(entity_label(0), "0 entities");
        assert_eq!(entity_label(1), "1 entity");
        assert_eq!(entity_label(2), "2 entities");
    }

    #[test]
    fn a_drag_follows_the_pointer_on_its_own_edge() {
        let mut layout = Layout::default();
        assert!(layout.begin_drag(Splitter::Hierarchy, 240.0, 10.0));
        assert!(layout.drag_to(300.0, 10.0));
        assert_eq!(layout.hierarchy_width, 300.0);

        let mut layout = Layout::default();
        layout.begin_drag(Splitter::Inspector, 900.0, 10.0);
        assert!(layout.drag_to(830.0, 10.0));
        assert_eq!(layout.inspector_width, 370.0);

        let mut layout = Layout::default();
        layout.begin_drag(Splitter::Content, 10.0, 700.0);
        assert!(layout.drag_to(10.0, 640.0));
        assert_eq!(layout.content_height, 220.0);
    }

    #[test]
    fn a_drag_stops_at_its_limits() {
        let mut layout = Layout::default();
        layout.begin_drag(Splitter::Hierarchy, 0.0, 0.0);
        assert!(layout.drag_to(-10_000.0, 0.0));
        assert_eq!(layout.hierarchy_width, MIN_HIERARCHY);
        assert!(layout.drag_to(10_000.0, 0.0));
        assert_eq!(layout.hierarchy_width, MAX_HIERARCHY);

        layout.end_drag();
        assert!(!layout.drag_to(500.0, 0.0));
    }

    #[test]
    fn a_move_with_nothing_dragged_does_nothing() {
        let mut layout = Layout::default();
        assert!(!layout.drag_to(400.0, 400.0));
        assert_eq!(layout.hierarchy_width, 240.0);
        assert_eq!(layout.content_height, 160.0);
    }

    #[test]
    fn a_second_press_does_not_steal_the_drag() {
        let mut layout = Layout::default();
        assert!(layout.begin_drag(Splitter::Hierarchy, 0.0, 0.0));
        assert!(!layout.begin_drag(Splitter::Content, 0.0, 0.0));
        assert!(layout.end_drag());
        assert!(!layout.end_drag());
    }
}
