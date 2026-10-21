//! The editor's state: the engine it drives, the model it shows, and the layout.
//!
//! The root view and the frame loop live in [`crate::view`], the toolbar and
//! status bar in [`crate::chrome`], and the command bodies in [`crate::actions`],
//! so this file is only the state they share and the frame work itself.

use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use gpui::{Context, RenderImage, Window};
use meats::Entity;
use mirror::gizmo::GizmoMode;
use mirror::play_state::PlayStatus;
use mirror::Editor;

use crate::frame_source::{BifrostSource, FrameSource};
use crate::layout::{self, Layout, PanelId};
use crate::profiler::FrameNumbers;
use crate::theme::EditorTheme;
use crate::viewport;

const ENGINE_NAME: &CStr = c"unhuman-editor";

/// A frame this long is a stall (pipeline compile, resize), not the steady cost;
/// the previous rows say more about the frame rate than this one does.
const SLOW_FRAME_MS: f32 = 250.0;

/// The engine's offscreen size before the first paint measures the viewport.
const ENGINE_EXTENT: chassis::Extent2d = chassis::Extent2d {
    width: 1280,
    height: 720,
};

/// One side of the panel in device pixels, within what the device can allocate.
fn scaled(logical: f32, scale: f32, limit: u32) -> u32 {
    (logical * scale).round().clamp(1.0, limit as f32) as u32
}

pub struct EditorApp {
    pub(crate) engine: adam::App,
    pub(crate) editor: Editor,
    pub(crate) layout: Layout,
    /// The editor-side numbers `mirror::Timings` does not own.
    pub(crate) numbers: FrameNumbers,
    /// Which menu is open, by [`crate::menu::TITLES`] index.
    pub(crate) menu: Option<usize>,
    pub(crate) theme: EditorTheme,
    pub(crate) theme_source: mirror::theme::Theme,
    pub(crate) backend: Option<String>,
    /// The renderer's last completed frame, as a gpui image the viewport draws.
    pub(crate) frame: Option<Arc<RenderImage>>,
    /// The viewport panel's size in logical pixels, as the last paint measured it.
    pub(crate) viewport: (f32, f32),
    /// The viewport panel's top-left corner in window coordinates, in logical
    /// pixels. The cursor arrives in window coordinates, so this is what turns it
    /// into a position in the viewport.
    pub(crate) viewport_origin: (f32, f32),
    /// What the viewport's mouse and keyboard have accumulated for this frame.
    pub(crate) input: bifrost::camera::CameraInput,
    /// The gizmo the frame's lines were last built for, so a frame that moved
    /// neither the selection nor the camera does not re-upload them.
    pub(crate) gizmo_lines: Option<(
        GizmoMode,
        Option<mirror::gizmo::GizmoTransform>,
        Option<Entity>,
    )>,
    /// What the viewport's key events arrive through: an element only sees keys
    /// while it is focused, and clicking the viewport is what focuses it.
    pub(crate) focus_handle: gpui::FocusHandle,
    pub(crate) layout_path: PathBuf,
    /// Whether a refused viewport resize has been reported, so it is reported once.
    resize_warned: bool,
    /// Whether per-pass GPU timings were granted; the profiler panel is the one
    /// thing that asks for them, and only once it is open.
    profiler_gpu_timing: bool,
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
        let layout_path = layout::config_path();

        let mut this = Self {
            engine: app,
            editor,
            layout: Layout::load(&layout_path),
            numbers: FrameNumbers::default(),
            menu: None,
            theme: EditorTheme::from(&theme_source),
            theme_source,
            backend,
            frame: None,
            viewport: (ENGINE_EXTENT.width as f32, ENGINE_EXTENT.height as f32),
            viewport_origin: (0.0, 0.0),
            input: bifrost::camera::CameraInput::default(),
            gizmo_lines: None,
            focus_handle: cx.focus_handle(),
            layout_path,
            resize_warned: false,
            profiler_gpu_timing: false,
        };

        // A fresh launch starts on a scene with a camera, a light and a model, so
        // the hierarchy and inspector are not empty. File > New Scene lands here
        // too.
        let engine = this.engine.engine_mut();
        this.editor
            .new_starter_scene(&mut engine.scene, &mut engine.assets);

        // A layout restored from disk can have the profiler already open.
        if this.layout.is_open(PanelId::Profiler) {
            this.enable_profiler_gpu_timing();
        }
        this
    }

    /// Asks the device for per-pass timestamps, once, when the profiler opens.
    ///
    /// The device reports whether it can; the renderer is told the same answer so
    /// a backend without timing does not pay for a wait every frame.
    pub(crate) fn enable_profiler_gpu_timing(&mut self) {
        if self.profiler_gpu_timing {
            return;
        }
        let enabled = self
            .engine
            .engine_mut()
            .gpu
            .as_mut()
            .is_some_and(|gpu| gpu.enable_gpu_timing());
        if let Some(plugin) = self.engine.plugin_mut::<bifrost::BifrostPlugin>() {
            plugin.set_gpu_timing(enabled);
        }
        self.profiler_gpu_timing = enabled;
    }

    /// Schedules the first engine frame. Every frame re-arms this one, so the
    /// engine advances at the rate the platform draws.
    pub fn start_frame_loop(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        cx.on_next_frame(window, crate::view::next_frame);
    }

    // ---- the frame loop ----------------------------------------------------

    /// Resizes the engine's frame to the viewport panel, when it has changed.
    ///
    /// Device pixels, so a frame on a scaled display is rendered at the
    /// resolution it is shown at. Clamped to the device's own texture limit: a
    /// frame is one texture per target, and asking a driver for a wider one is a
    /// validation error rather than a smaller image.
    pub(crate) fn sync_viewport_extent(&mut self, window: &Window) {
        let scale = window.scale_factor();
        let extent = {
            let Some(gpu) = self.engine.engine_mut().gpu.as_mut() else {
                return;
            };
            let limit = gpu.info().max_texture_size;
            chassis::Extent2d {
                width: scaled(self.viewport.0, scale, limit),
                height: scaled(self.viewport.1, scale, limit),
            }
        };

        // The camera's projection is the panel's aspect ratio, so it follows the
        // measurement now rather than after the next renderer hook.
        let camera_size = (extent.width as f32, extent.height as f32);
        let camera_stale = self
            .engine
            .plugin::<bifrost::BifrostPlugin>()
            .is_some_and(|plugin| plugin.camera().viewport_size() != camera_size);
        if camera_stale {
            if let Some(plugin) = self.engine.plugin_mut::<bifrost::BifrostPlugin>() {
                plugin
                    .camera_mut()
                    .set_viewport_size(camera_size.0, camera_size.1);
            }
        }

        let Some(gpu) = self.engine.engine_mut().gpu.as_mut() else {
            return;
        };
        if gpu.extent() == extent {
            return;
        }
        log::info!("editor: viewport {}x{}", extent.width, extent.height);

        if let Err(error) = gpu.resize(extent) {
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
    /// runs, then `adam::App` runs it. Returns the viewport image this frame
    /// replaced, for the caller to drop from the window.
    pub(crate) fn advance_frame(&mut self) -> Option<Arc<RenderImage>> {
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

        let started = Instant::now();
        if let Err(error) = self.engine.frame() {
            // No completed frame: leaving the last rows alone is better than
            // replacing them with a frame that never happened.
            log::error!("engine frame failed: {error}");
            return None;
        }
        let engine_ms = started.elapsed().as_secs_f32() * 1000.0;

        let (replaced, readback_ms) = self.capture_frame();

        let scene = &self.engine.engine().scene;
        self.numbers.entities = scene.entities().len();
        self.numbers.meshes = scene.meshes().len();

        // The editor sees one engine frame boundary, so record/submit is the
        // whole frame until bifrost exposes its own split.
        let timings = mirror::Timings {
            loop_ms: self.editor.profiler.stats().frame_ms,
            engine_ms,
            render_ms: engine_ms,
            readback_ms,
        };

        let counters = mirror::Counters {
            instances: self.numbers.meshes as u32,
            ..Default::default()
        };
        let now = Instant::now();
        let has_good_rows = self.editor.profiler.frame().timings.engine_ms > 0.0;
        if has_good_rows && engine_ms > SLOW_FRAME_MS {
            // Keep the steady rows; only the loop and OS samples move.
            self.editor.profiler.publish(now);
        } else {
            self.editor.profiler.record(timings, counters, now);
        }

        // Timings the renderer collected from the frame it just recorded; only
        // read when the panel asked for them, so a hidden panel costs nothing.
        if self.profiler_gpu_timing {
            let passes: Vec<mirror::PassTime> = self
                .engine
                .plugin::<bifrost::BifrostPlugin>()
                .and_then(bifrost::BifrostPlugin::renderer)
                .map(|renderer| {
                    renderer
                        .gpu_passes()
                        .iter()
                        .map(|timing| mirror::PassTime {
                            pass: timing.name.clone(),
                            ms: timing.ms,
                        })
                        .collect()
                })
                .unwrap_or_default();
            self.editor.profiler.frame_mut().gpu_passes = passes;
        }

        if self.editor.theme != self.theme_source {
            self.theme_source = self.editor.theme;
            self.theme = EditorTheme::from(&self.theme_source);
        }

        replaced
    }

    /// Turns the renderer's newest frame into a gpui image.
    ///
    /// Returns the image it replaced, if any: gpui's sprite atlas keeps a tile
    /// per image id until the image is dropped, and every frame is a new id, so
    /// holding the old one back would grow the atlas by a full-frame texture
    /// every frame. `None` when the renderer produced no new frame, in which
    /// case the last one stays.
    ///
    /// The poll is the readback seam: it hands over the frame read from a slot
    /// `READBACK_LAG` submissions back, so its cost is the CPU copy rather than
    /// a wait on the frame just recorded.
    fn capture_frame(&mut self) -> (Option<Arc<RenderImage>>, Option<f32>) {
        let Some(plugin) = self.engine.plugin_mut::<bifrost::BifrostPlugin>() else {
            self.numbers.upload_ms = 0.0;
            return (None, None);
        };
        let started = Instant::now();
        let frame = BifrostSource::new(plugin).poll();
        let readback_ms = started.elapsed().as_secs_f32() * 1000.0;

        let started = Instant::now();
        let replaced = match frame {
            Some(frame) => self.frame.replace(viewport::render_image(frame)),
            None => None,
        };
        self.numbers.upload_ms = started.elapsed().as_secs_f32() * 1000.0;
        (replaced, Some(readback_ms))
    }
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
}
