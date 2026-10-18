//! The renderer as a plugin.
//!
//! `Engine` owns data and plugins own behavior (spine section 1 rule 4), so
//! `Bifrost` lives here and never in `adam::Engine`. That keeps `adam` free of a
//! dependency on a renderer.
//!
//! `build` succeeds with `gpu: None`, because headless is the same binary with
//! no device (rule 6) and a renderer that refuses to construct would make the
//! headless run a different program. With no device there is no renderer and
//! every hook returns.

use std::sync::Arc;

use adam::{Engine, FrameCtx, Plugin};

use crate::camera::{CameraInput, EditorCamera};
use crate::passes::OverlayPasses;
use crate::renderer::{Bifrost, ViewSource};

pub struct BifrostPlugin {
    renderer: Option<Bifrost>,
    camera: EditorCamera,
    input: CameraInput,
    overlay: Option<Arc<dyn OverlayPasses>>,
    /// World-space lines the frame draws: the editor's gizmo.
    lines: Option<(Vec<crate::passes::LineVertex>, Vec<u32>)>,
    /// Whether `lines` was set since it was last uploaded.
    lines_pending: bool,
    /// Frames that failed to record, so a backend that refuses every frame does
    /// not log one error per frame for the life of the process.
    frame_failures: u32,
}

impl BifrostPlugin {
    pub fn new() -> Self {
        let mut camera = EditorCamera::new(45.0, 16.0 / 9.0, 0.1, 1000.0);
        // Tilt off the ground plane. At the camera's default pitch of zero the
        // focal point and the eye are both at y = 0, so the eye lies in the
        // grid's own plane, the grid is edge-on, and the viewport is empty.
        //
        // Positive, because pitch is elevation: the eye is `focal - forward *
        // distance`, and `forward.y` is `-sin(pitch)`. A negative pitch puts the
        // eye *below* y = 0 looking up, so the first frame is the grid's
        // underside and reads as an upside-down viewport.
        camera.set_pitch(0.5);
        camera.set_yaw(0.6);
        Self {
            renderer: None,
            camera,
            input: CameraInput::default(),
            overlay: None,
            lines: None,
            lines_pending: false,
            frame_failures: 0,
        }
    }

    /// Installs the 2D and text slice's passes.
    pub fn with_overlay(mut self, overlay: Arc<dyn OverlayPasses>) -> Self {
        self.overlay = Some(overlay);
        self
    }

    /// The orbit camera. A caller with input owns the one copy of it and writes
    /// it here.
    pub fn camera_mut(&mut self) -> &mut EditorCamera {
        &mut self.camera
    }

    pub fn camera(&self) -> &EditorCamera {
        &self.camera
    }

    /// Where a caller copies `eyes::Input` into: bifrost's manifest has no
    /// `eyes`, and the camera takes the state as a value either way.
    pub fn input_mut(&mut self) -> &mut CameraInput {
        &mut self.input
    }

    pub fn renderer(&self) -> Option<&Bifrost> {
        self.renderer.as_ref()
    }

    pub fn renderer_mut(&mut self) -> Option<&mut Bifrost> {
        self.renderer.as_mut()
    }

    /// The last frame the renderer read back to the CPU, or `None` before the
    /// first frame or with a surface.
    pub fn frame_image(&self) -> Option<&crate::renderer::FrameImage> {
        self.renderer.as_ref().and_then(Bifrost::frame_image)
    }

    /// Sets the world-space lines the frame draws, uploaded on the next frame.
    ///
    /// The editor's gizmo is the caller. Held here rather than uploaded by the
    /// caller because uploading needs the device, and a caller holding the plugin
    /// and the engine at once would be two mutable borrows of one `App`.
    pub fn set_lines(&mut self, vertices: Vec<crate::passes::LineVertex>, indices: Vec<u32>) {
        self.lines = Some((vertices, indices));
        self.lines_pending = true;
    }

    /// Resizes the camera to the device's extent, if they differ.
    ///
    /// The device is the authority on the frame's size and the renderer sizes its
    /// targets from `Gpu::extent` on the next `render`; this is the other half,
    /// the camera's viewport, which the projection's aspect ratio comes from.
    /// Returns whether anything changed.
    pub fn follow_extent(&mut self, gpu: &mut dyn chassis::Gpu) -> bool {
        let extent = gpu.extent();
        let (width, height) = self.camera.viewport_size();
        if (extent.width as f32, extent.height as f32) == (width, height) {
            return false;
        }

        self.camera
            .set_viewport_size(extent.width as f32, extent.height as f32);
        true
    }
}

impl Default for BifrostPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for BifrostPlugin {
    fn name(&self) -> &'static str {
        "bifrost"
    }

    fn build(&mut self, eng: &mut Engine) -> adam::Result<()> {
        let Some(gpu) = eng.gpu.as_mut() else {
            // Headless: there is no device and no frame to draw. Not an error.
            return Ok(());
        };

        let mut renderer = Bifrost::new(gpu.as_mut(), &mut eng.assets)?;
        if let Some(overlay) = self.overlay.take() {
            renderer.set_overlay(overlay);
        }

        self.renderer = Some(renderer);
        Ok(())
    }

    fn on_frame(&mut self, eng: &mut Engine, _ctx: &mut FrameCtx) {
        let Some(gpu) = eng.gpu.as_mut() else {
            return;
        };

        // Before the renderer is borrowed: a frontend that resized the device
        // gets targets at the new size for this frame, not the next one.
        self.follow_extent(gpu.as_mut());

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        // The editor camera moves before the scene is read, so a frame draws the
        // camera state it was given rather than the previous frame's. Scroll is
        // separate from `update`: the orbit drag is held state, a dolly is an
        // event, and `update` reads only the held part.
        self.camera.update(&self.input);
        self.camera.on_scroll(self.input.scroll.1);

        if self.lines_pending {
            self.lines_pending = false;
            let lines = self.lines.clone();
            if let Some((vertices, indices)) = lines {
                if let Err(error) = renderer.set_lines(gpu.as_mut(), &vertices, &indices) {
                    log::warn!("bifrost: gizmo lines: {error}");
                }
            }
        }

        // Upload anything the frame will want: a hit after the first frame is a
        // hash lookup.
        for sample in eng.scene.meshes() {
            if let Err(error) = renderer.ensure_mesh(gpu.as_mut(), &eng.assets, sample.mesh) {
                log::warn!("bifrost: mesh {:?} failed to upload: {error}", sample.mesh);
            }
        }
    }

    fn on_render(&mut self, eng: &mut Engine, _ctx: &mut FrameCtx) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(gpu) = eng.gpu.as_mut() else {
            return;
        };

        // The scene's primary camera if it has one, the editor camera otherwise.
        let view = match eng.scene.primary_camera() {
            Some(entity) => ViewSource::Scene(entity),
            None => ViewSource::Editor,
        };

        if let Err(error) = renderer.render(
            gpu.as_mut(),
            &eng.jobs,
            &eng.scene,
            view,
            Some(&self.camera),
        ) {
            if self.frame_failures == 0 {
                log::error!("bifrost: frame failed: {error}");
            } else {
                log::debug!("bifrost: frame failed again ({error})");
            }
            self.frame_failures += 1;
        } else {
            self.frame_failures = 0;
        }
    }

    fn on_stop(&mut self, eng: &mut Engine) {
        if let (Some(renderer), Some(gpu)) = (self.renderer.as_mut(), eng.gpu.as_mut()) {
            renderer.destroy(gpu.as_mut());
        }
        self.renderer = None;
    }
}
