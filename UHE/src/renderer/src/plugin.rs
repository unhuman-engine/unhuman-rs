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
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::camera::{CameraInput, EditorCamera};
use crate::font;
use crate::passes::{OverlayFrame, OverlayPasses, SpriteTextOverlay};
use crate::renderer::{Bifrost, ViewSource};

pub struct BifrostPlugin {
    renderer: Option<Bifrost>,
    camera: EditorCamera,
    input: CameraInput,
    overlay: Option<Arc<dyn OverlayPasses>>,
    /// Whether `build` should install the standard sprite/text overlay.
    hud_requested: bool,
    /// The producer half of that overlay: content queued here reaches the next
    /// frame without handing the caller a GPU pass.
    hud: Option<Arc<OverlayFrame>>,
    /// World-space lines the frame draws: the editor's gizmo.
    lines: Option<(Vec<crate::passes::LineVertex>, Vec<u32>)>,
    /// Whether `lines` was set since it was last uploaded.
    lines_pending: bool,
    /// The physics debug view, drawn as a separate batch so the gizmo's depth
    /// bias does not pull the simulation's wireframes forward too.
    debug_lines: Option<(Vec<crate::passes::LineVertex>, Vec<u32>)>,
    debug_pending: bool,
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
            hud_requested: false,
            hud: None,
            lines: None,
            lines_pending: false,
            debug_lines: None,
            debug_pending: false,
            frame_failures: 0,
        }
    }

    /// Installs the 2D and text slice's passes.
    pub fn with_overlay(mut self, overlay: Arc<dyn OverlayPasses>) -> Self {
        self.overlay = Some(overlay);
        self
    }

    /// Builds the standard sprite/text overlay in `build`, loading the engine's
    /// face through `vault`. `hud` then queues lines for the next frame.
    pub fn with_hud(mut self) -> Self {
        self.hud_requested = true;
        self
    }

    /// The HUD producer, if `with_hud` was asked for and the font loaded.
    pub fn hud(&self) -> Option<&OverlayFrame> {
        self.hud.as_deref()
    }

    /// Queues one line of screen-space text at `position` pixels from the
    /// top-left, `size` pixels per line.
    pub fn push_hud(&self, text: impl Into<String>, position: Vec2, size: f32, color: Vec4) {
        let Some(hud) = &self.hud else {
            return;
        };
        let component = meats::Text {
            text: text.into(),
            color,
            ..meats::Text::default()
        };
        let transform = Mat4::from_translation(position.extend(0.0))
            * Mat4::from_scale(Vec3::splat(size.max(1.0)));
        hud.push_text(component, transform);
    }

    /// Drops every queued HUD line; a producer calls this before pushing a frame's.
    pub fn clear_hud(&self) {
        if let Some(hud) = &self.hud {
            hud.clear();
        }
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

    /// Matches the renderer's timing state to what the device granted.
    pub fn set_gpu_timing(&mut self, enabled: bool) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_gpu_timing(enabled);
        }
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

    /// Sets the physics debug view from `femur`'s world-space segments.
    ///
    /// Each layer is a field of `PhysicsDebug`, so a caller toggles one by
    /// clearing it (or by calling [`Self::set_debug_segments`] with a filtered
    /// list) without rebuilding the others.
    pub fn set_physics_debug(&mut self, debug: femur::PhysicsDebug) {
        let (vertices, indices) = crate::passes::lines::physics_debug_batch(&debug);
        self.set_debug_segments_raw(vertices, indices);
    }

    /// Sets an arbitrary list of world-space segments, in the same format.
    pub fn set_debug_segments(&mut self, segments: &[femur::DebugLine]) {
        let (vertices, indices) = crate::passes::lines::debug_batch(segments);
        self.set_debug_segments_raw(vertices, indices);
    }

    /// Removes the physics debug view. The next frame draws nothing.
    pub fn clear_physics_debug(&mut self) {
        self.set_debug_segments_raw(Vec::new(), Vec::new());
    }

    fn set_debug_segments_raw(
        &mut self,
        vertices: Vec<crate::passes::LineVertex>,
        indices: Vec<u32>,
    ) {
        self.debug_lines = Some((vertices, indices));
        self.debug_pending = true;
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

/// Builds the standard overlay from the engine's face, plus its producer handle.
///
/// The face is resolved through `vault` so it travels with the asset root; a
/// generator options default covers Latin-1, which is every HUD string.
fn sprite_text_overlay(
    gpu: &mut dyn chassis::Gpu,
    assets: &mut vault::Assets,
) -> Result<(Arc<dyn OverlayPasses>, Arc<OverlayFrame>), String> {
    let id = assets
        .load(font::DEFAULT_FONT_PATH)
        .map_err(|error| format!("{}: {error}", font::DEFAULT_FONT_PATH))?;
    let ttf = assets
        .bytes(id)
        .ok_or_else(|| format!("{} has no bytes", font::DEFAULT_FONT_PATH))?;
    let atlas = font::generate::generate(ttf, &font::GeneratorOptions::default())
        .map_err(|error| format!("{}: {error}", font::DEFAULT_FONT_PATH))?;

    // The scene colour, which is where both 2D passes draw.
    let overlay = SpriteTextOverlay::new(gpu.resources(), chassis::TextureFormat::Rgba8Srgb, atlas)
        .map_err(|error| error.to_string())?;
    let frame = overlay.frame();
    Ok((overlay.into_arc(), frame))
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
        if self.hud_requested {
            match sprite_text_overlay(gpu.as_mut(), &mut eng.assets) {
                Ok((overlay, frame)) => {
                    renderer.set_overlay(overlay);
                    self.hud = Some(frame);
                }
                // No face is a missing HUD, not a missing frame.
                Err(error) => log::warn!("bifrost: HUD disabled: {error}"),
            }
        } else if let Some(overlay) = self.overlay.take() {
            renderer.set_overlay(overlay);
        }

        self.renderer = Some(renderer);
        Ok(())
    }

    fn on_frame(&mut self, eng: &mut Engine, ctx: &mut FrameCtx) {
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

        if self.debug_pending {
            self.debug_pending = false;
            let lines = self.debug_lines.clone();
            if let Some((vertices, indices)) = lines {
                if let Err(error) = renderer.set_debug_lines(gpu.as_mut(), &vertices, &indices) {
                    log::warn!("bifrost: physics debug lines: {error}");
                }
            }
        }

        // Upload anything the frame will want: a hit after the first frame is a
        // hash lookup. An entity whose mesh carries a skeleton gets its runtime
        // animator here, so a producer only has to add `meats::Animator`.
        for sample in eng.scene.meshes() {
            if let Err(error) = renderer.ensure_mesh(gpu.as_mut(), &eng.assets, sample.mesh) {
                log::warn!("bifrost: mesh {:?} failed to upload: {error}", sample.mesh);
                continue;
            }
            let Some(state) = sample.animator else {
                continue;
            };
            if !renderer.is_skinned(sample.entity) {
                let hint = eng
                    .scene
                    .get(sample.entity)
                    .and_then(|entry| entry.get::<meats::Animator>().map(|a| a.clip_path.clone()))
                    .unwrap_or_default();
                match renderer.attach_skinned(
                    gpu.as_mut(),
                    &eng.assets,
                    sample.entity,
                    sample.mesh,
                    (!hint.is_empty()).then_some(hint.as_str()),
                ) {
                    Ok(true) => {}
                    Ok(false) => log::debug!(
                        "bifrost: {:?} has no skeleton or clip to animate",
                        sample.mesh
                    ),
                    Err(error) => {
                        log::warn!("bifrost: {:?} skinning setup failed: {error}", sample.mesh)
                    }
                }
            }
            renderer.set_skinned_playback(sample.entity, state.playing, state.speed, state.looping);
        }
        renderer.advance_skinned(&eng.scene, ctx.step_seconds());
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
        self.hud = None;
    }
}
