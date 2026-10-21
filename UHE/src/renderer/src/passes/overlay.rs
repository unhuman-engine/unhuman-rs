//! The 2D and text slice's passes, wired into a bifrost frame.
//!
//! `batch2d` and `text` own how a sprite or glyph is drawn; this owns which passes
//! a frame declares, what they read and write, the buffers each batch's contents
//! need, and the bind groups their texture lookups resolve to.
//!
//! A caller queues content through [`OverlayFrame`], which has no GPU resources.
//! The frame calls begin, edit and prepare in that order, then hands the same
//! object to the backend as `Arc<dyn OverlayPasses>` to record with. The batching
//! state is behind a mutex because a backend may record on worker threads.
//! Buffers are per frame because `Resources` has no update-in-place call, so
//! `prepare` creates this frame's vertices and destroys the previous frame's.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chassis::graph::{PassScope, RenderGraph};
use chassis::{
    BindGroup, Buffer, CommandList, DescriptorSetLayoutDesc, Error, Extent, ImageView, Resources,
    Sampler, SamplerDesc, Texture, TextureDesc, TextureFormat, TextureUsage,
};
use glam::{Mat4, Vec4};

use crate::batch2d::{Batch2D, Globals, PipelineError, QuadPipeline, SpriteGlobals};
use crate::font::Atlas;
use crate::passes::{FrameTargets, OverlayPasses};
use crate::text::{TextGlobals, TextRenderer};

/// The two passes this slice draws in.
pub const PASS_2D: &str = "bifrost.2d";
pub const PASS_TEXT: &str = "bifrost.text";

/// Content a producer queues for the next frame.
///
/// Kept separate from the prepared pass so a caller never touches a GPU handle:
/// text is laid out in `edit`, once the overlay's atlas and device exist.
#[derive(Default)]
pub struct OverlayFrame {
    commands: Mutex<Vec<Command>>,
}

enum Command {
    Text {
        component: meats::Text,
        transform: Mat4,
    },
    Quad {
        transform: Mat4,
        color: Vec4,
    },
}

impl OverlayFrame {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues text at a caller transform. The overlay supplies the atlas, so the
    /// component's `font` and `font_path` are ignored.
    pub fn push_text(&self, component: meats::Text, transform: Mat4) {
        self.lock().push(Command::Text {
            component,
            transform,
        });
    }

    pub fn push_quad(&self, transform: Mat4, color: Vec4) {
        self.lock().push(Command::Quad { transform, color });
    }

    /// Drops everything queued but not yet framed.
    pub fn clear(&self) {
        self.lock().clear();
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Command>> {
        self.commands.lock().expect("overlay queue mutex poisoned")
    }

    fn take(&self) -> Vec<Command> {
        std::mem::take(&mut *self.lock())
    }
}

/// Pixel coordinates, origin top-left and y down, to clip space.
pub fn screen_projection(width: u32, height: u32) -> Mat4 {
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    Mat4::from_cols_array_2d(&[
        [2.0 / width, 0.0, 0.0, 0.0],
        [0.0, -2.0 / height, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, 1.0],
    ])
}

/// Everything a frame mutates: the batches, and the GPU objects built from them.
struct State {
    batch: Batch2D,
    texts: TextRenderer,

    sprite_vertices: Option<Buffer>,
    sprite_indices: Option<Buffer>,
    text_vertices: Option<Buffer>,
    text_indices: Option<Buffer>,
    sprite_globals: Option<Globals>,
    text_globals: Option<Globals>,

    /// One group per distinct texture a batch bound; `None` is the untextured
    /// case, which binds the 1x1 white texel in slot 0.
    sprite_textures: HashMap<Option<Texture>, BindGroup>,
    text_textures: HashMap<Texture, BindGroup>,
    /// Views minted for textures the overlay does not own, such as sprite assets.
    sprite_views: HashMap<Texture, ImageView>,
    text_views: HashMap<Texture, ImageView>,
}

impl State {
    fn new(quads: usize) -> Self {
        Self {
            batch: Batch2D::with_capacity(quads),
            texts: TextRenderer::new(),
            sprite_vertices: None,
            sprite_indices: None,
            text_vertices: None,
            text_indices: None,
            sprite_globals: None,
            text_globals: None,
            sprite_textures: HashMap::new(),
            text_textures: HashMap::new(),
            sprite_views: HashMap::new(),
            text_views: HashMap::new(),
        }
    }

    fn has_sprites(&self) -> bool {
        !self.batch.batches().is_empty()
    }

    fn has_text(&self) -> bool {
        !self.texts.runs().is_empty()
    }

    fn destroy_groups(&mut self, resources: &mut dyn Resources) {
        for (_, group) in self.sprite_textures.drain() {
            resources.destroy_bind_group(group);
        }
        for (_, group) in self.text_textures.drain() {
            resources.destroy_bind_group(group);
        }
        self.sprite_views.clear();
        self.text_views.clear();
        for mut globals in [self.sprite_globals.take(), self.text_globals.take()]
            .into_iter()
            .flatten()
        {
            globals.destroy(resources);
        }
        for buffer in [
            self.sprite_vertices.take(),
            self.sprite_indices.take(),
            self.text_vertices.take(),
            self.text_indices.take(),
        ]
        .into_iter()
        .flatten()
        {
            resources.destroy_buffer(buffer);
        }
    }
}

/// The 2D and text passes over one frame's batches.
pub struct SpriteTextOverlay {
    sprite: QuadPipeline,
    text: QuadPipeline,
    color_format: TextureFormat,
    state: Mutex<State>,
    pending: Arc<OverlayFrame>,

    /// The face every queued `meats::Text` is laid out with.
    atlas: Atlas,
    atlas_texture: Texture,
    atlas_view: ImageView,
    white: Texture,
    white_view: ImageView,
    sampler: Sampler,
}

impl SpriteTextOverlay {
    /// Builds both pipelines and uploads `atlas`. The colour format is the scene
    /// colour's: both draw into it, not into the swapchain.
    pub fn new(
        resources: &mut dyn Resources,
        color_format: TextureFormat,
        atlas: Atlas,
    ) -> Result<Self, PipelineError> {
        let sprite = QuadPipeline::sprite(resources, color_format, None)?;
        let text = crate::text::create_text_pipeline(resources, color_format, None)?;

        let atlas_texture = resources.create_texture(TextureDesc::new_2d(
            Extent::new(atlas.width.max(1), atlas.height.max(1)),
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        ))?;
        let expected = atlas.width as usize * atlas.height as usize * 4;
        if atlas.rgba8.len() == expected {
            resources.upload_texture(&atlas_texture, &atlas.rgba8, atlas.width, atlas.height)?;
        }
        let atlas_view = resources.create_image_view(&atlas_texture)?;

        let white = resources.create_texture(TextureDesc::new_2d(
            Extent::new(1, 1),
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        ))?;
        resources.upload_texture(&white, &[255, 255, 255, 255], 1, 1)?;
        let white_view = resources.create_image_view(&white)?;

        Ok(Self {
            sprite,
            text,
            color_format,
            state: Mutex::new(State::new(0)),
            pending: Arc::new(OverlayFrame::new()),
            atlas,
            atlas_texture,
            atlas_view,
            white,
            white_view,
            sampler: resources.create_sampler(&SamplerDesc::clamped())?,
        })
    }

    pub fn color_format(&self) -> TextureFormat {
        self.color_format
    }

    pub fn sprite_texture_layout(&self) -> &DescriptorSetLayoutDesc {
        self.sprite.texture_layout()
    }

    pub fn text_texture_layout(&self) -> &DescriptorSetLayoutDesc {
        self.text.texture_layout()
    }

    /// The producer handle, which outlives the prepared pass and owns no GPU
    /// resource.
    pub fn frame(&self) -> Arc<OverlayFrame> {
        Arc::clone(&self.pending)
    }

    /// The shared handle `Bifrost::set_overlay` takes.
    pub fn into_arc(self) -> Arc<dyn OverlayPasses> {
        Arc::new(self)
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("overlay mutex poisoned")
    }

    /// Starts a frame: the batches from the previous one are cleared.
    pub fn begin(&self) {
        let mut state = self.state();
        state.batch.begin();
        state.texts.begin();
    }

    /// Lays the queued content out against this overlay's atlas.
    pub fn edit(&self) {
        let commands = self.pending.take();
        if commands.is_empty() {
            return;
        }

        let mut guard = self.state();
        let state = &mut *guard;
        for command in commands {
            match command {
                Command::Text {
                    component,
                    transform,
                } => state
                    .texts
                    .push(&component, &self.atlas, self.atlas_texture, transform),
                Command::Quad { transform, color } => state.batch.push_quad(transform, color),
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && {
            let state = self.state();
            !state.has_sprites() && !state.has_text()
        }
    }

    /// Uploads this frame's batches and builds their bind groups.
    pub fn prepare(
        &self,
        resources: &mut dyn Resources,
        view_projection: Mat4,
    ) -> Result<(), Error> {
        // One `DerefMut` and then plain field projections: `state.batch` is read
        // while `state.sprite_textures` is written, and two projections through
        // a guard are not guaranteed to split.
        let mut guard = self.state();
        let state = &mut *guard;
        state.destroy_groups(resources);

        if state.has_sprites() {
            let mut globals = self.sprite.globals();
            globals.upload(resources, &SpriteGlobals::new(view_projection))?;
            state.sprite_globals = Some(globals);

            let (vertices, indices) = crate::batch2d::create_buffers(resources, &state.batch)?;
            state.sprite_vertices = Some(vertices);
            state.sprite_indices = Some(indices);

            for batch in state.batch.batches() {
                if state.sprite_textures.contains_key(&batch.texture) {
                    continue;
                }
                let view = match batch.texture {
                    None => self.white_view,
                    Some(texture) => match state.sprite_views.get(&texture) {
                        Some(view) => *view,
                        None => {
                            let view = resources.create_image_view(&texture)?;
                            state.sprite_views.insert(texture, view);
                            view
                        }
                    },
                };
                let group = resources.create_bind_group(
                    self.sprite.texture_layout(),
                    &[],
                    &[(0, view)],
                    &[(1, self.sampler)],
                )?;
                state.sprite_textures.insert(batch.texture, group);
            }
        }

        if state.has_text() {
            let mut globals = self.text.globals();
            globals.upload(resources, &TextGlobals::new(view_projection, &self.atlas))?;
            state.text_globals = Some(globals);

            let (vertices, indices) = crate::text::create_buffers(resources, &state.texts)?;
            state.text_vertices = Some(vertices);
            state.text_indices = Some(indices);

            for run in state.texts.runs() {
                if state.text_textures.contains_key(&run.texture) {
                    continue;
                }
                let view = if run.texture == self.atlas_texture {
                    self.atlas_view
                } else {
                    match state.text_views.get(&run.texture) {
                        Some(view) => *view,
                        None => {
                            let view = resources.create_image_view(&run.texture)?;
                            state.text_views.insert(run.texture, view);
                            view
                        }
                    }
                };
                let group = resources.create_bind_group(
                    self.text.texture_layout(),
                    &[],
                    &[(0, view)],
                    &[(1, self.sampler)],
                )?;
                state.text_textures.insert(run.texture, group);
            }
        }

        Ok(())
    }

    /// Drops every GPU resource this overlay owns.
    pub fn destroy(&self, resources: &mut dyn Resources) {
        let mut guard = self.state();
        guard.destroy_groups(resources);
        resources.destroy_pipeline(self.sprite.pipeline());
        resources.destroy_shader(self.sprite.shader());
        resources.destroy_pipeline(self.text.pipeline());
        resources.destroy_shader(self.text.shader());
        resources.destroy_texture(self.atlas_texture);
        resources.destroy_texture(self.white);
        resources.destroy_sampler(self.sampler);
    }

    pub fn sprite_pipeline(&self) -> &QuadPipeline {
        &self.sprite
    }

    pub fn text_pipeline(&self) -> &QuadPipeline {
        &self.text
    }
}

impl OverlayPasses for SpriteTextOverlay {
    fn begin(&self) {
        SpriteTextOverlay::begin(self);
    }

    fn edit(&self) {
        SpriteTextOverlay::edit(self);
    }

    fn prepare(&self, resources: &mut dyn Resources, view_projection: Mat4) -> Result<(), Error> {
        SpriteTextOverlay::prepare(self, resources, view_projection)
    }

    fn declare(&self, graph: &mut RenderGraph, targets: &FrameTargets) {
        let state = self.state();
        // Both load the scene colour: the 3D passes wrote it, and the frame's
        // composition reads it after these.
        if state.has_sprites() {
            graph.pass(PASS_2D).color_load(targets.backbuffer);
        }
        if state.has_text() {
            graph.pass(PASS_TEXT).color_load(targets.backbuffer);
        }
    }

    fn record(
        &self,
        pass: &str,
        _scope: &PassScope<'_>,
        list: &mut dyn CommandList,
    ) -> Result<(), Error> {
        let state = self.state();

        match pass {
            PASS_2D => {
                let (Some(vertices), Some(indices)) = (state.sprite_vertices, state.sprite_indices)
                else {
                    return Ok(());
                };
                let groups = &state.sprite_textures;
                let globals = state
                    .sprite_globals
                    .as_ref()
                    .and_then(Globals::group)
                    .ok_or_else(|| {
                        Error::InvalidDescriptor(
                            "pass 'bifrost.2d' has no globals: prepare did not run".to_owned(),
                        )
                    })?;
                state.batch.record(
                    list,
                    &self.sprite.pipeline(),
                    &vertices,
                    &indices,
                    &globals,
                    |texture| {
                        *groups.get(&texture).expect(
                            "a batch texture with no group: prepare skipped one, which is a bug",
                        )
                    },
                );
                Ok(())
            }
            PASS_TEXT => {
                let (Some(vertices), Some(indices)) = (state.text_vertices, state.text_indices)
                else {
                    return Ok(());
                };
                let groups = &state.text_textures;
                let globals = state
                    .text_globals
                    .as_ref()
                    .and_then(Globals::group)
                    .ok_or_else(|| {
                        Error::InvalidDescriptor(
                            "pass 'bifrost.text' has no globals: prepare did not run".to_owned(),
                        )
                    })?;
                state.texts.record(
                    list,
                    &self.text.pipeline(),
                    &vertices,
                    &indices,
                    &globals,
                    |texture| {
                        *groups
                            .get(&texture)
                            .expect("an atlas with no group: prepare skipped one, which is a bug")
                    },
                );
                Ok(())
            }
            other => Err(Error::InvalidDescriptor(format!(
                "the overlay has no pass called '{other}'"
            ))),
        }
    }

    fn destroy(&self, resources: &mut dyn Resources) {
        SpriteTextOverlay::destroy(self, resources);
    }
}
