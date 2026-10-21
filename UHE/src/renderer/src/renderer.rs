//! The renderer frontend: the frame, and everything the frame owns.
//!
//! The frame's state lives in fields on [`Bifrost`], not a static, so a renderer
//! can be constructed in isolation and `tests/scene` need not recompile engine
//! sources to get one.
//!
//! A frame resolves its camera ([`ViewSource`] or the editor camera), declares
//! its passes into its own render graph (reads and writes only, because the
//! graph owns every barrier), compiles the graph, prepares the uniform sets and
//! the draw list, then hands the compiled graph, the slot table its textures
//! were minted into, and one [`PassRecorder`] to the backend. Recording,
//! submitting and presenting stay the backend's: it owns the executor, the
//! command lists and the swapchain, and a frontend that recorded its own frame
//! would have to name a driver to do it.

use std::collections::HashMap;
use std::sync::Arc;

use chassis::graph::{
    BufferId, CompiledGraph, ImageState, PassRecorder, PassType, RenderGraph, SlotTable, TextureId,
};
use chassis::{
    BindGroup, BufferDesc, BufferUsage, Error, Extent, Extent2d, Gpu, GpuPassTime, ImageView,
    Resources, Sampler, SamplerDesc, Submission, Texture, TextureDesc, TextureFormat, TextureUsage,
};
use glam::{Mat4, Vec3};

use crate::animation::{animator_from_model, select_clip, Animator, LoopMode};
use crate::camera::{CameraInput, EditorCamera};
use crate::frame_ring::{frame_depth, FrameRing};
use crate::light::{self, LightBuffer, LightData};
use crate::material::{
    self, CachedTexture, Material, MaterialBinding, MaterialCache, MaterialKey, MaterialTextures,
    MaterialVariant, TextureCache,
};
use crate::mesh::{self, GpuMesh, MeshCache, MeshKey, VertexLayoutKey};
use crate::passes::{
    self, uniforms, CompositePass, DrawArena, EditorPass, FrameRecorder, FrameTargets,
    GeometryPass, GridPass, GridUniforms, ImageSet, LineBatch, LinePass, LineUniforms,
    MaterialPipelines, OverlayPasses, PaletteSet, PipelineCache, PipelineKey, PreparedDraw,
    ReadbackPass, ReadbackRing, ShaderSet, ShaderVariant, ShadowPass, SsaoPass, SsaoUniforms,
};

/// Where the frame's camera comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ViewSource {
    /// A scene camera entity: its `Camera` component's projection, and the
    /// entity's world transform as the view.
    Scene(meats::Entity),
    /// The editor's orbit camera, passed alongside this to `render`.
    Editor,
}

/// The shadow map's edge, in texels.
pub const SHADOW_MAP_SIZE: u32 = 2048;

/// The directional light's orthographic half-extent, in world units. Nothing
/// else in the tree suggests a value, so a shadow needs this number.
pub const SHADOW_EXTENT: f32 = 40.0;

/// The clear colour, taken from the editor theme.
pub const CLEAR_COLOR: [f32; 4] = [0.1, 0.1, 0.12, 1.0];

/// One composed frame the renderer copied back to the CPU.
///
/// Only produced headless: a caller with a surface already has the frame on
/// screen, and a swapchain image is not the renderer's to copy.
#[derive(Clone, Debug)]
pub struct FrameImage {
    pub width: u32,
    pub height: u32,
    /// The composition target's bytes, tightly packed. `Bgra8Srgb` unless a
    /// surface said otherwise, so red and blue are swapped relative to RGBA.
    pub bgra: Vec<u8>,
}

/// The frame's own textures, recreated when the extent changes.
struct FrameTextures {
    color: Texture,
    color_view: ImageView,
    depth: Texture,
    depth_view: ImageView,
    shadow: Texture,
    shadow_view: ImageView,
    ssao: Texture,
    ssao_view: ImageView,
    /// Half-resolution occlusion, before the depth-aware upsample.
    ssao_raw: Texture,
    ssao_raw_view: ImageView,
    /// The composition target when there is no swapchain to present to.
    offscreen: Texture,
    /// Clamped linear: a render target's edge should clamp, not wrap.
    sampler: Sampler,
}

impl FrameTextures {
    /// Null handles, so a live set can be swapped out and destroyed, the same
    /// trick as a graph slot with no resource behind it.
    fn null() -> Self {
        Self {
            color: Texture::null(),
            color_view: ImageView::null(),
            depth: Texture::null(),
            depth_view: ImageView::null(),
            shadow: Texture::null(),
            shadow_view: ImageView::null(),
            ssao: Texture::null(),
            ssao_view: ImageView::null(),
            ssao_raw: Texture::null(),
            ssao_raw_view: ImageView::null(),
            offscreen: Texture::null(),
            sampler: Sampler::null(),
        }
    }

    fn create(
        resources: &mut dyn Resources,
        extent: Extent2d,
        shadow_size: u32,
        composite_format: TextureFormat,
    ) -> Result<Self, Error> {
        let width = extent.width.max(1);
        let height = extent.height.max(1);

        let color = resources.create_texture(TextureDesc::new_2d(
            Extent::new(width, height),
            // sRGB, to match the model pipeline's colour output.
            TextureFormat::Rgba8Srgb,
            TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        ))?;
        let color_view = resources.create_image_view(&color)?;

        // Depth32Float: SSAO samples the depth buffer, and a depth-stencil
        // format is not sampleable on wgpu.
        let depth = resources.create_texture(TextureDesc::new_2d(
            Extent::new(width, height),
            TextureFormat::Depth32Float,
            TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED,
        ))?;
        let depth_view = resources.create_image_view(&depth)?;

        let shadow = resources.create_texture(TextureDesc::new_2d(
            Extent::new(shadow_size, shadow_size),
            TextureFormat::Depth32Float,
            TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED,
        ))?;
        let shadow_view = resources.create_image_view(&shadow)?;

        let ssao = resources.create_texture(TextureDesc::new_2d(
            Extent::new(width, height),
            TextureFormat::R8Unorm,
            TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        ))?;
        let ssao_view = resources.create_image_view(&ssao)?;

        let ssao_raw = resources.create_texture(TextureDesc::new_2d(
            Extent::new(width.div_ceil(2), height.div_ceil(2)),
            TextureFormat::R8Unorm,
            TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        ))?;
        let ssao_raw_view = resources.create_image_view(&ssao_raw)?;

        // `TRANSFER_SRC` only matters headless: the readback pass copies this
        // image into a buffer for the CPU, and the graph's barrier for that
        // copy needs the usage declared.
        let offscreen = resources.create_texture(TextureDesc::new_2d(
            Extent::new(width, height),
            composite_format,
            TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED | TextureUsage::TRANSFER_SRC,
        ))?;

        let sampler = resources.create_sampler(&SamplerDesc::clamped())?;

        Ok(Self {
            color,
            color_view,
            depth,
            depth_view,
            shadow,
            shadow_view,
            ssao,
            ssao_view,
            ssao_raw,
            ssao_raw_view,
            offscreen,
            sampler,
        })
    }

    fn destroy(self, resources: &mut dyn Resources) {
        resources.destroy_texture(self.color);
        resources.destroy_texture(self.depth);
        resources.destroy_texture(self.shadow);
        resources.destroy_texture(self.ssao);
        resources.destroy_texture(self.ssao_raw);
        resources.destroy_texture(self.offscreen);
        resources.destroy_sampler(self.sampler);
    }
}

/// One skinned entity: its runtime animator and the palette copied from it each
/// frame, already in the draw's model space.
struct SkinnedDraw {
    animator: Animator,
    matrices: Vec<[[f32; 4]; 4]>,
    /// The clip the runtime is playing, so a scene path change is one switch.
    clip: String,
}

/// The renderer frontend. Owned by `BifrostPlugin`, never by `Engine`.
pub struct Bifrost {
    /// The device's extent, which is what a resize is detected against.
    extent: Extent2d,
    /// The extent the frame's targets are created at: `extent` with the width
    /// rounded up to a multiple of 64 texels, so the readback copy's row pitch
    /// equals the packed row and no de-pad pass is needed. Equals `extent` at
    /// every width the copy would not pad.
    render_extent: Extent2d,
    swapchain_format: TextureFormat,
    /// Whether there is a swapchain to present to. Without one the composition
    /// renders to `textures.offscreen` instead of the backbuffer.
    has_surface: bool,
    textures: FrameTextures,

    shaders: ShaderSet,
    pipelines: PipelineCache,
    meshes: MeshCache,
    materials: MaterialCache,
    textures_by_asset: TextureCache,
    /// Material descriptions by asset, for `MeshSample::material` to resolve.
    material_assets: HashMap<vault::AssetId, Material>,
    lights: LightBuffer,

    graph: RenderGraph,
    targets: Option<FrameTargets>,

    /// The ring's per-frame data. Each slot's contents were last used `depth`
    /// frames ago; `render` waits that far before rebuilding them.
    frame_sets: FrameRing<uniforms::BindSet>,
    shadow_sets: FrameRing<uniforms::BindSet>,
    ssao_sets: FrameRing<uniforms::BindSet>,
    ssao_upsample_sets: FrameRing<uniforms::BindSet>,
    /// The grid's set and the gizmo lines' set: two, because each holds its own
    /// matrices and a pass that bound one buffer for both would draw one of them
    /// through the other's matrix.
    grid_sets: FrameRing<uniforms::BindSet>,
    line_sets: FrameRing<uniforms::BindSet>,
    /// The physics debug view's set, separate from the gizmo's so toggling one
    /// does not disturb the other's uniforms.
    debug_sets: FrameRing<uniforms::BindSet>,
    image_sets: FrameRing<ImageSet>,
    palette_sets: FrameRing<PaletteSet>,
    draw_arenas: FrameRing<DrawArena>,
    /// The submission that wrote each slot, so the next lap waits only for the
    /// frame that is about to be overwritten.
    submissions: FrameRing<Option<Submission>>,

    mesh_layout: VertexLayoutKey,
    /// The animators this frame draws, keyed by the entity that owns the mesh.
    skinned: HashMap<meats::Entity, SkinnedDraw>,
    /// All attached palettes packed for this frame's draws.
    frame_palette: Vec<[[f32; 4]; 4]>,
    line_batch: Option<LineBatch>,
    debug_batch: Option<LineBatch>,
    show_grid: bool,

    overlay: Option<Arc<dyn OverlayPasses>>,
    frame_index: u64,
    /// The ring slot this frame uses, chosen (and waited for) before anything
    /// reads or writes per-frame data.
    frame_slot: usize,
    /// The buffers the composed frame is copied into and read back from, as a
    /// ring so a frame reads its predecessor's frame and not its own. `None`
    /// with a surface -- there is nothing to read back -- or when the device
    /// cannot make a readable buffer.
    readback: Option<ReadbackRing>,
    /// The last frame read back, or `None` before the first one.
    frame_image: Option<FrameImage>,
    /// Readback failures are logged once, not once per frame.
    readback_warned: bool,
    /// Whether the frame waits the submission and collects GPU timestamps.
    gpu_timing: bool,
    /// The last frame's per-pass times, owned so the panel reads stable names.
    gpu_passes: Vec<GpuPassTime>,
    /// The last frame's whole-frame GPU time.
    gpu_frame_ms: Option<f32>,
    /// A timing failure is logged once, not once per frame.
    gpu_timing_warned: bool,
}

impl Bifrost {
    /// Builds the renderer against a live device.
    ///
    /// `assets` is where a caller's meshes and images come from; the shader
    /// sources are embedded and compiled in process.
    pub fn new(gpu: &mut dyn Gpu, assets: &mut vault::Assets) -> Result<Self, Error> {
        let _ = assets;

        // The extent the executor will record at, from the GPU itself. Sizing
        // the targets from anything else -- a fixed default, a swapchain read
        // back separately -- puts them at odds with the passes' viewport,
        // headless, where the executor uses the device's own extent.
        let extent = gpu.extent();
        let render_extent = render_extent_for(extent);
        let swapchain_format = gpu
            .swapchain()
            .map(|swapchain| swapchain.format())
            .unwrap_or(TextureFormat::Bgra8Srgb);
        let has_surface = gpu.swapchain().is_some();
        let depth = frame_depth(gpu.info().frames_in_flight);

        let resources = gpu.resources();
        let shaders = ShaderSet::load(resources)?;
        let textures =
            FrameTextures::create(resources, render_extent, SHADOW_MAP_SIZE, swapchain_format)?;

        let mut bifrost = Self {
            extent,
            render_extent,
            swapchain_format,
            has_surface,
            textures,
            shaders,
            pipelines: PipelineCache::new(),
            meshes: MeshCache::new(),
            materials: MaterialCache::new(),
            textures_by_asset: TextureCache::new(),
            material_assets: HashMap::new(),
            lights: LightBuffer::new(),
            graph: RenderGraph::new(),
            targets: None,
            frame_sets: FrameRing::new(depth),
            shadow_sets: FrameRing::new(depth),
            ssao_sets: FrameRing::new(depth),
            ssao_upsample_sets: FrameRing::new(depth),
            grid_sets: FrameRing::new(depth),
            line_sets: FrameRing::new(depth),
            debug_sets: FrameRing::new(depth),
            image_sets: FrameRing::new(depth),
            palette_sets: FrameRing::new(depth),
            draw_arenas: FrameRing::new(depth),
            submissions: FrameRing::new(depth),
            mesh_layout: VertexLayoutKey::of(&mesh::vertex_layout()),
            skinned: HashMap::new(),
            frame_palette: Vec::new(),
            line_batch: None,
            debug_batch: None,
            show_grid: true,
            overlay: None,
            frame_index: 0,
            frame_slot: 0,
            readback: None,
            frame_image: None,
            readback_warned: false,
            gpu_timing: false,
            gpu_passes: Vec::new(),
            gpu_frame_ms: None,
            gpu_timing_warned: false,
        };

        // The 1x1 textures a material with no map of a kind samples, made once.
        bifrost.textures_by_asset.white(gpu.resources())?;
        bifrost.textures_by_asset.neutral_normal(gpu.resources())?;
        bifrost.recreate_readback(gpu.resources(), render_extent);

        Ok(bifrost)
    }

    /// Installs the 2D and text slice's passes into the frame.
    pub fn set_overlay(&mut self, overlay: Arc<dyn OverlayPasses>) {
        self.overlay = Some(overlay);
    }

    pub fn overlay(&self) -> Option<&Arc<dyn OverlayPasses>> {
        self.overlay.as_ref()
    }

    /// The last composed frame read back for the CPU.
    ///
    /// `None` before the first frame, with a surface (the frame is presented
    /// instead), or when the device refused a readable buffer.
    pub fn frame_image(&self) -> Option<&FrameImage> {
        self.frame_image.as_ref()
    }

    /// Starts collecting per-pass GPU times on the next frame.
    ///
    /// Returns `false` when the backend cannot write timestamps; the frame then
    /// keeps rendering without the wait a collection costs.
    pub fn enable_gpu_timing(&mut self, gpu: &mut dyn Gpu) -> bool {
        self.gpu_timing = gpu.enable_gpu_timing();
        if !self.gpu_timing {
            self.gpu_passes.clear();
            self.gpu_frame_ms = None;
        }
        self.gpu_timing
    }

    /// Sets the timing state a caller already negotiated with the device.
    pub fn set_gpu_timing(&mut self, enabled: bool) {
        self.gpu_timing = enabled;
        if !enabled {
            self.gpu_passes.clear();
            self.gpu_frame_ms = None;
        }
    }

    /// The last frame's per-pass GPU times, empty when timing is off.
    pub fn gpu_passes(&self) -> &[GpuPassTime] {
        &self.gpu_passes
    }

    /// The last frame's whole-frame GPU time, `None` when unavailable.
    pub fn gpu_frame_ms(&self) -> Option<f32> {
        self.gpu_frame_ms
    }

    /// Rebuilds the readback ring for `extent`, or drops it when there is a
    /// surface to present to instead.
    fn recreate_readback(&mut self, resources: &mut dyn Resources, extent: Extent2d) {
        if let Some(old) = self.readback.take() {
            old.destroy(resources);
        }
        self.frame_image = None;
        if self.has_surface {
            return;
        }

        // Packed, not pitch-sized: `extent` is the render extent, whose width is
        // a multiple of 64 texels, so the copy's row pitch already equals
        // `width * 4` and there is no padding to strip.
        let size = u64::from(extent.width.max(1)) * u64::from(extent.height.max(1)) * 4;
        match ReadbackRing::new(resources, size) {
            Ok(ring) => self.readback = Some(ring),
            Err(error) => {
                // Once, not once per frame: no readable buffer is a property of
                // the device, and repeating it buries the first line.
                if !self.readback_warned {
                    log::warn!(
                        "bifrost: no readback buffer ({error}); the editor frame stays on the GPU"
                    );
                    self.readback_warned = true;
                }
                self.readback = None;
            }
        }
    }

    pub fn meshes(&self) -> &MeshCache {
        &self.meshes
    }

    pub fn materials(&self) -> &MaterialCache {
        &self.materials
    }

    pub fn textures(&self) -> &TextureCache {
        &self.textures_by_asset
    }

    pub fn pipelines(&self) -> &PipelineCache {
        &self.pipelines
    }

    pub fn lights(&self) -> &LightBuffer {
        &self.lights
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// The pass slots of the last frame, which a caller can inspect or draw into.
    pub fn targets(&self) -> Option<FrameTargets> {
        self.targets
    }

    pub fn show_grid(&mut self, show: bool) {
        self.show_grid = show;
    }

    /// Registers a material description under an asset id for
    /// `MeshSample::material` to resolve.
    pub fn set_material(&mut self, id: vault::AssetId, material: Material) {
        self.material_assets.insert(id, material);
    }

    /// Attaches the model's first skin and clips to `entity`.
    ///
    /// `clip` selects by name; when it misses, Idle is preferred and the first
    /// clip is the fallback. False when the model has no skeleton or animation.
    pub fn attach_skinned(
        &mut self,
        _gpu: &mut dyn Gpu,
        assets: &vault::Assets,
        entity: meats::Entity,
        mesh: vault::AssetId,
        clip: Option<&str>,
    ) -> Result<bool, Error> {
        let (Some(meshes), Some(model)) = (assets.meshes(mesh), assets.model(mesh)) else {
            return Ok(false);
        };
        let Some(mut animator) = animator_from_model(meshes, model) else {
            return Ok(false);
        };
        let Some(index) = select_clip(animator.clips(), clip) else {
            return Ok(false);
        };
        animator.play_animation(index as i32);
        animator.seek(0.0);
        let matrices = palette_of(&animator);
        let clip = animator.clips()[index].name.clone();
        self.skinned.insert(
            entity,
            SkinnedDraw {
                animator,
                matrices,
                clip,
            },
        );
        Ok(true)
    }

    /// Whether this entity already has a runtime animator.
    pub fn is_skinned(&self, entity: meats::Entity) -> bool {
        self.skinned.contains_key(&entity)
    }

    /// The palette the last attach or advance produced for `entity`.
    pub fn skinned_palette(&self, entity: meats::Entity) -> Option<&[[[f32; 4]; 4]]> {
        self.skinned
            .get(&entity)
            .map(|draw| draw.matrices.as_slice())
    }

    /// The runtime clip index `entity` is playing, or `None` before attach.
    pub fn skinned_animation_index(&self, entity: meats::Entity) -> Option<i32> {
        self.skinned
            .get(&entity)
            .map(|draw| draw.animator.current_animation_index())
    }

    /// The runtime play head in seconds, or `None` before attach.
    pub fn skinned_animation_time(&self, entity: meats::Entity) -> Option<f32> {
        self.skinned
            .get(&entity)
            .map(|draw| draw.animator.current_time())
    }

    /// The playing clip's duration in seconds, or `None` before attach.
    pub fn skinned_clip_duration(&self, entity: meats::Entity) -> Option<f32> {
        self.skinned
            .get(&entity)
            .map(|draw| draw.animator.duration())
    }

    /// Switches `entity` to the named clip when it differs from the playing one.
    ///
    /// An unknown name leaves the last valid clip playing. Returns whether the
    /// runtime is on the requested clip, or was already.
    pub fn set_skinned_clip(&mut self, entity: meats::Entity, name: &str) -> bool {
        let Some(draw) = self.skinned.get_mut(&entity) else {
            return false;
        };
        if name.is_empty() || draw.clip == name {
            return true;
        }
        if !draw.animator.play_animation_by_name(name) {
            return false;
        }
        draw.clip.clear();
        draw.clip.push_str(name);
        draw.matrices = palette_of(&draw.animator);
        true
    }

    /// Applies a scene animator's clip and playback state to `entity`.
    pub fn apply_animator(&mut self, entity: meats::Entity, animator: &meats::Animator) {
        self.set_skinned_clip(entity, &animator.clip_path);
        self.set_skinned_playback(entity, animator.playing, animator.speed, animator.looping);
    }

    /// Copies playback state from the scene component into the runtime animator.
    pub fn set_skinned_playback(
        &mut self,
        entity: meats::Entity,
        playing: bool,
        speed: f32,
        looping: bool,
    ) {
        let Some(draw) = self.skinned.get_mut(&entity) else {
            return;
        };
        if playing {
            draw.animator.play();
        } else {
            draw.animator.pause();
        }
        draw.animator.set_time_scale(speed);
        draw.animator.set_loop_mode(if looping {
            LoopMode::Loop
        } else {
            LoopMode::None
        });
    }

    /// Jumps one entity's clip and rebuilds its palette, for scrubbing and tests.
    pub fn seek_skinned(&mut self, entity: meats::Entity, time: f32) -> bool {
        let Some(draw) = self.skinned.get_mut(&entity) else {
            return false;
        };
        draw.animator.seek(time);
        draw.matrices = palette_of(&draw.animator);
        true
    }

    /// Advances every attached animator and repacks its palette.
    ///
    /// An entity destroyed since the last frame is dropped: its palette has no
    /// draw to belong to and would otherwise grow the frame buffer forever.
    pub fn advance_skinned(&mut self, scene: &meats::Scene, dt: f32) {
        self.skinned.retain(|entity, _| scene.contains(*entity));
        for draw in self.skinned.values_mut() {
            draw.animator.update_animation(dt);
            draw.matrices = palette_of(&draw.animator);
        }
    }

    /// Uploads a mesh if it is not resident. `None` when the asset has no
    /// CPU-side mesh, which is not an error: a path that failed to load has no
    /// mesh to draw.
    pub fn ensure_mesh(
        &mut self,
        gpu: &mut dyn Gpu,
        assets: &vault::Assets,
        id: vault::AssetId,
    ) -> Result<Option<GpuMesh>, Error> {
        let key = MeshKey::new(id, self.mesh_layout);
        if let Some(mesh) = self.meshes.get(&key) {
            return Ok(Some(mesh));
        }
        let Some(data) = assets.mesh(id) else {
            return Ok(None);
        };
        Ok(Some(self.meshes.get_or_upload(
            gpu.resources(),
            key,
            data,
        )?))
    }

    /// Drops every GPU resource a mesh holds.
    pub fn release_mesh(&mut self, gpu: &mut dyn Gpu, id: vault::AssetId) -> bool {
        self.meshes
            .destroy(gpu.resources(), &MeshKey::new(id, self.mesh_layout))
    }

    /// Uploads an image as a texture. A repeated call is a cache hit.
    pub fn ensure_texture(
        &mut self,
        gpu: &mut dyn Gpu,
        assets: &vault::Assets,
        id: vault::AssetId,
    ) -> Result<(), Error> {
        if self.textures_by_asset.get(id).is_some() {
            return Ok(());
        }
        let Some(image) = assets.image(id).cloned() else {
            return Ok(());
        };
        self.textures_by_asset
            .get_or_upload(gpu.resources(), id, &image)?;
        Ok(())
    }

    /// Destroys a texture **and every material that bound it**.
    ///
    /// A bind group naming a destroyed texture is a descriptor pointing at freed
    /// memory, so the two cannot be dropped independently.
    pub fn release_texture(&mut self, gpu: &mut dyn Gpu, id: vault::AssetId) -> bool {
        let resources = gpu.resources();
        let Some(texture) = self.textures_by_asset.destroy(resources, id) else {
            return false;
        };
        self.materials.invalidate_texture(resources, texture);
        true
    }

    /// Replaces the gizmo line batch.
    pub fn set_lines(
        &mut self,
        gpu: &mut dyn Gpu,
        vertices: &[passes::LineVertex],
        indices: &[u32],
    ) -> Result<(), Error> {
        upload_lines(gpu.resources(), &mut self.line_batch, vertices, indices)
    }

    /// Replaces the physics debug batch. An empty batch clears the view.
    pub fn set_debug_lines(
        &mut self,
        gpu: &mut dyn Gpu,
        vertices: &[passes::LineVertex],
        indices: &[u32],
    ) -> Result<(), Error> {
        upload_lines(gpu.resources(), &mut self.debug_batch, vertices, indices)
    }

    /// Advances an editor camera by one frame's input.
    ///
    /// The input is passed in, so a caller with a window owns the only copy of it.
    pub fn update_editor_camera(&self, camera: &mut EditorCamera, input: &CameraInput) {
        camera.update(input);
        camera.on_scroll(input.scroll.1);
    }

    /// Renders one frame.
    pub fn render(
        &mut self,
        gpu: &mut dyn Gpu,
        jobs: &grind::Jobs,
        scene: &meats::Scene,
        view: ViewSource,
        editor_camera: Option<&EditorCamera>,
    ) -> Result<(), Error> {
        let _ = jobs;

        // The device is the authority on the frame's size: its swapchain's when
        // there is one, its own headless extent otherwise, which is exactly what
        // `Gpu::extent` answers. Reading the swapchain here as well was a second
        // answer to one question, and the headless one was stale: a device
        // resized under the renderer left the targets at the old size while the
        // executor recorded at the new one.
        let extent = gpu.extent();
        if extent.width == 0 || extent.height == 0 {
            // A minimised window: a zero extent is not a frame to draw, it is a
            // frame to skip.
            return Ok(());
        }

        // Before any per-frame GPU data is touched, retire the frame that last
        // wrote this slot. Slot `f % depth` was used by frame `f - depth`, so
        // this is the oldest submission and bounds the frames in flight.
        let slot = (self.frame_index % self.submissions.depth() as u64) as usize;
        self.submissions.wait(gpu, slot)?;
        self.frame_slot = slot;

        if extent != self.extent {
            // Resizing destroys textures every in-flight frame may still read,
            // so this is the one place that waits for the whole ring.
            self.submissions.wait_all(gpu)?;
            let render_extent = render_extent_for(extent);
            let resources = gpu.resources();
            let textures = FrameTextures::create(
                resources,
                render_extent,
                SHADOW_MAP_SIZE,
                self.swapchain_format,
            )?;
            let old = std::mem::replace(&mut self.textures, textures);
            old.destroy(resources);
            self.recreate_readback(resources, render_extent);
            self.extent = extent;
            self.render_extent = render_extent;
        }
        if let Some(format) = gpu.swapchain().map(|swapchain| swapchain.format()) {
            self.swapchain_format = format;
        }

        let (view_projection, camera_position) = resolve_view(scene, view, editor_camera)?;
        let light_data = light::extract(scene);
        let shadow_matrix = shadow_view_projection(&light_data, camera_position);

        // A frame starts by draining what a producer queued since the last one,
        // so `declare_frame` sees the passes this frame will actually draw.
        if let Some(overlay) = &self.overlay {
            overlay.begin();
            overlay.edit();
        }

        // declare the frame. The ring slot this frame writes is chosen before the
        // graph names it, because the readback pass's target has to be in the
        // frame's slot table.
        let readback_target = self.readback.as_ref().and_then(ReadbackRing::target);
        let (targets, ssao_raw, readback_slot) = self.declare_frame(readback_target.is_some());
        self.targets = Some(targets);

        let compiled = self
            .graph
            .compile()
            .map_err(|errors| Error::InvalidDescriptor(describe(&errors)))?;

        // prepare what the recorders read
        let resources = gpu.resources();

        // The overlay's content is laid out in pixels, so it gets its own
        // screen projection instead of the scene camera's.
        if let Some(overlay) = &self.overlay {
            overlay.prepare(
                resources,
                passes::overlay::screen_projection(
                    self.render_extent.width,
                    self.render_extent.height,
                ),
            )?;
        }

        let mut slots = SlotTable::default();
        slots.set_texture(targets.scene_color, self.textures.color);
        slots.set_texture(targets.scene_depth, self.textures.depth);
        slots.set_texture(targets.shadow, self.textures.shadow);
        slots.set_texture(targets.ssao, self.textures.ssao);
        slots.set_texture(ssao_raw, self.textures.ssao_raw);
        // With a surface, the executor resolves the backbuffer slot from the
        // acquired swapchain image. Headless, nothing can, so bind the offscreen
        // target here.
        if !self.has_surface {
            slots.set_texture(targets.backbuffer, self.textures.offscreen);
        }
        // The readback pass writes this frame's ring buffer; the graph named it,
        // the slot table turns the name into the handle.
        if let (Some(buffer), Some(slot)) = (readback_target, readback_slot) {
            slots.set_buffer(slot, buffer);
        }

        let (light_buffer, light_count) = self.lights.ensure(resources, &light_data)?;
        let frame_uniforms = uniforms::FrameUniforms::new(
            view_projection.to_cols_array_2d(),
            shadow_matrix.to_cols_array_2d(),
            camera_position.to_array(),
            light_count,
            true,
        );
        let frame_group = self.frame_sets.get_mut(slot).rebuild(
            resources,
            &passes::pipeline::frame_layout(),
            &frame_uniforms,
            &[(1, light_buffer, 0)],
            &[(2, self.textures.shadow_view)],
            &[(3, self.textures.sampler)],
        )?;

        let draws = self.prepare_draws(resources, scene, camera_position)?;
        let palette_group = self.palette_sets.get_mut(slot).rebuild(
            resources,
            &passes::pipeline::palette_layout(),
            &self.frame_palette,
        )?;
        let recorder = self.build_recorders(
            resources,
            &compiled,
            shadow_matrix,
            view_projection,
            &frame_group,
            &palette_group,
            draws,
        )?;

        // One trait object: the backend owns the command lists and may record the
        // passes on worker threads.
        let recorder: Arc<dyn PassRecorder> = recorder;
        let submission = gpu.record_frame(&compiled, &slots, recorder)?;

        if self.gpu_timing {
            // Timestamps are readable only after the frame that wrote them has
            // retired, so profiling trades this wait for the per-pass split.
            let collected = gpu
                .wait_for_submission(submission.clone())
                .and_then(|()| gpu.collect_gpu_timings());
            match collected {
                Ok(()) => {
                    self.gpu_passes.clear();
                    self.gpu_passes.extend_from_slice(gpu.gpu_timings());
                    self.gpu_frame_ms = gpu.gpu_frame_ms();
                }
                Err(error) => {
                    self.gpu_passes.clear();
                    self.gpu_frame_ms = None;
                    if !self.gpu_timing_warned {
                        log::warn!("bifrost: GPU timings unavailable ({error})");
                        self.gpu_timing_warned = true;
                    }
                }
            }
        }

        // The ring now owns this slot until the wait at the next lap. The
        // readback ring gets its own clone for its older, scoped read.
        self.submissions.record(slot, submission.clone());
        self.frame_index += 1;
        if let Some(ring) = self.readback.as_mut() {
            ring.record_submission(submission);
        }

        // The frame just submitted wrote `readback_target`; this reads the ring
        // slot `READBACK_LAG` submissions behind it, so the map does not wait on
        // this frame's own copy.
        self.capture_readback(gpu);

        Ok(())
    }

    /// Reads the ring slot written [`passes::READBACK_LAG`] submissions ago into
    /// `frame_image`.
    ///
    /// Does nothing without a ring: a surface already has the frame. During the
    /// ring's first `READBACK_LAG` frames there is no retired slot to read, so
    /// the frame stays on the GPU until the ring has filled.
    fn capture_readback(&mut self, gpu: &mut dyn Gpu) {
        let Some(ring) = self.readback.as_mut() else {
            return;
        };
        let extent = self.render_extent;
        let packed = extent.width as usize * extent.height as usize * 4;

        if let Some((buffer, frame)) = ring.readable() {
            // Wait only for the submission that wrote this slot, not for the one
            // just issued: that is the whole overlap.
            let read = match ring.submission(frame) {
                Some(submission) => gpu.resources().read_buffer_after(&buffer, submission),
                None => gpu.resources().read_buffer(&buffer),
            };
            match read {
                Ok(bgra) if bgra.len() >= packed => {
                    // Packed, because the render width is a multiple of 64
                    // texels: the copy's pitch equals the packed row.
                    self.frame_image = Some(FrameImage {
                        width: extent.width,
                        height: extent.height,
                        bgra: bgra[..packed].to_vec(),
                    });
                }
                Ok(bgra) => {
                    if !self.readback_warned {
                        log::warn!(
                            "bifrost: readback returned {} bytes for a {}x{} frame",
                            bgra.len(),
                            extent.width,
                            extent.height
                        );
                        self.readback_warned = true;
                    }
                }
                Err(error) => {
                    if !self.readback_warned {
                        log::warn!(
                            "bifrost: readback failed ({error}); the frame stays on the GPU"
                        );
                        self.readback_warned = true;
                    }
                }
            }
        }

        // One step per frame that used a target, whether or not it was readable.
        ring.advance();
    }

    /// Declares this frame's resources and passes.
    ///
    /// `readback` is whether this frame has a ring slot to copy into, which the
    /// readback pass is declared against.
    fn declare_frame(&mut self, readback: bool) -> (FrameTargets, TextureId, Option<BufferId>) {
        let graph = &mut self.graph;
        graph.reset();

        // Imports, not transients: these textures outlive the frame, and imports
        // are the graph's culling roots (`compiler::cull`). `Undefined` as the
        // entry state makes the first transition free.
        let scene_color =
            graph.import_texture("scene.color", ImageState::Undefined, ImageState::ShaderRead);
        let scene_depth = graph.import_texture(
            "scene.depth",
            ImageState::Undefined,
            ImageState::DepthAttachment,
        );
        let shadow =
            graph.import_texture("shadow.map", ImageState::Undefined, ImageState::ShaderRead);
        let ssao = graph.import_texture("ssao.ao", ImageState::Undefined, ImageState::ShaderRead);
        let ssao_raw =
            graph.import_texture("ssao.raw", ImageState::Undefined, ImageState::ShaderRead);
        // Present only when there is a swapchain. Headless, the composition ends
        // at a colour state, so the backend sees no backbuffer slot to acquire
        // and draws to the offscreen target instead.
        let backbuffer_state = if self.has_surface {
            ImageState::Present
        } else {
            ImageState::ColorAttachment
        };
        let backbuffer =
            graph.import_texture("backbuffer", ImageState::Undefined, backbuffer_state);

        // The shadow map: depth only, cleared to the far plane.
        graph
            .pass(passes::PASS_SHADOW)
            .kind(chassis::graph::PassType::Graphics)
            .depth(shadow, 1.0);

        // Opaque geometry, then the blended draws (`GeometryPass`). The clear
        // belongs to this pass: a separate clear pass would be culled, because
        // `compiler::cull` keeps a pass only when a live pass reads what it
        // produced, and a `LoadOp::Load` is not a read there.
        graph
            .pass(passes::PASS_OPAQUE)
            .color(scene_color, CLEAR_COLOR)
            .depth(scene_depth, 1.0)
            .read_texture(shadow);

        // Ambient occlusion at half rate, from the depth the opaque pass wrote,
        // then a depth-aware upsample into the texture composition reads.
        graph
            .pass(passes::PASS_SSAO)
            .color(ssao_raw, [1.0, 1.0, 1.0, 1.0])
            .read_texture(scene_depth);
        graph
            .pass(passes::PASS_SSAO_UPSAMPLE)
            .color(ssao, [1.0, 1.0, 1.0, 1.0])
            .read_texture(ssao_raw)
            .read_texture(scene_depth);

        // The editor's grid and gizmos, blended, after the scene. The scene depth
        // is loaded so the grid is occluded by geometry instead of drawing over it.
        graph
            .pass(passes::PASS_EDITOR)
            .color_load(scene_color)
            .depth_load(scene_depth);

        // The 2D and text slice's passes, which draw into the scene colour
        // after the 3D scene does. Declared here, before the composition, so
        // the composition is still the last writer of the backbuffer.
        let targets = FrameTargets {
            scene_color,
            scene_depth,
            shadow,
            ssao,
            backbuffer,
        };
        if let Some(overlay) = &self.overlay {
            overlay.declare(graph, &targets);
        }

        // The composition: the scene colour and the occlusion into the
        // backbuffer, whose slot the executor resolves.
        graph
            .pass(passes::PASS_COMPOSITE)
            .color(backbuffer, [0.0, 0.0, 0.0, 1.0])
            .read_texture(scene_color)
            .read_texture(ssao);

        // Copy the composed image out for the CPU, when there is no surface to
        // present it to. Declared after the composition so it reads the final
        // image, and as a transfer pass so the graph emits the image-to-buffer
        // barrier that `vkCmdCopyImageToBuffer` needs.
        let readback = readback.then(|| {
            let buffer = graph.import_buffer("readback");
            graph
                .pass(passes::PASS_READBACK)
                .kind(PassType::Transfer)
                .read_texture(backbuffer)
                .write_buffer(buffer);
            buffer
        });

        (targets, ssao_raw, readback)
    }

    /// Turns the scene's meshes into draws, and uploads their uniforms.
    fn prepare_draws(
        &mut self,
        resources: &mut dyn Resources,
        scene: &meats::Scene,
        camera_position: Vec3,
    ) -> Result<Vec<PreparedDraw>, Error> {
        let slot = self.frame_slot;
        struct Pending {
            mesh: GpuMesh,
            material: Option<MaterialBinding>,
            variant: MaterialVariant,
            distance: f32,
            block: usize,
        }

        let mut blocks: Vec<Vec<u8>> = Vec::new();
        let mut pending: Vec<Pending> = Vec::new();
        let mut palette: Vec<[[f32; 4]; 4]> = Vec::new();

        for sample in scene.meshes() {
            let Some(mesh) = self
                .meshes
                .get(&MeshKey::new(sample.mesh, self.mesh_layout))
            else {
                // Not uploaded: `ensure_mesh` has not seen it.
                log::trace!("bifrost: mesh {:?} has no GPU buffers yet", sample.mesh);
                continue;
            };

            let material = sample
                .material
                .and_then(|id| self.material_assets.get(&id))
                .copied()
                .unwrap_or_default();
            let variant =
                MaterialVariant::of(&material, material.alpha == material::AlphaMode::Blend);

            let material_binding = self.material_binding(resources, &material, variant)?;

            let (palette_first, palette_count) = match self.skinned.get(&sample.entity) {
                Some(draw) => {
                    let first = palette.len() as u32;
                    palette.extend_from_slice(&draw.matrices);
                    (first, draw.matrices.len() as u32)
                }
                None => (0, 0),
            };
            let uniforms = uniforms::DrawUniforms::new(
                sample.transform.to_cols_array_2d(),
                sample.tint.to_array(),
                entity_id(sample.entity),
            )
            .with_palette(palette_first, palette_count);
            let block = blocks.len();
            blocks.push(bytemuck::bytes_of(&uniforms).to_vec());

            pending.push(Pending {
                mesh,
                material: material_binding,
                variant,
                distance: (sample.transform.w_axis.truncate() - camera_position).length(),
                block,
            });
        }

        self.draw_arenas.get_mut(slot).begin(
            resources,
            &passes::pipeline::draw_layout(),
            &blocks,
        )?;
        self.frame_palette = palette;

        // The material slot is a stable per-material id, which makes it usable
        // as a sort key; the mesh is keyed by its vertex handle.
        let material_slot =
            |draw: &Pending| draw.material.map(|entry| entry.slot).unwrap_or(u32::MAX);
        let mesh_id = |draw: &Pending| (draw.mesh.vertex.index(), draw.mesh.vertex.generation());

        // Opaque first, then the blended draws. Within the opaques the order is
        // chosen so consecutive draws share pipeline, material and mesh, which
        // lets `GeometryPass` bind each once. Blended draws keep their
        // back-to-front order, which is what makes the blend correct; batching
        // them would trade a correct image for a few binds.
        pending.sort_by(|a, b| {
            a.variant.blends().cmp(&b.variant.blends()).then_with(|| {
                if a.variant.blends() {
                    b.distance.total_cmp(&a.distance)
                } else {
                    a.variant
                        .cmp(&b.variant)
                        .then_with(|| material_slot(a).cmp(&material_slot(b)))
                        .then_with(|| mesh_id(a).cmp(&mesh_id(b)))
                        .then_with(|| b.distance.total_cmp(&a.distance))
                }
            })
        });

        Ok(pending
            .into_iter()
            .filter_map(|draw| {
                Some(PreparedDraw {
                    mesh: draw.mesh,
                    uniforms: self.draw_arenas.get(slot).group(draw.block)?,
                    material: draw.material.map(|binding| binding.group),
                    variant: draw.variant,
                })
            })
            .collect())
    }

    /// One material's bind group, created once per distinct [`MaterialKey`].
    fn material_binding(
        &mut self,
        resources: &mut dyn Resources,
        material: &Material,
        variant: MaterialVariant,
    ) -> Result<Option<MaterialBinding>, Error> {
        let white = self.textures_by_asset.white(resources)?;
        let neutral = self.textures_by_asset.neutral_normal(resources)?;

        let mut resolved: [CachedTexture; material::SLOT_COUNT] = [white; material::SLOT_COUNT];
        let mut images: Vec<(u32, ImageView)> = Vec::with_capacity(material::SLOT_COUNT);

        for (slot, cached_slot) in resolved.iter_mut().enumerate() {
            let cached = material
                .texture(slot)
                .and_then(|id| self.textures_by_asset.get(id))
                .unwrap_or(if slot == material::SLOT_NORMAL {
                    neutral
                } else {
                    white
                });
            *cached_slot = cached;
            images.push((2 + slot as u32, cached.view));
        }

        let handles = MaterialTextures {
            albedo: Some(resolved[material::SLOT_ALBEDO].texture),
            metallic_roughness: Some(resolved[material::SLOT_METALLIC_ROUGHNESS].texture),
            normal: Some(resolved[material::SLOT_NORMAL].texture),
            occlusion: Some(resolved[material::SLOT_OCCLUSION].texture),
            emissive: Some(resolved[material::SLOT_EMISSIVE].texture),
        };

        let key = MaterialKey::of(material, handles, variant);
        let layout = passes::pipeline::material_layout();
        let samplers = [(1u32, self.textures.sampler)];
        Ok(Some(self.materials.get_or_insert(
            resources, key, material, &images, &samplers, &layout,
        )?))
    }

    /// Builds every pass's recorder for this frame.
    #[allow(clippy::too_many_arguments)]
    fn build_recorders(
        &mut self,
        resources: &mut dyn Resources,
        compiled: &CompiledGraph,
        shadow_matrix: Mat4,
        view_projection: Mat4,
        frame_group: &BindGroup,
        palette_group: &BindGroup,
        draws: Vec<PreparedDraw>,
    ) -> Result<Arc<FrameRecorder>, Error> {
        let slot = self.frame_slot;
        let samples = chassis::SampleCount::X1;
        let color = TextureFormat::Rgba8Srgb;
        let depth = TextureFormat::Depth32Float;
        let mut recorder = FrameRecorder::new();

        if compiled.find(passes::PASS_SHADOW).is_some() {
            let pipeline = self.pipelines.get_or_create(
                resources,
                PipelineKey::shadow(depth, samples),
                &self.shaders,
            )?;
            let uniforms = uniforms::ShadowUniforms::new(shadow_matrix.to_cols_array_2d());
            let group = self.shadow_sets.get_mut(slot).rebuild(
                resources,
                &passes::pipeline::shadow_layout(),
                &uniforms,
                &[],
                &[],
                &[],
            )?;
            // Blended draws are left out: a transparent surface in a shadow map
            // writes the depth of its whole quad, not of the part the blend lets
            // through.
            let shadow_draws: Vec<PreparedDraw> = draws
                .iter()
                .copied()
                .filter(|draw| !draw.transparent())
                .collect();
            recorder.shadow = Some(ShadowPass::new(
                pipeline,
                group,
                *palette_group,
                shadow_draws,
            ));
        }

        if compiled.find(passes::PASS_OPAQUE).is_some() {
            let pipelines = MaterialPipelines {
                opaque: self.pipelines.get_or_create(
                    resources,
                    PipelineKey::model(ShaderVariant::Model, color, depth, samples),
                    &self.shaders,
                )?,
                opaque_double_sided: self.pipelines.get_or_create(
                    resources,
                    PipelineKey::model(ShaderVariant::ModelDoubleSided, color, depth, samples),
                    &self.shaders,
                )?,
                transparent: self.pipelines.get_or_create(
                    resources,
                    PipelineKey::model(ShaderVariant::ModelTransparent, color, depth, samples),
                    &self.shaders,
                )?,
                transparent_double_sided: self.pipelines.get_or_create(
                    resources,
                    PipelineKey::model(
                        ShaderVariant::ModelTransparentDoubleSided,
                        color,
                        depth,
                        samples,
                    ),
                    &self.shaders,
                )?,
            };
            recorder.geometry = Some(GeometryPass::new(
                pipelines,
                *frame_group,
                *palette_group,
                draws,
            ));
        }

        if compiled.find(passes::PASS_SSAO).is_some() {
            let pipeline = self.pipelines.get_or_create(
                resources,
                PipelineKey::fullscreen(ShaderVariant::Ssao, TextureFormat::R8Unorm, samples),
                &self.shaders,
            )?;
            let uniforms = SsaoUniforms::new(
                &passes::ssao::fixed_kernel(0x9E37_79B9_7F4A_7C15),
                view_projection.to_cols_array_2d(),
                view_projection.inverse().to_cols_array_2d(),
                passes::ssao::DEFAULT_RADIUS,
                passes::ssao::DEFAULT_BIAS,
                passes::ssao::DEFAULT_POWER,
            );
            let group = self.ssao_sets.get_mut(slot).rebuild(
                resources,
                &passes::pipeline::ssao_layout(),
                &uniforms,
                &[],
                &[(1, self.textures.depth_view)],
                &[(2, self.textures.sampler)],
            )?;
            recorder.ssao = Some(SsaoPass::new(pipeline, group));

            let upsample = self.pipelines.get_or_create(
                resources,
                PipelineKey::fullscreen(
                    ShaderVariant::SsaoUpsample,
                    TextureFormat::R8Unorm,
                    samples,
                ),
                &self.shaders,
            )?;
            let upsample_group = self.ssao_upsample_sets.get_mut(slot).rebuild(
                resources,
                &passes::pipeline::ssao_upsample_layout(),
                &uniforms,
                &[],
                &[
                    (1, self.textures.depth_view),
                    (3, self.textures.ssao_raw_view),
                ],
                &[(2, self.textures.sampler)],
            )?;
            recorder.ssao_upsample = Some(SsaoPass::new(upsample, upsample_group));
        }

        if compiled.find(passes::PASS_EDITOR).is_some() {
            let grid = if self.show_grid {
                let pipeline = self.pipelines.get_or_create(
                    resources,
                    PipelineKey::grid(color, depth, samples),
                    &self.shaders,
                )?;
                let uniforms = GridUniforms::new(view_projection.to_cols_array_2d());
                let group = self.grid_sets.get_mut(slot).rebuild(
                    resources,
                    &passes::pipeline::grid_layout(),
                    &uniforms,
                    &[],
                    &[],
                    &[],
                )?;
                Some(GridPass::new(pipeline, group))
            } else {
                None
            };

            let lines = match self.line_batch {
                Some(batch) => {
                    let pipeline = self.pipelines.get_or_create(
                        resources,
                        PipelineKey::lines(color, depth, samples),
                        &self.shaders,
                    )?;
                    let uniforms = LineUniforms::new(
                        view_projection.to_cols_array_2d(),
                        Mat4::IDENTITY.to_cols_array_2d(),
                    );
                    let group = self.line_sets.get_mut(slot).rebuild(
                        resources,
                        &passes::pipeline::line_layout(),
                        &uniforms,
                        &[],
                        &[],
                        &[],
                    )?;
                    Some(LinePass::new(pipeline, group, batch))
                }
                None => None,
            };

            let debug = match self.debug_batch {
                Some(batch) => {
                    let pipeline = self.pipelines.get_or_create(
                        resources,
                        PipelineKey::lines(color, depth, samples),
                        &self.shaders,
                    )?;
                    let uniforms = LineUniforms::new(
                        view_projection.to_cols_array_2d(),
                        Mat4::IDENTITY.to_cols_array_2d(),
                    );
                    let group = self.debug_sets.get_mut(slot).rebuild(
                        resources,
                        &passes::pipeline::line_layout(),
                        &uniforms,
                        &[],
                        &[],
                        &[],
                    )?;
                    Some(LinePass::new(pipeline, group, batch))
                }
                None => None,
            };

            if grid.is_some() || lines.is_some() || debug.is_some() {
                recorder.editor = Some(EditorPass::new(grid, lines, debug));
            }
        }

        if compiled.find(passes::PASS_COMPOSITE).is_some() {
            let pipeline = self.pipelines.get_or_create(
                resources,
                PipelineKey::fullscreen(ShaderVariant::Composite, self.swapchain_format, samples),
                &self.shaders,
            )?;
            let group = self.image_sets.get_mut(slot).rebuild(
                resources,
                &passes::pipeline::composite_layout(),
                &[(0, self.textures.color_view), (1, self.textures.ssao_view)],
                &[(2, self.textures.sampler)],
            )?;
            recorder.composite = Some(CompositePass::new(pipeline, group));
        }

        if compiled.find(passes::PASS_READBACK).is_some() {
            // The same pure lookup the slot table used: no advance between them.
            if let Some(target) = self.readback.as_ref().and_then(ReadbackRing::target) {
                recorder.readback = Some(ReadbackPass {
                    source: self.textures.offscreen,
                    target,
                });
            }
        }

        recorder.overlay = self.overlay.clone();
        Ok(Arc::new(recorder))
    }

    /// Drops every GPU resource this renderer owns.
    pub fn destroy(&mut self, gpu: &mut dyn Gpu) {
        // Freeing a resource the GPU may still read is invalid usage on both
        // backends, so the ring's newest submissions are retired first.
        if let Err(error) = self.submissions.wait_all(gpu) {
            log::warn!("bifrost: destroying with submissions still in flight: {error}");
        }
        let resources = gpu.resources();
        if let Some(overlay) = &self.overlay {
            overlay.destroy(resources);
        }
        self.meshes.clear(resources);
        self.materials.clear(resources);
        self.textures_by_asset.clear(resources);
        self.pipelines.clear(resources);
        self.lights.destroy(resources);
        for set in self.frame_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.shadow_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.ssao_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.ssao_upsample_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.grid_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.line_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.debug_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.image_sets.iter_mut() {
            set.destroy(resources);
        }
        for set in self.palette_sets.iter_mut() {
            set.destroy(resources);
        }
        for arena in self.draw_arenas.iter_mut() {
            arena.destroy(resources);
        }
        if let Some(batch) = self.line_batch.take() {
            resources.destroy_buffer(batch.vertex);
            resources.destroy_buffer(batch.index);
        }
        if let Some(batch) = self.debug_batch.take() {
            resources.destroy_buffer(batch.vertex);
            resources.destroy_buffer(batch.index);
        }
        if let Some(ring) = self.readback.take() {
            ring.destroy(resources);
        }
        self.shaders.destroy(resources);
        let textures = std::mem::replace(&mut self.textures, FrameTextures::null());
        textures.destroy(resources);
    }
}

/// Replaces one line batch's buffers, releasing the previous pair first.
fn upload_lines(
    resources: &mut dyn Resources,
    batch: &mut Option<LineBatch>,
    vertices: &[passes::LineVertex],
    indices: &[u32],
) -> Result<(), Error> {
    if let Some(old) = batch.take() {
        resources.destroy_buffer(old.vertex);
        resources.destroy_buffer(old.index);
    }
    if vertices.is_empty() || indices.is_empty() {
        return Ok(());
    }

    let (vertex_bytes, index_bytes) = passes::lines::pack(vertices, indices);
    let vertex = resources.create_buffer(
        BufferDesc::new(vertex_bytes.len() as u64, BufferUsage::VERTEX),
        Some(&vertex_bytes),
    )?;
    let index = resources.create_buffer(
        BufferDesc::new(index_bytes.len() as u64, BufferUsage::INDEX),
        Some(&index_bytes),
    )?;
    *batch = Some(LineBatch {
        vertex,
        index,
        index_count: indices.len() as u32,
    });
    Ok(())
}

/// Packs an animator's current bone matrices for the GPU.
fn palette_of(animator: &Animator) -> Vec<[[f32; 4]; 4]> {
    animator
        .final_bone_matrices()
        .iter()
        .map(|matrix| matrix.to_cols_array_2d())
        .collect()
}

/// The extent the frame's targets are created at.
///
/// The width is rounded up to a multiple of 64 texels -- the copy alignment of
/// [`chassis::COPY_BYTES_PER_ROW_ALIGNMENT`] in four-byte texels -- so the
/// readback copy's row pitch equals `width * 4` and there is no padding to
/// strip. The height is unchanged: rows, not columns, are padded.
fn render_extent_for(extent: Extent2d) -> Extent2d {
    Extent2d {
        width: chassis::copy_row_pitch(extent.width.max(1), 4) / 4,
        height: extent.height.max(1),
    }
}

/// The view-projection and the camera's world position for this frame.
fn resolve_view(
    scene: &meats::Scene,
    view: ViewSource,
    editor_camera: Option<&EditorCamera>,
) -> Result<(Mat4, Vec3), Error> {
    match view {
        ViewSource::Editor => {
            let camera = editor_camera.ok_or_else(|| {
                Error::InvalidDescriptor(
                    "ViewSource::Editor with no editor camera passed to render".to_owned(),
                )
            })?;
            Ok((camera.view_projection(), camera.position()))
        }
        ViewSource::Scene(entity) => {
            let transform = scene.world_transform(entity);
            let reference = scene.get(entity).ok_or_else(|| {
                Error::InvalidDescriptor(format!("view entity {entity:?} is not in the scene"))
            })?;
            let camera = reference.get::<meats::Camera>().ok_or_else(|| {
                Error::InvalidDescriptor(format!("view entity {entity:?} has no Camera component"))
            })?;
            // `projection * inverse(transform)`.
            Ok((
                camera.projection() * transform.inverse(),
                transform.w_axis.truncate(),
            ))
        }
    }
}

/// A directional light's shadow matrix, in the camera's neighbourhood.
///
/// Places an orthographic box on the camera and looks along the first
/// directional light. With no directional light the identity is returned, and
/// the fragment shader's out-of-range uv test then leaves every surface
/// unshadowed.
fn shadow_view_projection(lights: &[LightData], camera_position: Vec3) -> Mat4 {
    let Some(light) = lights
        .iter()
        .find(|light| light.light_type() == light::TYPE_DIRECTIONAL)
    else {
        return Mat4::IDENTITY;
    };

    let direction = Vec3::new(
        light.position_or_direction[0],
        light.position_or_direction[1],
        light.position_or_direction[2],
    );
    let direction = if direction.length_squared() > 0.0 {
        direction.normalize()
    } else {
        Vec3::NEG_Y
    };

    let eye = camera_position - direction * SHADOW_EXTENT;
    let view = glam::camera::rh::view::look_at_mat4(eye, camera_position, Vec3::Y);
    let projection = glam::camera::rh::proj::directx::orthographic(
        -SHADOW_EXTENT,
        SHADOW_EXTENT,
        -SHADOW_EXTENT,
        SHADOW_EXTENT,
        0.1,
        SHADOW_EXTENT * 4.0,
    );
    projection * view
}

/// The id an editor picks with.
///
/// `Entity::index` identifies the live entity a pick buffer names; the
/// persistent identity is the `Id` component's uuid, which a pick buffer has no
/// use for.
fn entity_id(entity: meats::Entity) -> i32 {
    entity.index() as i32
}

fn describe(errors: &[chassis::graph::types::ValidationError]) -> String {
    errors
        .iter()
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}
