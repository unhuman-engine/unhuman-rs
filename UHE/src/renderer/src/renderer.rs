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
    BufferId, CompiledGraph, ImageState, PassRecorder, PassType, RenderGraph, SlotTable,
};
use chassis::{
    BindGroup, Buffer, BufferDesc, BufferUsage, Error, Extent, Extent2d, Gpu, ImageView, Resources,
    Sampler, SamplerDesc, Texture, TextureDesc, TextureFormat, TextureUsage,
};
use glam::{Mat4, Vec3};

use crate::camera::{CameraInput, EditorCamera};
use crate::light::{self, LightBuffer, LightData};
use crate::material::{
    self, CachedTexture, Material, MaterialBinding, MaterialCache, MaterialKey, MaterialTextures,
    MaterialVariant, TextureCache,
};
use crate::mesh::{self, GpuMesh, MeshCache, MeshKey, VertexLayoutKey};
use crate::passes::{
    self, uniforms, CompositePass, DrawArena, EditorPass, FrameRecorder, FrameTargets,
    GeometryPass, GridPass, GridUniforms, ImageSet, LineBatch, LinePass, LineUniforms,
    MaterialPipelines, OverlayPasses, PipelineCache, PipelineKey, PreparedDraw, ReadbackPass,
    ShaderSet, ShaderVariant, ShadowPass, SsaoPass, SsaoUniforms,
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
            offscreen,
            sampler,
        })
    }

    fn destroy(self, resources: &mut dyn Resources) {
        resources.destroy_texture(self.color);
        resources.destroy_texture(self.depth);
        resources.destroy_texture(self.shadow);
        resources.destroy_texture(self.ssao);
        resources.destroy_texture(self.offscreen);
        resources.destroy_sampler(self.sampler);
    }
}

/// The renderer frontend. Owned by `BifrostPlugin`, never by `Engine`.
pub struct Bifrost {
    extent: Extent2d,
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

    frame_set: uniforms::BindSet,
    shadow_set: uniforms::BindSet,
    ssao_set: uniforms::BindSet,
    /// The grid's set and the gizmo lines' set: two, because each holds its own
    /// matrices and a pass that bound one buffer for both would draw one of them
    /// through the other's matrix.
    grid_set: uniforms::BindSet,
    line_set: uniforms::BindSet,
    image_set: ImageSet,
    draw_arena: DrawArena,

    mesh_layout: VertexLayoutKey,
    line_batch: Option<LineBatch>,
    show_grid: bool,

    overlay: Option<Arc<dyn OverlayPasses>>,
    frame_index: u64,
    /// The buffer the composed frame is copied into and read back from. `None`
    /// with a surface -- there is nothing to read back -- or when the device
    /// cannot make a readable buffer.
    readback: Option<Buffer>,
    /// The last frame read back, or `None` before the first one.
    frame_image: Option<FrameImage>,
    /// Readback failures are logged once, not once per frame.
    readback_warned: bool,
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
        let swapchain_format = gpu
            .swapchain()
            .map(|swapchain| swapchain.format())
            .unwrap_or(TextureFormat::Bgra8Srgb);
        let has_surface = gpu.swapchain().is_some();

        let resources = gpu.resources();
        let shaders = ShaderSet::load(resources)?;
        let textures = FrameTextures::create(resources, extent, SHADOW_MAP_SIZE, swapchain_format)?;

        let mut bifrost = Self {
            extent,
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
            frame_set: uniforms::BindSet::new(),
            shadow_set: uniforms::BindSet::new(),
            ssao_set: uniforms::BindSet::new(),
            grid_set: uniforms::BindSet::new(),
            line_set: uniforms::BindSet::new(),
            image_set: ImageSet::new(),
            draw_arena: DrawArena::new(),
            mesh_layout: VertexLayoutKey::of(&mesh::vertex_layout()),
            line_batch: None,
            show_grid: true,
            overlay: None,
            frame_index: 0,
            readback: None,
            frame_image: None,
            readback_warned: false,
        };

        // The 1x1 textures a material with no map of a kind samples, made once.
        bifrost.textures_by_asset.white(gpu.resources())?;
        bifrost.textures_by_asset.neutral_normal(gpu.resources())?;
        bifrost.recreate_readback(gpu.resources(), extent);

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

    /// Recreates the readback buffer for `extent`, or drops it when there is a
    /// surface to present to instead.
    fn recreate_readback(&mut self, resources: &mut dyn Resources, extent: Extent2d) {
        if let Some(old) = self.readback.take() {
            resources.destroy_buffer(old);
        }
        self.frame_image = None;
        if self.has_surface {
            return;
        }

        // `MAP_READ` is what wgpu needs to map it; Vulkan ignores the flag and
        // makes the buffer host-visible because it is a transfer destination.
        // `TRANSFER_DST` is both.
        //
        // Sized by the copy's *row pitch*, not by `width * 4`: the copy pads every
        // row to the alignment, and a width that is not a multiple of 64 texels
        // gets a longer buffer than the image it carries. `capture_readback`
        // strips the padding.
        let pitch = chassis::copy_row_pitch(extent.width.max(1), 4);
        let size = u64::from(pitch) * u64::from(extent.height.max(1));
        match resources.create_buffer(
            BufferDesc::new(size, BufferUsage::TRANSFER_DST | BufferUsage::MAP_READ),
            None,
        ) {
            Ok(buffer) => self.readback = Some(buffer),
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
        let resources = gpu.resources();
        if let Some(batch) = self.line_batch.take() {
            resources.destroy_buffer(batch.vertex);
            resources.destroy_buffer(batch.index);
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
        self.line_batch = Some(LineBatch {
            vertex,
            index,
            index_count: indices.len() as u32,
        });
        Ok(())
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
        if extent != self.extent {
            let resources = gpu.resources();
            let textures =
                FrameTextures::create(resources, extent, SHADOW_MAP_SIZE, self.swapchain_format)?;
            let old = std::mem::replace(&mut self.textures, textures);
            old.destroy(resources);
            self.recreate_readback(resources, extent);
            self.extent = extent;
        }
        if let Some(format) = gpu.swapchain().map(|swapchain| swapchain.format()) {
            self.swapchain_format = format;
        }

        let (view_projection, camera_position) = resolve_view(scene, view, editor_camera)?;
        let light_data = light::extract(scene);
        let shadow_matrix = shadow_view_projection(&light_data, camera_position);

        // declare the frame
        let (targets, readback_slot) = self.declare_frame();
        self.targets = Some(targets);

        let compiled = self
            .graph
            .compile()
            .map_err(|errors| Error::InvalidDescriptor(describe(&errors)))?;

        // prepare what the recorders read
        let resources = gpu.resources();

        let mut slots = SlotTable::default();
        slots.set_texture(targets.scene_color, self.textures.color);
        slots.set_texture(targets.scene_depth, self.textures.depth);
        slots.set_texture(targets.shadow, self.textures.shadow);
        slots.set_texture(targets.ssao, self.textures.ssao);
        // With a surface, the executor resolves the backbuffer slot from the
        // acquired swapchain image. Headless, nothing can, so bind the offscreen
        // target here.
        if !self.has_surface {
            slots.set_texture(targets.backbuffer, self.textures.offscreen);
        }
        // The readback pass writes this buffer; the graph named it, the slot
        // table turns the name into the handle.
        if let (Some(buffer), Some(slot)) = (self.readback, readback_slot) {
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
        let frame_group = self.frame_set.rebuild(
            resources,
            &passes::pipeline::frame_layout(),
            &frame_uniforms,
            &[(1, light_buffer, 0)],
            &[(2, self.textures.shadow_view)],
            &[(3, self.textures.sampler)],
        )?;

        let draws = self.prepare_draws(resources, scene, camera_position)?;
        let recorder = self.build_recorders(
            resources,
            &compiled,
            shadow_matrix,
            view_projection,
            &frame_group,
            draws,
        )?;

        // run the frame
        self.frame_index += 1;
        // One trait object: the backend owns the command lists and may record the
        // passes on worker threads.
        let recorder: Arc<dyn PassRecorder> = recorder;
        gpu.record_frame(&compiled, &slots, recorder)?;

        // `record_frame` submitted and waited, so the copy into the readback
        // buffer has retired and it is safe to map.
        self.capture_readback(gpu);

        Ok(())
    }

    /// Reads this frame's composition back into `frame_image`.
    ///
    /// Does nothing without a readback buffer: a surface already has the frame.
    fn capture_readback(&mut self, gpu: &mut dyn Gpu) {
        let Some(buffer) = self.readback else {
            return;
        };
        let extent = self.extent;
        let pitch = chassis::copy_row_pitch(extent.width, 4) as usize;
        let packed = extent.width as usize * 4;
        let height = extent.height as usize;

        match gpu.resources().read_buffer(&buffer) {
            Ok(bgra) if bgra.len() >= pitch * height => {
                self.frame_image = Some(FrameImage {
                    width: extent.width,
                    height: extent.height,
                    bgra: strip_copy_padding(&bgra, pitch, packed, height),
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
                    log::warn!("bifrost: readback failed ({error}); the frame stays on the GPU");
                    self.readback_warned = true;
                }
            }
        }
    }

    /// Declares this frame's resources and passes.
    fn declare_frame(&mut self) -> (FrameTargets, Option<BufferId>) {
        let readback_enabled = self.readback.is_some();
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

        // Ambient occlusion, from the depth the opaque pass wrote.
        graph
            .pass(passes::PASS_SSAO)
            .color(ssao, [1.0, 1.0, 1.0, 1.0])
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
        let readback = readback_enabled.then(|| {
            let buffer = graph.import_buffer("readback");
            graph
                .pass(passes::PASS_READBACK)
                .kind(PassType::Transfer)
                .read_texture(backbuffer)
                .write_buffer(buffer);
            buffer
        });

        (targets, readback)
    }

    /// Turns the scene's meshes into draws, and uploads their uniforms.
    fn prepare_draws(
        &mut self,
        resources: &mut dyn Resources,
        scene: &meats::Scene,
        camera_position: Vec3,
    ) -> Result<Vec<PreparedDraw>, Error> {
        struct Pending {
            mesh: GpuMesh,
            material: Option<MaterialBinding>,
            variant: MaterialVariant,
            distance: f32,
            block: usize,
        }

        let mut blocks: Vec<Vec<u8>> = Vec::new();
        let mut pending: Vec<Pending> = Vec::new();

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

            let uniforms = uniforms::DrawUniforms::new(
                sample.transform.to_cols_array_2d(),
                sample.tint.to_array(),
                entity_id(sample.entity),
            );
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

        self.draw_arena
            .begin(resources, &passes::pipeline::draw_layout(), &blocks)?;

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
                    uniforms: self.draw_arena.group(draw.block)?,
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
    fn build_recorders(
        &mut self,
        resources: &mut dyn Resources,
        compiled: &CompiledGraph,
        shadow_matrix: Mat4,
        view_projection: Mat4,
        frame_group: &BindGroup,
        draws: Vec<PreparedDraw>,
    ) -> Result<Arc<FrameRecorder>, Error> {
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
            let group = self.shadow_set.rebuild(
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
            recorder.shadow = Some(ShadowPass::new(pipeline, group, shadow_draws));
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
            recorder.geometry = Some(GeometryPass::new(pipelines, *frame_group, draws));
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
            let group = self.ssao_set.rebuild(
                resources,
                &passes::pipeline::ssao_layout(),
                &uniforms,
                &[],
                &[(1, self.textures.depth_view)],
                &[(2, self.textures.sampler)],
            )?;
            recorder.ssao = Some(SsaoPass::new(pipeline, group));
        }

        if compiled.find(passes::PASS_EDITOR).is_some() {
            let grid = if self.show_grid {
                let pipeline = self.pipelines.get_or_create(
                    resources,
                    PipelineKey::grid(color, depth, samples),
                    &self.shaders,
                )?;
                let uniforms = GridUniforms::new(view_projection.to_cols_array_2d());
                let group = self.grid_set.rebuild(
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
                    let group = self.line_set.rebuild(
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

            if grid.is_some() || lines.is_some() {
                recorder.editor = Some(EditorPass::new(grid, lines));
            }
        }

        if compiled.find(passes::PASS_COMPOSITE).is_some() {
            let pipeline = self.pipelines.get_or_create(
                resources,
                PipelineKey::fullscreen(ShaderVariant::Composite, self.swapchain_format, samples),
                &self.shaders,
            )?;
            let group = self.image_set.rebuild(
                resources,
                &passes::pipeline::composite_layout(),
                &[(0, self.textures.color_view), (1, self.textures.ssao_view)],
                &[(2, self.textures.sampler)],
            )?;
            recorder.composite = Some(CompositePass::new(pipeline, group));
        }

        if compiled.find(passes::PASS_READBACK).is_some() {
            if let Some(target) = self.readback {
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
        let resources = gpu.resources();
        self.meshes.clear(resources);
        self.materials.clear(resources);
        self.textures_by_asset.clear(resources);
        self.pipelines.clear(resources);
        self.lights.destroy(resources);
        self.frame_set.destroy(resources);
        self.shadow_set.destroy(resources);
        self.ssao_set.destroy(resources);
        self.grid_set.destroy(resources);
        self.line_set.destroy(resources);
        self.image_set.destroy(resources);
        self.draw_arena.destroy(resources);
        if let Some(batch) = self.line_batch.take() {
            resources.destroy_buffer(batch.vertex);
            resources.destroy_buffer(batch.index);
        }
        if let Some(buffer) = self.readback.take() {
            resources.destroy_buffer(buffer);
        }
        self.shaders.destroy(resources);
        let textures = std::mem::replace(&mut self.textures, FrameTextures::null());
        textures.destroy(resources);
    }
}

/// Copies the rows of a padded readback into a tightly packed image.
///
/// A copy's rows are padded to [`chassis::COPY_BYTES_PER_ROW_ALIGNMENT`] and
/// [`FrameImage`] promises a packed buffer -- `width * height * 4` bytes, which
/// is what a caller turns into an image -- so the padding comes off here, once,
/// rather than in every consumer.
fn strip_copy_padding(bytes: &[u8], pitch: usize, packed: usize, height: usize) -> Vec<u8> {
    let mut image = Vec::with_capacity(packed * height);
    for row in bytes.chunks_exact(pitch).take(height) {
        image.extend_from_slice(&row[..packed]);
    }
    image
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
