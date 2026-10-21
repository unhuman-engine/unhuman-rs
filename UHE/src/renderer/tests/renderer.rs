//! The renderer's decisions that need no device: the frame's graph, the cache
//! keys, the caches themselves through a recorded fake RHI, the shaders, and the
//! one call a frame makes into a backend.
//!
//! There is deliberately **no `#[ignore]`d GPU test here**. `bifrost` never
//! depends on a backend crate, so it cannot construct a real device; a
//! device-facing test belongs to the backend suites and to
//! `UHE/src/engine/tests/backend_parity.rs`, which exist for exactly that. What a
//! frame asks of a backend is `Gpu::record_frame`, whose arguments are pure data,
//! so that seam is tested here against a fake `Gpu` that records the call.

use std::sync::Arc;

use bifrost::material::{
    self, Material, MaterialCache, MaterialKey, MaterialTextures, MaterialVariant, TextureCache,
};
use bifrost::mesh::{self, GpuMesh, MeshCache, MeshKey, VertexLayoutKey};
use bifrost::passes::geometry::{GeometryPass, MaterialPipelines, PreparedDraw};
use bifrost::passes::pipeline::{self, PipelineCache, PipelineKey, ShaderSet, ShaderVariant};
use bifrost::passes::{DrawArena, READBACK_LAG, READBACK_RING};
use bifrost::passes::{FrameTargets, OverlayPasses, SpriteTextOverlay, PASS_COMPOSITE, PASS_TEXT};
use bifrost::{Bifrost, EditorCamera, ViewSource};
use chassis::graph::types::{ErrorKind, PassDesc, Queue};
use chassis::graph::{self, ImageState, RenderGraph};
use chassis::{
    BindGroup, BindGroupTag, Buffer, BufferDesc, BufferTag, CommandList, ComputePipelineDesc,
    DescriptorKind, DescriptorSetLayoutDesc, Extent2d, Gpu, ImageView, ImageViewTag, Pipeline,
    PipelineDesc, PipelineTag, Registry, Resources, Sampler, SamplerDesc, SamplerTag, Shader,
    ShaderTag, SlotTable, Texture, TextureDesc, TextureTag,
};
use vault::{AssetId, Assets};

// ---- a recorded fake RHI ----

/// A `Resources` that records what it was asked for.
///
/// Handles are real generational handles from `chassis::Registry`, so a destroy
/// bumps the generation exactly as a backend's would and a re-creation is
/// observably a different handle, which is what the mesh-cache test asserts.
#[derive(Default)]
struct FakeResources {
    buffers: Registry<(), BufferTag>,
    textures: Registry<(), TextureTag>,
    views: Registry<(), ImageViewTag>,
    shaders: Registry<(), ShaderTag>,
    pipelines: Registry<(), PipelineTag>,
    groups: Registry<(), BindGroupTag>,
    samplers: Registry<(), SamplerTag>,

    buffer_creates: usize,
    /// Every `write_buffer` call, as `(buffer, offset, byte count)`.
    buffer_writes: Vec<(Buffer, u64, usize)>,
    buffer_destroys: Vec<Buffer>,
    texture_destroys: Vec<Texture>,
    /// Every `upload_texture` call: the texture, the byte count, and the extent
    /// it was told to fill.
    uploads: Vec<(Texture, usize, u32, u32)>,
    pipeline_creates: usize,
    group_creates: usize,
    group_destroys: usize,
    sampler_creates: usize,
    /// What `read_buffer` hands back: the bytes a copy of the readback buffer
    /// would produce. Empty by default, which reads as "no frame".
    readback: Vec<u8>,
    /// Every buffer `read_buffer` was called with, in order.
    read_buffers: Vec<Buffer>,
    /// `(buffer, submission)` for every `read_buffer_after`, in order.
    read_after: Vec<(Buffer, u64)>,
}

impl FakeResources {
    fn new() -> Self {
        Self::default()
    }

    fn shader(&mut self) -> Shader {
        self.shaders.create(())
    }

    fn view(&mut self) -> ImageView {
        self.views.create(())
    }

    fn sampler(&mut self) -> Sampler {
        self.samplers.create(())
    }

    fn texture_handle(&mut self) -> Texture {
        self.textures.create(())
    }
}

impl Resources for FakeResources {
    fn create_buffer(
        &mut self,
        _desc: BufferDesc,
        _initial: Option<&[u8]>,
    ) -> chassis::Result<Buffer> {
        self.buffer_creates += 1;
        Ok(self.buffers.create(()))
    }

    fn write_buffer(&mut self, buffer: &Buffer, offset: u64, data: &[u8]) -> chassis::Result<()> {
        self.buffer_writes.push((*buffer, offset, data.len()));
        Ok(())
    }

    fn create_texture(&mut self, _desc: TextureDesc) -> chassis::Result<Texture> {
        Ok(self.textures.create(()))
    }

    fn create_image_view(&mut self, _texture: &Texture) -> chassis::Result<ImageView> {
        Ok(self.views.create(()))
    }

    fn create_shader(&mut self, _label: &str, _spirv: &[u8]) -> chassis::Result<Shader> {
        Ok(self.shaders.create(()))
    }

    fn create_pipeline(&mut self, _desc: PipelineDesc) -> chassis::Result<Pipeline> {
        self.pipeline_creates += 1;
        Ok(self.pipelines.create(()))
    }

    fn create_compute_pipeline(&mut self, _desc: ComputePipelineDesc) -> chassis::Result<Pipeline> {
        self.pipeline_creates += 1;
        Ok(self.pipelines.create(()))
    }

    fn create_bind_group(
        &mut self,
        _layout: &DescriptorSetLayoutDesc,
        _buffers: &[(u32, Buffer, u64)],
        _images: &[(u32, ImageView)],
        _samplers: &[(u32, Sampler)],
    ) -> chassis::Result<BindGroup> {
        self.group_creates += 1;
        Ok(self.groups.create(()))
    }

    fn create_sampler(&mut self, _desc: &SamplerDesc) -> chassis::Result<Sampler> {
        self.sampler_creates += 1;
        Ok(self.samplers.create(()))
    }

    fn destroy_buffer(&mut self, buffer: Buffer) {
        self.buffer_destroys.push(buffer);
        let _ = self.buffers.remove(buffer);
    }

    fn destroy_texture(&mut self, texture: Texture) {
        self.texture_destroys.push(texture);
        let _ = self.textures.remove(texture);
    }

    fn upload_texture(
        &mut self,
        texture: &Texture,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> chassis::Result<()> {
        self.uploads.push((*texture, data.len(), width, height));
        Ok(())
    }

    fn read_buffer(&mut self, buffer: &Buffer) -> chassis::Result<Vec<u8>> {
        self.read_buffers.push(*buffer);
        Ok(self.readback.clone())
    }

    fn read_buffer_after(
        &mut self,
        buffer: &Buffer,
        submission: chassis::Submission,
    ) -> chassis::Result<Vec<u8>> {
        let frame = *submission
            .downcast_ref::<u64>()
            .expect("the fake's only token is the frame number");
        self.read_after.push((*buffer, frame));
        self.read_buffer(buffer)
    }

    fn destroy_pipeline(&mut self, pipeline: Pipeline) {
        let _ = self.pipelines.remove(pipeline);
    }

    fn destroy_shader(&mut self, shader: Shader) {
        let _ = self.shaders.remove(shader);
    }

    fn destroy_bind_group(&mut self, group: BindGroup) {
        self.group_destroys += 1;
        let _ = self.groups.remove(group);
    }

    fn destroy_sampler(&mut self, sampler: Sampler) {
        let _ = self.samplers.remove(sampler);
    }
}

/// Six distinct shader handles, one per source, which is all a `ShaderSet` is.
fn shaders(resources: &mut FakeResources) -> ShaderSet {
    ShaderSet {
        model: resources.shader(),
        depth: resources.shader(),
        ssao: resources.shader(),
        ssao_upsample: resources.shader(),
        grid: resources.shader(),
        line: resources.shader(),
        composite: resources.shader(),
    }
}

fn asset(index: u32) -> AssetId {
    AssetId::from_parts(index, 1)
}

fn triangle() -> vault::MeshData {
    vault::MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        indices: vec![0, 1, 2],
        ..vault::MeshData::default()
    }
}

// ---- the frame's graph ----

/// The frame's shape, compiled with no device: a depth prepass, the geometry,
/// and the composition into an imported target.
///
/// The composition writes an *import*; that is what makes it a culling root, and
/// without a root `compiler::cull` drops the whole frame (`compiler.rs:296-330`).
#[test]
fn a_three_pass_frame_compiles_in_declaration_order() {
    let mut graph = RenderGraph::new();

    let backbuffer = graph.import_texture("backbuffer", ImageState::Undefined, ImageState::Present);
    let color = graph.import_texture("color", ImageState::Undefined, ImageState::ShaderRead);
    let depth = graph.import_texture("depth", ImageState::Undefined, ImageState::DepthAttachment);
    let shadow = graph.import_texture("shadow", ImageState::Undefined, ImageState::ShaderRead);

    graph.pass("shadow").depth(shadow, 1.0);
    graph
        .pass("opaque")
        .color_load(color)
        .depth(depth, 1.0)
        .read_texture(shadow);
    graph
        .pass("composite")
        .color(backbuffer, [0.0, 0.0, 0.0, 1.0])
        .read_texture(color);

    assert!(
        graph.validate().is_empty(),
        "a frame with no read-write conflict must validate: {}",
        graph.dump()
    );

    let compiled = graph.compile().expect("compiles");
    let order: Vec<&str> = compiled
        .passes
        .iter()
        .map(|pass| pass.name.as_str())
        .collect();
    assert_eq!(order, ["shadow", "opaque", "composite"]);

    // Declaration order is already topological: a write binds the version it
    // produces and a read binds the latest, so the frame is a chain.
    assert_eq!(compiled.find("opaque").map(|pass| pass.index), Some(1));
    assert_eq!(compiled.passes[0].queue, Queue::Graphics);

    // The prepass owns the transition into the shadow map it writes, which is
    // what lets every pass record on its own with no barrier API in sight.
    assert!(
        !compiled.passes[0].pre_barriers.is_empty(),
        "the first writer of a texture must transition it"
    );
}

/// A cycle is refused, and it takes hand-written pass descriptions to build one:
/// the fluent builder binds every read to the version that exists when it is
/// declared, so its declaration order is already a topological order.
#[test]
fn a_cycle_is_refused_rather_than_ordered() {
    let mut resources = chassis::graph::resources::Resources::new();
    let a = resources.create_texture("a", ImageState::Undefined, ImageState::ShaderRead);
    let b = resources.create_texture("b", ImageState::Undefined, ImageState::ShaderRead);

    // Version 1 of `a` is produced by pass 1, and version 1 of `b` by pass 0.
    resources.push_version(
        chassis::graph::ResourceAccess::Texture(a, 0),
        1,
        ImageState::ShaderRead,
    );
    resources.push_version(
        chassis::graph::ResourceAccess::Texture(b, 0),
        0,
        ImageState::ShaderRead,
    );

    let passes = vec![
        PassDesc {
            name: "a".to_owned(),
            reads: vec![chassis::graph::ResourceAccess::Texture(a, 1)],
            ..PassDesc::default()
        },
        PassDesc {
            name: "b".to_owned(),
            reads: vec![chassis::graph::ResourceAccess::Texture(b, 1)],
            ..PassDesc::default()
        },
    ];

    // Both passes are individually valid: they read versions that exist and
    // write nothing, so only the ordering can refuse them.
    assert!(graph::validate(&passes, &resources).is_empty());

    match graph::compile(&passes, &resources) {
        Ok(_) => panic!("two passes reading each other's output is a cycle"),
        Err(errors) => {
            assert_eq!(errors.len(), 1, "{errors:?}");
            assert_eq!(errors[0].kind, ErrorKind::Cycle);
        }
    }
}

/// A pass cannot read and write one texture: `PassStateConflict`, and the reason
/// the grid and the gizmo lines are one pass here rather than two colour-writing
/// passes.
#[test]
fn a_pass_that_reads_what_it_writes_is_refused() {
    let mut graph = RenderGraph::new();
    let color = graph.import_texture("color", ImageState::Undefined, ImageState::ShaderRead);
    graph.pass("grid").color_load(color).read_texture(color);

    let errors = graph.validate();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].kind, ErrorKind::PassStateConflict);
}

// ---- pipelines ----

/// Every field that changes the pipeline is in the key, and the four model
/// variants differ from one another.
#[test]
fn a_pipeline_key_covers_the_state_a_pipeline_is_built_from() {
    use chassis::{SampleCount, TextureFormat};

    let model = PipelineKey::model(
        ShaderVariant::Model,
        TextureFormat::Rgba8Srgb,
        TextureFormat::Depth32Float,
        SampleCount::X1,
    );
    let same = PipelineKey::model(
        ShaderVariant::Model,
        TextureFormat::Rgba8Srgb,
        TextureFormat::Depth32Float,
        SampleCount::X1,
    );
    let double_sided = PipelineKey::model(
        ShaderVariant::ModelDoubleSided,
        TextureFormat::Rgba8Srgb,
        TextureFormat::Depth32Float,
        SampleCount::X1,
    );
    let other_format = PipelineKey::model(
        ShaderVariant::Model,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Depth32Float,
        SampleCount::X1,
    );

    assert_eq!(model, same);
    assert_ne!(model, double_sided);
    assert_ne!(model, other_format);
    // The shadow key shares the vertex layout, so a mesh is one upload for both
    // passes, and differs in everything else.
    assert_ne!(
        model,
        PipelineKey::shadow(TextureFormat::Depth32Float, SampleCount::X1)
    );
}

/// The description a key produces: which stages run, what the state is, and how
/// many sets the pipeline declares.
#[test]
fn a_pipeline_description_matches_its_key() {
    use chassis::{SampleCount, ShaderStage, TextureFormat};

    let mut fake = FakeResources::new();
    let shaders = shaders(&mut fake);

    let model = pipeline::pipeline_desc(
        &PipelineKey::model(
            ShaderVariant::Model,
            TextureFormat::Rgba8Srgb,
            TextureFormat::Depth32Float,
            SampleCount::X1,
        ),
        &shaders,
    );
    assert_eq!(model.shaders.len(), 2, "vertex and fragment");
    assert_eq!(model.shaders[0].stage, ShaderStage::Vertex);
    assert_eq!(model.set_layouts.len(), 4, "frame, material, draw, palette");
    assert!(!model.blend.enabled);
    assert_eq!(model.primitive.cull_mode, 1, "back-face culled");
    assert_eq!(model.depth.format, Some(TextureFormat::Depth32Float));
    assert!(model.depth.write_enabled);
    assert_eq!(model.color_targets.len(), 1);

    // The double-sided variant is the same pipeline with culling off, which is
    // exactly what `Renderer3D::Init` builds a second one for.
    let double_sided = pipeline::pipeline_desc(
        &PipelineKey::model(
            ShaderVariant::ModelDoubleSided,
            TextureFormat::Rgba8Srgb,
            TextureFormat::Depth32Float,
            SampleCount::X1,
        ),
        &shaders,
    );
    assert_eq!(double_sided.primitive.cull_mode, 0);
    assert_eq!(double_sided.shaders, model.shaders);

    // Blending is pipeline state, so the transparent variant turns it on.
    let transparent = pipeline::pipeline_desc(
        &PipelineKey::model(
            ShaderVariant::ModelTransparent,
            TextureFormat::Rgba8Srgb,
            TextureFormat::Depth32Float,
            SampleCount::X1,
        ),
        &shaders,
    );
    assert!(transparent.blend.enabled);

    // A depth-only prepass: one stage, no colour target.
    let shadow = pipeline::pipeline_desc(
        &PipelineKey::shadow(TextureFormat::Depth32Float, SampleCount::X1),
        &shaders,
    );
    assert_eq!(shadow.shaders.len(), 1);
    assert!(shadow.color_targets.is_empty());
    assert_eq!(shadow.set_layouts.len(), 3, "shadow, draw, palette");

    // A full-screen pass has no vertex buffer at all.
    let ssao = pipeline::pipeline_desc(
        &PipelineKey::fullscreen(ShaderVariant::Ssao, TextureFormat::R8Unorm, SampleCount::X1),
        &shaders,
    );
    assert_eq!(ssao.vertex.stride, 0);
    assert!(ssao.vertex.attributes.is_empty());
    assert!(ssao.index_format.is_none(), "drawn with no index buffer");
}

/// One pipeline per key, created once, destroyed on request.
#[test]
fn the_pipeline_cache_creates_one_pipeline_per_key() {
    use chassis::{SampleCount, TextureFormat};

    let mut fake = FakeResources::new();
    let shaders = shaders(&mut fake);
    let mut cache = PipelineCache::new();

    let key = PipelineKey::model(
        ShaderVariant::Model,
        TextureFormat::Rgba8Srgb,
        TextureFormat::Depth32Float,
        SampleCount::X1,
    );
    let first = cache
        .get_or_create(&mut fake, key.clone(), &shaders)
        .expect("creates");
    let second = cache
        .get_or_create(&mut fake, key.clone(), &shaders)
        .expect("hit");
    assert_eq!(first, second);
    assert_eq!(fake.pipeline_creates, 1, "a hit must not create anything");
    assert_eq!(cache.len(), 1);

    let other = PipelineKey::grid(
        TextureFormat::Rgba8Srgb,
        TextureFormat::Depth32Float,
        SampleCount::X1,
    );
    assert_ne!(
        cache
            .get_or_create(&mut fake, other, &shaders)
            .expect("creates"),
        first
    );
    assert_eq!(fake.pipeline_creates, 2);

    assert!(cache.destroy(&mut fake, &key));
    assert!(
        !cache.destroy(&mut fake, &key),
        "destroying twice is not a destroy"
    );
    assert_eq!(cache.len(), 1);
}

/// The grid's axis lines and the gizmo's arrows are the same geometry at the
/// origin, so their colours have to differ for the handle to be visible.
#[test]
fn the_grid_axes_are_a_different_colour_from_the_gizmo_axes() {
    use bifrost::passes::lines::{GRID_AXIS_X, GRID_AXIS_Z};

    // The editor's gizmo colours (`UHE-EDITOR/editor/src/gizmo.rs`), which this
    // crate does not own.
    let gizmo_x = [0.90, 0.20, 0.25, 1.0];
    let gizmo_z = [0.30, 0.45, 0.95, 1.0];

    let distance =
        |a: [f32; 4], b: [f32; 4]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>().sqrt();

    assert!(
        distance(GRID_AXIS_X, gizmo_x) > 0.2,
        "the grid's X line hides the gizmo's X arrow"
    );
    assert!(
        distance(GRID_AXIS_Z, gizmo_z) > 0.2,
        "the grid's Z line hides the gizmo's Z arrow"
    );
}

/// One `DebugLine` is one line-list primitive: an index pair and its two ends,
/// carrying the segment's RGB as opaque vertex colour.
#[test]
fn a_debug_view_becomes_two_vertices_and_one_index_pair_per_segment() {
    let segment = |x: f32| femur::DebugLine {
        from: glam::Vec3::new(x, 0.0, 0.0),
        to: glam::Vec3::new(x, 1.0, 0.0),
        color: glam::Vec3::new(0.2, 0.8, 1.0),
    };
    let debug = femur::PhysicsDebug {
        shapes: vec![segment(0.0), segment(1.0)],
        contacts: vec![segment(2.0)],
        broad_phase: vec![segment(3.0)],
    };

    let (vertices, indices) = bifrost::passes::lines::physics_debug_batch(&debug);
    let segments = debug.shapes.len() + debug.contacts.len() + debug.broad_phase.len();

    assert_eq!(vertices.len(), segments * 2);
    assert_eq!(indices.len(), segments * 2);
    for (index, vertex) in vertices.iter().enumerate() {
        assert_eq!(vertex.color, [0.2, 0.8, 1.0, 1.0], "vertex {index}");
    }
    assert_eq!(indices[..4], [0, 1, 2, 3]);
    assert_eq!(vertices[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(vertices[1].position, [0.0, 1.0, 0.0]);
}

// ---- materials ----

/// A material key is the textures it binds and every value the shader reads, so
/// two materials that differ anywhere are two entries.
#[test]
fn a_material_key_covers_textures_and_factors() {
    let mut fake = FakeResources::new();
    let texture = fake.texture_handle();

    let textures = MaterialTextures {
        albedo: Some(texture),
        ..MaterialTextures::default()
    };
    let material = Material::default();
    let key = MaterialKey::of(&material, textures, MaterialVariant::Opaque);

    assert_eq!(
        key,
        MaterialKey::of(&Material::default(), textures, MaterialVariant::Opaque)
    );

    let rough = Material {
        roughness_factor: 0.5,
        ..material
    };
    assert_ne!(
        key,
        MaterialKey::of(&rough, textures, MaterialVariant::Opaque),
        "a factor the shader reads is part of the key"
    );

    let white = MaterialTextures {
        albedo: Some(texture),
        normal: Some(texture),
        ..MaterialTextures::default()
    };
    assert_ne!(
        key,
        MaterialKey::of(&material, white, MaterialVariant::Opaque),
        "so is a texture"
    );

    assert_ne!(
        key,
        MaterialKey::of(&material, textures, MaterialVariant::Transparent),
        "and the variant, because culling and blending are pipeline state"
    );
}

/// A depth texture is declared as one, not as a sampled colour image.
///
/// Two of the renderer's sets read a `texture_depth_2d`: the frame set's shadow
/// map and the SSAO set's depth buffer. wgpu maps a depth view to
/// `TextureSampleType::Depth` and refuses a bind group whose view does not match
/// its layout's sample type by name, so declaring either `SampledImage` would make
/// the default backend fail at the frame's first bind group. Vulkan takes one
/// `SAMPLED_IMAGE` descriptor for both, which is why this went unnoticed there.
#[test]
fn the_two_depth_reads_are_declared_as_depth() {
    let kind_of = |layout: &DescriptorSetLayoutDesc, binding: u32| {
        layout
            .bindings
            .iter()
            .find(|b| b.binding == binding)
            .unwrap_or_else(|| panic!("binding {binding} of {}", layout.label))
            .kind
    };

    assert_eq!(
        kind_of(&pipeline::frame_layout(), 2),
        DescriptorKind::SampledDepth,
        "the shadow map, at the binding the frame's set is built with",
    );
    assert_eq!(
        kind_of(&pipeline::ssao_layout(), 1),
        DescriptorKind::SampledDepth,
        "the depth buffer SSAO reads",
    );

    // Nothing else samples depth: the sets that read a render target read colour
    // textures, and the rest are uniform buffers and samplers.
    for layout in [pipeline::material_layout(), pipeline::composite_layout()] {
        assert!(
            layout
                .bindings
                .iter()
                .all(|b| b.kind != DescriptorKind::SampledDepth),
            "{} declares no depth texture",
            layout.label,
        );
    }
    assert_eq!(
        kind_of(&pipeline::composite_layout(), 0),
        DescriptorKind::SampledImage,
        "the scene colour stays a sampled image",
    );
}

/// `FillMaterialGPU` puts every value in the slot the shader reads it from.
#[test]
fn a_material_packs_into_the_shader_layout() {
    let mut uv_transforms = [material::UVTransform::IDENTITY; material::SLOT_COUNT];
    uv_transforms[material::SLOT_ALBEDO] = material::UVTransform {
        cos_rotation: 0.0,
        sin_rotation: 1.0,
        scale: [2.0, 3.0],
        offset: [0.5, 0.25],
    };
    let material = Material {
        base_color_factor: [0.25, 0.5, 0.75, 1.0],
        metallic_factor: 0.9,
        roughness_factor: 0.1,
        normal_scale: 2.0,
        occlusion_strength: 0.5,
        emissive_factor: [1.0, 0.0, 0.0],
        emissive_strength: 3.0,
        alpha: material::AlphaMode::Mask,
        alpha_cutoff: 0.25,
        unlit: true,
        uv_transforms,
        ..Material::default()
    };

    let gpu = material::fill_gpu(&material);

    assert_eq!(gpu.base_color_factor, [0.25, 0.5, 0.75, 1.0]);
    assert_eq!(gpu.emissive_factor_strength, [1.0, 0.0, 0.0, 3.0]);
    assert_eq!(gpu.factors, [0.9, 0.1, 2.0, 0.5]);
    assert_eq!(gpu.params, [0.25, 1.0, 1.0, 0.0]);
    assert_eq!(
        gpu.uv_transform[material::SLOT_ALBEDO],
        [0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(gpu.uv_offset[material::SLOT_ALBEDO], [0.5, 0.25, 0.0, 0.0]);
    // A slot with no transform is the identity, so an absent
    // KHR_texture_transform is not a zero scale.
    assert_eq!(
        gpu.uv_transform[material::SLOT_EMISSIVE],
        [1.0, 0.0, 1.0, 1.0]
    );
}

/// A material is created once per key and is destroyed with its texture: a bind
/// group naming a destroyed texture is a descriptor pointing at freed memory.
#[test]
fn destroying_a_texture_drops_every_material_that_bound_it() {
    let mut fake = FakeResources::new();
    let texture = fake.texture_handle();
    let view = fake.view();
    let sampler = fake.sampler();

    let textures = MaterialTextures {
        albedo: Some(texture),
        ..MaterialTextures::default()
    };
    let key = MaterialKey::of(&Material::default(), textures, MaterialVariant::Opaque);
    let layout = pipeline::material_layout();

    let mut cache = MaterialCache::new();
    let first = cache
        .get_or_insert(
            &mut fake,
            key,
            &Material::default(),
            &[(2, view)],
            &[(1, sampler)],
            &layout,
        )
        .expect("creates");
    let second = cache
        .get_or_insert(
            &mut fake,
            key,
            &Material::default(),
            &[(2, view)],
            &[(1, sampler)],
            &layout,
        )
        .expect("hit");
    assert_eq!(first.group, second.group);
    assert_eq!(cache.len(), 1);

    assert_eq!(cache.invalidate_texture(&mut fake, texture), 1);
    assert!(cache.is_empty());
    assert_eq!(fake.group_destroys, 1);
    assert_eq!(
        fake.buffer_destroys.len(),
        1,
        "the parameter buffer went too"
    );

    assert_eq!(
        cache.invalidate_texture(&mut fake, texture),
        0,
        "already gone"
    );
}

// ---- textures ----

/// A texture's bytes reach the GPU through `Resources::upload_texture`.
///
/// This is the test that would have caught the version that set
/// `TextureDesc::initial_data` instead: `stormglass` never reads that field, so
/// the texture would have been created empty and every material would have
/// sampled black on the default backend, with no error anywhere. Asserting on
/// the call (its handle, its byte count and its extent) is what makes the
/// difference observable without a device.
#[test]
fn a_texture_upload_goes_through_upload_texture_with_the_image_bytes() {
    let mut fake = FakeResources::new();
    let mut cache = TextureCache::new();
    let image = vault::Image {
        width: 2,
        height: 1,
        rgba8: vec![1, 2, 3, 4, 5, 6, 7, 8],
        srgb: true,
    };

    let first = cache
        .get_or_upload(&mut fake, asset(3), &image)
        .expect("uploads");
    assert_eq!(fake.uploads.len(), 1, "one upload, and it is issued");
    let (handle, bytes, width, height) = fake.uploads[0];
    assert_eq!(
        handle, first.texture,
        "the bytes go into the texture it made"
    );
    assert_eq!(bytes, image.rgba8.len(), "all of them");
    assert_eq!((width, height), (image.width, image.height));

    // The second call for one asset is a cache hit, so no bytes move again.
    let hit = cache
        .get_or_upload(&mut fake, asset(3), &image)
        .expect("hit");
    assert_eq!(hit.texture, first.texture);
    assert_eq!(fake.uploads.len(), 1, "a hit uploads nothing");

    // And a destroyed texture is uploaded again rather than handed back stale.
    assert!(cache.destroy(&mut fake, asset(3)).is_some());
    assert_eq!(fake.texture_destroys.len(), 1);
    let again = cache
        .get_or_upload(&mut fake, asset(3), &image)
        .expect("re-uploads");
    assert_eq!(fake.uploads.len(), 2);
    assert_ne!(
        again.texture, first.texture,
        "a generation bump, not the old handle"
    );
    assert_eq!(
        again.sampler, first.sampler,
        "the sampler is shared, not re-made"
    );
    assert_eq!(fake.sampler_creates, 1);
}

/// The 1x1 stand-ins a material with no map samples: white for the factors, and
/// `(0.5, 0.5, 1)` for a normal map, which is not white.
#[test]
fn the_neutral_maps_are_one_texel_and_the_normal_is_not_white() {
    let mut fake = FakeResources::new();
    let mut cache = TextureCache::new();

    let white = cache.white(&mut fake).expect("white");
    let neutral = cache.neutral_normal(&mut fake).expect("neutral");
    assert_eq!(fake.uploads.len(), 2);
    assert_eq!(fake.uploads[0].1, 4, "one RGBA texel");
    assert_eq!(fake.uploads[0].2, 1);

    // Same call twice is the same texture, not a second upload.
    assert_eq!(
        cache.white(&mut fake).expect("white").texture,
        white.texture
    );
    assert_eq!(fake.uploads.len(), 2);

    assert_ne!(white.texture, neutral.texture);
    assert_eq!(white.sampler, neutral.sampler);
}

// ---- meshes ----

/// The same lookup twice is one upload, and a destroyed mesh is uploaded again
/// rather than handed back as a stale handle.
#[test]
fn a_mesh_cache_returns_one_handle_and_reuploads_after_a_destroy() {
    let mut fake = FakeResources::new();
    let mut cache = MeshCache::new();
    let key = MeshKey::new(asset(7), VertexLayoutKey::of(&mesh::vertex_layout()));
    let data = triangle();

    let first = cache.get_or_upload(&mut fake, key, &data).expect("uploads");
    assert_eq!(first.index_count, 3);
    assert_eq!(first.vertex_count, 3);
    assert_eq!(fake.buffer_creates, 2, "one vertex buffer and one index");

    let second = cache.get_or_upload(&mut fake, key, &data).expect("hit");
    assert_eq!(first, second);
    assert_eq!(fake.buffer_creates, 2, "a hit uploads nothing");

    assert!(cache.destroy(&mut fake, &key));
    assert_eq!(fake.buffer_destroys.len(), 2);
    assert!(!cache.destroy(&mut fake, &key));

    let third = cache
        .get_or_upload(&mut fake, key, &data)
        .expect("uploads again");
    assert_eq!(fake.buffer_creates, 4);
    // A generation bump, not the same handle: a handle that outlived its
    // resource is what the generational registry exists to catch.
    assert_ne!(third.vertex, first.vertex);
    assert_eq!(third.index_count, first.index_count);
}

/// Two meshes with different vertex layouts never share a buffer, and the key
/// says so.
#[test]
fn a_mesh_key_includes_the_vertex_layout() {
    let layout = mesh::vertex_layout();
    let same = VertexLayoutKey::of(&layout);
    assert_eq!(same, VertexLayoutKey::of(&mesh::vertex_layout()));

    let mut other = layout.clone();
    other.stride += 4;
    assert_ne!(same, VertexLayoutKey::of(&other));

    assert_ne!(
        MeshKey::new(asset(1), same),
        MeshKey::new(asset(2), same),
        "and the asset"
    );
}

/// Interleaving fills a missing attribute with a default rather than dropping
/// the vertex, and an absent normal map's neutral is not white.
#[test]
fn interleaving_uses_the_defaults_for_absent_attributes() {
    let data = vault::MeshData {
        normals: vec![[0.0, 1.0, 0.0]],
        colors: vec![[1.0, 0.0, 0.0, 1.0]],
        ..triangle()
    };

    let vertices = mesh::interleave(&data);
    assert_eq!(vertices.len(), 3);
    assert_eq!(vertices[0].normal, [0.0, 1.0, 0.0]);
    assert_eq!(vertices[0].color, [1.0, 0.0, 0.0, 1.0]);

    assert_eq!(vertices[1].normal, [0.0, 0.0, 1.0], "the neutral normal");
    assert_eq!(vertices[1].uv, [0.0, 0.0]);
    assert_eq!(vertices[1].tangent, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        vertices[1].color, [1.0; 4],
        "white, so the multiply is neutral"
    );

    // The layout's offsets are the struct's, computed rather than written down.
    let layout = mesh::vertex_layout();
    assert_eq!(layout.stride as usize, std::mem::size_of::<mesh::Vertex>());
    assert_eq!(layout.attributes.len(), 7);
    assert_eq!(
        layout.attributes[0].offset,
        std::mem::offset_of!(mesh::Vertex, position) as u32
    );
}

// ---- shaders ----

/// Every embedded shader parses, validates, and compiles to SPIR-V for each
/// entry point it declares.
///
/// `forge::wgsl` validates with naga's validator, which is the same rule set
/// wgpu enforces at shader-module creation, so a shader that passes here cannot
/// fail on either backend for a *shader* reason. Running it means the renderer's
/// six shaders are checked without a device, which is the point of compiling
/// WGSL in process instead of shelling out to Slang.
#[test]
fn every_embedded_shader_compiles_for_its_entry_points() {
    let shaders = [
        ("model", pipeline::MODEL_WGSL),
        ("depth", pipeline::DEPTH_WGSL),
        ("ssao", pipeline::SSAO_WGSL),
        ("grid", pipeline::GRID_WGSL),
        ("line", pipeline::LINE_WGSL),
        ("composite", pipeline::COMPOSITE_WGSL),
    ];

    for (label, source) in shaders {
        forge::wgsl::check(source, label).unwrap_or_else(|error| panic!("{label}: {error}"));

        let declared = forge::wgsl::entry_points(source, label)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert!(
            !declared.is_empty(),
            "{label} declares no entry point, so nothing would be compiled"
        );

        let spirv =
            forge::wgsl::to_spirv(source, label).unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(spirv.len() % 4, 0, "{label}: SPIR-V is a word stream");
        assert!(!spirv.is_empty());

        let module = forge::Module::from_bytes(&spirv).expect("a module");
        let parsed = chassis::spirv::entry_points(&module.words).expect("a module");
        let declared_names: Vec<&str> = declared.iter().map(|(name, _)| name.as_str()).collect();
        let parsed_names: Vec<&str> = parsed.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(
            parsed_names, declared_names,
            "{label}: the module keeps the names the source declared"
        );
    }

    // Every graphics shader the vertex/fragment pairs are built from declares
    // the two entry points the pipelines ask for by name.
    for (label, source) in [
        ("model", pipeline::MODEL_WGSL),
        ("ssao", pipeline::SSAO_WGSL),
        ("grid", pipeline::GRID_WGSL),
        ("line", pipeline::LINE_WGSL),
        ("composite", pipeline::COMPOSITE_WGSL),
    ] {
        let entries = forge::wgsl::entry_points(source, label).expect("parses");
        let names: Vec<&str> = entries.iter().map(|(name, _)| name.as_str()).collect();
        assert!(names.contains(&pipeline::VS_MAIN), "{label}: {names:?}");
        assert!(names.contains(&pipeline::FS_MAIN), "{label}: {names:?}");
    }
}

/// A module with more than one stage yields one entry per `ExecutionModel`, with
/// the name each entry point declared.
#[test]
fn the_parser_reads_every_stage_from_a_multi_entry_module() {
    let spirv = forge::wgsl::to_spirv(pipeline::MODEL_WGSL, "model").expect("compiles");
    let module = forge::Module::from_bytes(&spirv).expect("a module");
    let entries = chassis::spirv::entry_points(&module.words).expect("a module");

    assert!(entries.iter().any(|entry| {
        entry.stage == chassis::ShaderStage::Vertex && entry.name == pipeline::VS_MAIN
    }));
    assert!(entries.iter().any(|entry| {
        entry.stage == chassis::ShaderStage::Fragment && entry.name == pipeline::FS_MAIN
    }));
}

// ---- the frame loop's one call ----

/// A `Gpu` that records what a frame handed it, with no device behind it.
///
/// `record_frame` takes pure data -- a compiled graph, a slot table, a recorder --
/// so the hand-off is testable without a driver. What the passes *mean* is the
/// backends' business; what matters here is that a frame with passes reaches one,
/// and that a refusal comes back to the caller rather than being logged away.
struct FakeGpu {
    resources: FakeResources,
    info: chassis::GpuInfo,
    graph: FakeGraph,
    extent: chassis::Extent2d,
    frames: usize,
    passes: usize,
    slots: usize,
    /// The readback buffer each frame's slot table named, in order.
    readback_targets: Vec<Option<Buffer>>,
    /// The tokens `wait_for_submission` was called with, in order.
    waited: Vec<u64>,
}

impl FakeGpu {
    fn new() -> Self {
        Self {
            resources: FakeResources::new(),
            info: chassis::GpuInfo {
                name: "fake".to_owned(),
                sync_tier: chassis::SyncTier::Legacy,
                max_texture_size: 8192,
                frames_in_flight: 3,
            },
            graph: FakeGraph,
            extent: chassis::Extent2d {
                width: 1280,
                height: 720,
            },
            frames: 0,
            passes: 0,
            slots: 0,
            readback_targets: Vec::new(),
            waited: Vec::new(),
        }
    }
}

impl chassis::Gpu for FakeGpu {
    fn info(&self) -> &chassis::GpuInfo {
        &self.info
    }

    fn swapchain(&self) -> Option<&dyn chassis::Swapchain> {
        None
    }

    fn extent(&self) -> chassis::Extent2d {
        self.extent
    }

    fn resize(&mut self, extent: chassis::Extent2d) -> chassis::Result<()> {
        self.extent = extent;
        Ok(())
    }

    fn graph(&mut self) -> &mut dyn chassis::RenderGraph {
        &mut self.graph
    }

    fn resources(&mut self) -> &mut dyn Resources {
        &mut self.resources
    }

    fn record_frame(
        &mut self,
        frame: &graph::CompiledGraph,
        slots: &graph::SlotTable,
        _recorder: Arc<dyn graph::PassRecorder>,
    ) -> chassis::Result<chassis::Submission> {
        self.frames += 1;
        self.passes = frame.passes.len();
        self.slots = slots.texture_count();
        // The readback pass writes the one imported buffer, so slot 0 is it.
        self.readback_targets.push(slots.buffer(0));
        Ok(chassis::Submission::new(self.frames as u64))
    }

    fn wait_for_submission(&mut self, submission: chassis::Submission) -> chassis::Result<()> {
        self.waited.push(
            *submission
                .downcast_ref::<u64>()
                .expect("the fake's only token is the frame number"),
        );
        Ok(())
    }

    fn frame_begin(&mut self) -> chassis::Result<()> {
        Ok(())
    }

    fn submit(&mut self) -> chassis::Result<()> {
        Ok(())
    }

    fn wait_idle(&mut self) {}
}

/// The `RenderGraph` trait, which a frame never declares through.
///
/// The frame declares into the fluent builder (`chassis::graph::RenderGraph`);
/// this exists only because `Gpu::graph` hands one out.
#[derive(Default)]
struct FakeGraph;

impl chassis::RenderGraph for FakeGraph {
    fn reset(&mut self) {}

    fn add_pass(&mut self, _desc: chassis::RenderPassDesc) -> chassis::PassId {
        0
    }

    fn add_dependency(&mut self, _producer: chassis::PassId, _consumer: chassis::PassId) {}

    fn compile(&mut self) -> chassis::Result<Vec<chassis::PassId>> {
        Ok(Vec::new())
    }

    fn execute(&mut self, _jobs: &grind::Jobs) -> chassis::Result<()> {
        Ok(())
    }
}

/// A `Gpu` that leaves `record_frame` at its default: what a backend that has not
/// adopted the method runs. The refusal is the difference between a frame that
/// fails loudly and a frame that quietly did nothing.
struct NoFrameGpu {
    resources: FakeResources,
    info: chassis::GpuInfo,
    graph: FakeGraph,
    extent: chassis::Extent2d,
}

impl NoFrameGpu {
    fn new() -> Self {
        Self {
            resources: FakeResources::new(),
            info: chassis::GpuInfo {
                name: "no-frame".to_owned(),
                sync_tier: chassis::SyncTier::Legacy,
                max_texture_size: 8192,
                frames_in_flight: 1,
            },
            graph: FakeGraph,
            extent: chassis::Extent2d {
                width: 1280,
                height: 720,
            },
        }
    }
}

impl chassis::Gpu for NoFrameGpu {
    fn info(&self) -> &chassis::GpuInfo {
        &self.info
    }

    fn swapchain(&self) -> Option<&dyn chassis::Swapchain> {
        None
    }

    fn extent(&self) -> chassis::Extent2d {
        self.extent
    }

    fn resize(&mut self, extent: chassis::Extent2d) -> chassis::Result<()> {
        self.extent = extent;
        Ok(())
    }

    fn graph(&mut self) -> &mut dyn chassis::RenderGraph {
        &mut self.graph
    }

    fn resources(&mut self) -> &mut dyn Resources {
        &mut self.resources
    }

    fn wait_for_submission(&mut self, _submission: chassis::Submission) -> chassis::Result<()> {
        Ok(())
    }

    fn frame_begin(&mut self) -> chassis::Result<()> {
        Ok(())
    }

    fn submit(&mut self) -> chassis::Result<()> {
        Ok(())
    }

    fn wait_idle(&mut self) {}
}

fn renderer_for(gpu: &mut dyn chassis::Gpu) -> Bifrost {
    let mut assets = vault::Assets::new(".");
    Bifrost::new(&mut *gpu, &mut assets).expect("the frame's own resources need no device")
}

fn editor_camera() -> EditorCamera {
    EditorCamera::new(45.0, 16.0 / 9.0, 0.1, 1000.0)
}

/// A frame with passes reaches `record_frame`, once, carrying the slots the
/// renderer minted for them.
#[test]
fn a_frame_with_passes_reaches_record_frame() {
    let mut gpu = FakeGpu::new();
    let mut renderer = renderer_for(&mut gpu);
    let jobs = grind::Jobs::new();
    let scene = meats::Scene::new();
    let camera = editor_camera();

    renderer
        .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
        .expect("a fake device takes the whole frame");

    assert_eq!(gpu.frames, 1, "one frame, one call");
    assert!(
        gpu.passes >= 5,
        "the frame's passes reached the backend, not just the graph: {}",
        gpu.passes
    );
    assert!(
        gpu.slots >= 4,
        "the handles the renderer minted reached the backend: {}",
        gpu.slots
    );
}

/// A backend that cannot execute a frontend frame is an error out of `render`,
/// which is the branch the renderer used to swallow with a warning.
#[test]
fn an_unsupported_backend_surfaces_from_render() {
    let mut gpu = NoFrameGpu::new();
    let mut renderer = renderer_for(&mut gpu);
    let jobs = grind::Jobs::new();
    let scene = meats::Scene::new();
    let camera = editor_camera();

    let error = renderer
        .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
        .expect_err("the trait's default refuses a frontend frame");
    assert!(
        matches!(error, chassis::Error::Unsupported(_)),
        "a refusal, named: {error:?}"
    );
}

// ---- the per-draw uniform arena ----

/// The arena is refilled, not re-created: the second frame for the same draws
/// writes into the same buffer and reuses every bind group.
///
/// This is the gate on the per-draw allocation the renderer used to pay. Before
/// the arena kept its buffer and groups, each frame created one buffer and one
/// bind group per draw, so `buffer_creates` and `group_creates` climbed with the
/// draw count on every frame.
#[test]
fn the_draw_arena_reuses_its_buffer_and_groups_across_frames() {
    let mut fake = FakeResources::new();
    let layout = pipeline::draw_layout();
    let blocks: Vec<Vec<u8>> = (0..64).map(|i| vec![i as u8; 96]).collect();
    let mut arena = DrawArena::new();

    let made = arena
        .begin(&mut fake, &layout, &blocks)
        .expect("first frame");
    assert_eq!(made, blocks.len());
    let buffer_creates = fake.buffer_creates;
    let group_creates = fake.group_creates;
    assert_eq!(
        group_creates,
        blocks.len(),
        "one group per draw on the first frame"
    );

    // Same draws next frame: a write, and nothing else.
    arena
        .begin(&mut fake, &layout, &blocks)
        .expect("second frame");
    assert_eq!(
        fake.buffer_creates, buffer_creates,
        "the arena buffer is reused, not re-created"
    );
    assert_eq!(
        fake.group_creates, group_creates,
        "the bind groups are reused, not re-created"
    );
    assert_eq!(
        fake.buffer_writes.len(),
        2,
        "one write per frame, for the whole frame"
    );

    // A shorter frame reuses the buffer and rebuilds only the groups it needs.
    arena
        .begin(&mut fake, &layout, &blocks[..4])
        .expect("shrinks");
    assert_eq!(fake.buffer_creates, buffer_creates, "still the one buffer");
    assert_eq!(
        fake.group_creates,
        group_creates + 4,
        "only the groups the shorter frame has"
    );

    arena.destroy(&mut fake);
    assert!(arena.group(0).is_none(), "destroyed groups stop resolving");
}

// ---- the geometry pass's binds ----

/// A `CommandList` that counts how often each bind is issued.
#[derive(Default)]
struct RecordingList {
    frame_binds: usize,
    material_binds: usize,
    draw_binds: usize,
    pipeline_binds: usize,
    vertex_binds: usize,
    index_binds: usize,
    draws: usize,
}

impl CommandList for RecordingList {
    fn begin_rendering(&mut self, _desc: &chassis::RenderPassDesc) {}
    fn end_rendering(&mut self) {}
    fn begin_compute(&mut self) {}
    fn end_compute(&mut self) {}
    fn set_viewport(&mut self, _extent: Extent2d) {}
    fn set_scissor(&mut self, _x: u32, _y: u32, _w: u32, _h: u32) {}

    fn bind_pipeline(&mut self, _pipeline: &Pipeline) {
        self.pipeline_binds += 1;
    }

    fn bind_group(&mut self, set: u32, _group: &BindGroup) {
        match set {
            bifrost::passes::geometry::FRAME_SET => self.frame_binds += 1,
            bifrost::passes::geometry::MATERIAL_SET => self.material_binds += 1,
            bifrost::passes::geometry::DRAW_SET => self.draw_binds += 1,
            _ => {}
        }
    }

    fn set_push_constants(&mut self, _data: &[u8]) -> chassis::Result<()> {
        Err(chassis::Error::Unsupported(
            "push constants are not declared by this RHI; use a uniform buffer".to_owned(),
        ))
    }
    fn bind_vertex_buffer(&mut self, _slot: u32, _buffer: &Buffer, _offset: u64) {
        self.vertex_binds += 1;
    }
    fn bind_index_buffer(&mut self, _buffer: &Buffer, _offset: u64) {
        self.index_binds += 1;
    }
    fn draw(&mut self, _vertices: u32, _instances: u32) {}
    fn draw_indexed(&mut self, _indices: u32, _instances: u32) {
        self.draws += 1;
    }
    fn dispatch(&mut self, _x: u32, _y: u32, _z: u32) {}
    fn copy_buffer(&mut self, _src: &Buffer, _dst: &Buffer) {}
    fn copy_texture_to_buffer(&mut self, _src: &Texture, _dst: &Buffer) {}
}

/// Consecutive draws that share pipeline, material and mesh bind each of those
/// once, while every draw still binds its own uniform block and issues one draw.
#[test]
fn geometry_binds_shared_state_once_per_run() {
    let mut fake = FakeResources::new();
    let material_a = fake.groups.create(());
    let material_b = fake.groups.create(());
    let mesh_a = GpuMesh {
        vertex: fake.buffers.create(()),
        index: fake.buffers.create(()),
        index_count: 3,
        vertex_count: 3,
    };
    let mesh_b = GpuMesh {
        vertex: fake.buffers.create(()),
        index: fake.buffers.create(()),
        index_count: 3,
        vertex_count: 3,
    };

    let draw = |mesh: GpuMesh, material: BindGroup| PreparedDraw {
        mesh,
        uniforms: BindGroup::null(),
        material: Some(material),
        variant: MaterialVariant::Opaque,
    };
    let draws = vec![
        draw(mesh_a, material_a),
        draw(mesh_a, material_a),
        draw(mesh_a, material_a),
        draw(mesh_a, material_b),
        draw(mesh_b, material_b),
    ];

    let pipelines = MaterialPipelines {
        opaque: Pipeline::null(),
        opaque_double_sided: Pipeline::null(),
        transparent: Pipeline::null(),
        transparent_double_sided: Pipeline::null(),
    };
    let pass = GeometryPass::new(pipelines, BindGroup::null(), BindGroup::null(), draws);

    let slots = SlotTable::default();
    let scope = chassis::PassScope {
        index: 0,
        name: bifrost::passes::PASS_OPAQUE,
        extent: Extent2d {
            width: 1280,
            height: 720,
        },
        slots: &slots,
    };

    let mut list = RecordingList::default();
    pass.record(&scope, &mut list).expect("records");

    assert_eq!(list.pipeline_binds, 1, "one pipeline for the run");
    assert_eq!(list.frame_binds, 1, "the frame set is bound once");
    assert_eq!(list.material_binds, 2, "one bind per distinct material");
    assert_eq!(list.vertex_binds, 2, "one bind per distinct mesh");
    assert_eq!(list.index_binds, 2, "the index buffer moves with the mesh");
    assert_eq!(list.draw_binds, 5, "every draw binds its own block");
    assert_eq!(list.draws, 5, "and is drawn");
}

// ---- the device's extent ----

/// A device resize is followed once, and the camera's viewport -- which the
/// projection's aspect ratio is derived from -- is the device's afterwards.
///
/// This is what makes the editor's viewport panel its own resolution: the panel
/// resizes the device, and the renderer picks the new extent up on the next
/// frame instead of the frontend keeping a second copy of the size.
#[test]
fn a_device_resize_is_followed_once() {
    let mut gpu = FakeGpu::new();
    let mut plugin = bifrost::BifrostPlugin::new();

    assert!(
        !plugin.follow_extent(&mut gpu),
        "the plugin starts at the fake's own extent"
    );
    assert_eq!(plugin.camera().viewport_size(), (1280.0, 720.0));

    chassis::Gpu::resize(
        &mut gpu,
        chassis::Extent2d {
            width: 640,
            height: 400,
        },
    )
    .expect("the fake records any extent");

    assert!(plugin.follow_extent(&mut gpu), "the extent changed");
    assert_eq!(plugin.camera().viewport_size(), (640.0, 400.0));
    assert!(!plugin.follow_extent(&mut gpu), "and only once");
}

/// The render width is a multiple of 64 texels, so the readback copy's row pitch
/// equals the packed row and no de-pad pass is needed.
///
/// A copy pads its rows to `COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes, 64 texels),
/// so an arbitrary viewport width -- any panel dragged to a size -- made the copy
/// produce a longer buffer than the image and the whole frame was copied a second
/// time to strip it. Rendering at the aligned width removes that copy.
#[test]
fn the_render_width_is_aligned_so_the_readback_needs_no_depad() {
    // 100 texels: a packed row of 400 bytes, 512 with the copy's padding.
    const WIDTH: u32 = 100;
    const HEIGHT: u32 = 3;
    const ALIGNED: u32 = 128;

    let mut gpu = FakeGpu::new();
    let mut renderer = renderer_for(&mut gpu);
    let jobs = grind::Jobs::new();
    let scene = meats::Scene::new();
    let camera = editor_camera();

    chassis::Gpu::resize(
        &mut gpu,
        Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    )
    .expect("the fake records any extent");

    // Packed at the aligned width: every row is its own byte pattern, so a row
    // taken from the wrong offset cannot pass.
    let packed = ALIGNED as usize * 4;
    let mut bytes = Vec::with_capacity(packed * HEIGHT as usize);
    for row in 0..HEIGHT as usize {
        bytes.extend(std::iter::repeat_n(row as u8 + 1, packed));
    }
    gpu.resources.readback = bytes;

    // The ring reads a frame that is `READBACK_LAG` submissions old, so it takes
    // that many frames plus one before a frame comes back.
    for _ in 0..=READBACK_LAG {
        renderer
            .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
            .expect("a fake device takes the whole frame");
    }

    let image = renderer.frame_image().expect("a readback frame");
    assert_eq!(image.width, ALIGNED, "the render width is a multiple of 64");
    assert_eq!(image.width % 64, 0, "a copy of this width needs no padding");
    assert_eq!(image.height, HEIGHT);
    assert_eq!(
        image.bgra.len(),
        packed * HEIGHT as usize,
        "packed, no padding"
    );

    for row in 0..HEIGHT as usize {
        let expected = vec![row as u8 + 1; packed];
        assert_eq!(
            &image.bgra[row * packed..(row + 1) * packed],
            expected.as_slice(),
            "row {row} came from the wrong offset"
        );
    }
}

// ---- the readback ring ----

/// The ring reads a frame from [`READBACK_LAG`] submissions ago, so no frame maps
/// the buffer it just wrote.
///
/// Before the ring there was one buffer, mapped every frame immediately after the
/// submission that wrote it, which is the wait that made the frame GPU + CPU.
#[test]
fn the_readback_ring_reads_an_older_submission() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 2;
    const FRAMES: usize = READBACK_RING + 1;

    let mut gpu = FakeGpu::new();
    let mut renderer = renderer_for(&mut gpu);
    let jobs = grind::Jobs::new();
    let scene = meats::Scene::new();
    let camera = editor_camera();

    chassis::Gpu::resize(
        &mut gpu,
        Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    )
    .expect("the fake records any extent");
    gpu.resources.readback = vec![0u8; (WIDTH * HEIGHT * 4) as usize];

    for _ in 0..FRAMES {
        renderer
            .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
            .expect("a fake device takes the whole frame");
    }

    let targets = &gpu.readback_targets;
    let reads = &gpu.resources.read_buffers;
    assert_eq!(targets.len(), FRAMES, "one readback slot per frame");
    assert_eq!(
        reads.len(),
        FRAMES - READBACK_LAG as usize,
        "the first frames are warm-up, with no retired slot to read"
    );

    // The read at frame `f` is the buffer written at frame `f - READBACK_LAG`.
    for (offset, read) in reads.iter().enumerate() {
        assert_eq!(
            Some(*read),
            targets[offset],
            "read {offset} is not the frame {READBACK_LAG} submissions back"
        );
    }

    // And no frame reads the buffer it wrote.
    for (offset, read) in reads.iter().enumerate() {
        let written = targets[offset + READBACK_LAG as usize];
        assert_ne!(Some(*read), written, "a frame read its own readback buffer");
    }

    // The wait is the slot's own submission, not the newest one: that is what
    // makes the ring an overlap rather than a rename of the old blocking read.
    assert_eq!(gpu.resources.read_after.len(), reads.len());
    for (offset, (buffer, submission)) in gpu.resources.read_after.iter().enumerate() {
        assert_eq!(Some(*buffer), targets[offset]);
        assert_eq!(
            *submission,
            (offset + 1) as u64,
            "read {offset} waited on the wrong submission"
        );
    }
}

/// Two consecutive frames of the same static scene create nothing.
///
/// The frame-wide uniform sets used to destroy and recreate a buffer and a bind
/// group every frame. They are now kept while their inputs are unchanged, so a
/// steady frame touches no allocator.
#[test]
fn a_steady_frame_creates_no_bind_groups() {
    let mut gpu = FakeGpu::new();
    let mut renderer = renderer_for(&mut gpu);
    let jobs = grind::Jobs::new();
    let scene = meats::Scene::new();
    let camera = editor_camera();

    // Each of the ring's slots builds its sets on first use, so the first
    // `frames_in_flight` frames are allocation, not churn.
    let depth = gpu.info().frames_in_flight as usize;
    for _ in 0..depth {
        renderer
            .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
            .expect("a fake device takes the whole frame");
    }

    let buffers = gpu.resources.buffer_creates;
    let groups = gpu.resources.group_creates;

    // Same extent, same scene and a camera that has not moved: the ring comes
    // back to slot zero, whose uniforms and bound resources are unchanged, so
    // there is nothing to rebuild.
    renderer
        .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
        .expect("a fake device takes the whole frame");

    assert_eq!(
        gpu.resources.buffer_creates, buffers,
        "a steady frame creates no buffers"
    );
    assert_eq!(
        gpu.resources.group_creates, groups,
        "a steady frame creates no bind groups"
    );
}

/// One triangle in a scratch glTF, so a frame has a draw to arena into.
struct TriangleAsset {
    root: std::path::PathBuf,
}

impl TriangleAsset {
    fn new(tag: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("bifrost-frame-ring-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch directory");

        let mut bin = Vec::new();
        let positions: [[f32; 3]; 6] = [
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, -2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        for position in positions {
            for value in position {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }
        for _ in positions {
            for value in [0.0f32, 0.0, 1.0] {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }
        for index in [0u16, 1, 2, 3, 4, 5] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        std::fs::write(root.join("tri.bin"), &bin).expect("triangle buffer");

        let gltf = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Tri", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ {
      "attributes": { "POSITION": 0, "NORMAL": 1 },
      "indices": 2
    } ]
  } ],
  "buffers": [ { "uri": "tri.bin", "byteLength": 156 } ],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 72, "target": 34962 },
    { "buffer": 0, "byteOffset": 72, "byteLength": 72, "target": 34962 },
    { "buffer": 0, "byteOffset": 144, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3",
      "min": [ -2.0, -2.0, 0.0 ], "max": [ 2.0, 2.0, 0.0 ]
    },
    { "bufferView": 1, "componentType": 5126, "count": 6, "type": "VEC3" },
    { "bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR" }
  ]
}"#;
        std::fs::write(root.join("tri.gltf"), gltf).expect("triangle glTF");
        Self { root }
    }

    fn load(&self) -> (Assets, AssetId) {
        let mut assets = Assets::new(&self.root);
        let id = assets.load("tri.gltf").expect("load triangle");
        (assets, id)
    }
}

impl Drop for TriangleAsset {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A moving camera touches every per-frame slot, and the frame ring waits for
/// the oldest submission before reusing one.
#[test]
fn a_moving_camera_waits_for_the_oldest_frame_and_rotates_slots() {
    let scratch = TriangleAsset::new("moving-camera");
    let (assets, id) = scratch.load();

    let mut gpu = FakeGpu::new();
    let mut renderer = renderer_for(&mut gpu);
    let jobs = grind::Jobs::new();

    let mut scene = meats::Scene::new();
    let mesh = scene.spawn("tri");
    assert!(scene.insert(
        mesh,
        meats::MeshRenderer {
            mesh: Some(id),
            ..meats::MeshRenderer::default()
        }
    ));
    renderer
        .ensure_mesh(&mut gpu, &assets, id)
        .expect("upload")
        .expect("the triangle has a mesh");

    let mut camera = editor_camera();
    let depth = gpu.info().frames_in_flight as usize;

    // Slots are fresh for the first `depth` frames; only the lap after that
    // waits, and it must free nothing before the wait.
    for frame in 0..depth {
        camera.set_yaw(0.6 + frame as f32 * 0.2);
        renderer
            .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
            .unwrap_or_else(|error| panic!("frame {frame}: {error}"));
        assert!(
            gpu.waited.is_empty(),
            "frame {frame} waited for a slot that had no submission"
        );
        assert!(
            gpu.resources.buffer_destroys.is_empty(),
            "frame {frame} freed a slot while a previous frame might read it"
        );
        assert!(
            gpu.resources.buffer_creates > 0,
            "the first lap allocates each slot"
        );
    }

    // Frame `depth` reuses slot zero, which frame zero wrote.
    camera.set_yaw(0.6 + depth as f32 * 0.2);
    renderer
        .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
        .expect("the lap after the ring");
    assert_eq!(
        gpu.waited,
        vec![1],
        "the reused slot waited on frame zero's token, not the newest"
    );
    assert!(
        !gpu.resources.buffer_destroys.is_empty(),
        "reusing the slot released the previous frame's sets after the wait"
    );

    // Two ringed writes per frame: the draw arena, then the palette. Slots
    // 0..depth are distinct, then slot 0 comes back: both were ringed rather
    // than refilled in place.
    let writes: Vec<Buffer> = gpu
        .resources
        .buffer_writes
        .iter()
        .map(|(buffer, _, _)| *buffer)
        .collect();
    assert_eq!(writes.len(), (depth + 1) * 2, "arena and palette per frame");
    let (pairs, _) = writes.as_chunks::<2>();
    for pair in pairs {
        assert_ne!(pair[0], pair[1], "arena and palette shared a buffer");
    }
    let arena: Vec<Buffer> = pairs.iter().map(|pair| pair[0]).collect();
    let palette: Vec<Buffer> = pairs.iter().map(|pair| pair[1]).collect();
    for pair in arena[..depth].windows(2) {
        assert_ne!(pair[0], pair[1], "two frames shared an arena slot");
    }
    for pair in palette[..depth].windows(2) {
        assert_ne!(pair[0], pair[1], "two frames shared a palette slot");
    }
    assert_eq!(
        arena[depth], arena[0],
        "the lap after the ring reused the first arena slot"
    );
    assert_eq!(
        palette[depth], palette[0],
        "the lap after the ring reused the first palette slot"
    );
}

/// The hand-written atlas, with a zero image so the pass's upload has bytes.
fn overlay_atlas() -> bifrost::font::Atlas {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest
        .ancestors()
        .map(|ancestor| ancestor.join("assets/fonts/test-atlas.json"))
        .find(|candidate| candidate.is_file())
        .expect("the atlas fixture sits above the crate");
    let mut atlas = bifrost::font::Atlas::parse(
        &std::fs::read_to_string(&path).expect("the atlas fixture is readable"),
    )
    .expect("the atlas fixture parses");
    atlas.rgba8 = vec![0; atlas.width as usize * atlas.height as usize * 4];
    atlas
}

/// A queued HUD line is laid out by `edit`, and the frame then declares the text
/// pass that draws it: the producer path a game uses.
#[test]
fn a_pushed_hud_declares_the_text_pass() {
    let mut resources = FakeResources::new();
    let overlay = SpriteTextOverlay::new(
        &mut resources,
        chassis::TextureFormat::Rgba8Srgb,
        overlay_atlas(),
    )
    .expect("both 2D pipelines build against a fake device");
    assert!(overlay.is_empty(), "nothing queued yet");

    overlay.frame().push_text(
        meats::Text {
            text: "AV".to_owned(),
            ..meats::Text::default()
        },
        glam::Mat4::IDENTITY,
    );
    assert!(
        !overlay.is_empty(),
        "a queued run makes the overlay non-empty before edit"
    );

    // The frame's order: begin clears the last frame, edit lays the queue out.
    overlay.begin();
    overlay.edit();
    assert!(!overlay.is_empty(), "the run survives begin/edit");

    let mut graph = RenderGraph::new();
    let scene_color =
        graph.import_texture("scene.color", ImageState::Undefined, ImageState::ShaderRead);
    let scene_depth = graph.import_texture(
        "scene.depth",
        ImageState::Undefined,
        ImageState::DepthAttachment,
    );
    let shadow = graph.import_texture("shadow.map", ImageState::Undefined, ImageState::ShaderRead);
    let ssao = graph.import_texture("ssao.ao", ImageState::Undefined, ImageState::ShaderRead);
    let backbuffer =
        graph.import_texture("backbuffer", ImageState::Undefined, ImageState::ShaderRead);
    let targets = FrameTargets {
        scene_color,
        scene_depth,
        shadow,
        ssao,
        backbuffer,
    };
    overlay.declare(&mut graph, &targets);
    graph
        .pass(PASS_COMPOSITE)
        .color(backbuffer, [0.0, 0.0, 0.0, 1.0])
        .read_texture(scene_color);

    let compiled = graph.compile().expect("the overlay's frame compiles");
    assert!(
        compiled.find(PASS_TEXT).is_some(),
        "a non-empty HUD declares the text pass: {}",
        graph.dump()
    );
}
