//! The 2D sprite batch.
//!
//! One texture at a time. A texture change starts the next draw, so a draw call
//! binds exactly the texture its vertices were recorded against.
//!
//! `batches` is complete after every `push_*`, not only after a `flush`: a caller
//! that pushes and then asks what to draw sees all of it. `flush` ends the draw in
//! progress, which is how a caller asks for two draws of one texture.
//!
//! Vertex positions transform the unit quad with the caller's `Mat4`; the index
//! pattern fixes the quad's winding.

use glam::{Mat4, Vec2, Vec3, Vec4};

use chassis::{
    BindGroup, BindingDesc, BlendState, Buffer, BufferDesc, BufferUsage, ColorTargetState,
    CommandList, DepthState, DescriptorKind, DescriptorSetLayoutDesc, ImageView, IndexFormat,
    Pipeline, PipelineDesc, PrimitiveState, Resources, SampleCount, Sampler, ShaderStage,
    ShaderStageDesc, StageMask, Texture, TextureFormat, VertexAttribute, VertexLayout,
};

pub const SPRITE_WGSL: &str = include_str!("../assets/shaders/sprite.wgsl");

/// A unit quad about the origin, counter-clockwise from the bottom left.
const QUAD_POSITIONS: [[f32; 3]; 4] = [
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
];

const QUAD_UV: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// One sprite vertex. No texture index: a batch holds one texture, so it would be
/// a constant 0. No entity id either; the RHI has no integer colour target for it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub tiling_factor: f32,
}

/// One indexed draw: every quad between two texture changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawBatch {
    /// The texture bound for this draw. `None` is an untextured quad; the pass
    /// binds a 1x1 white texel for it.
    pub texture: Option<Texture>,
    /// Indices, not bytes: `record` converts, because that is the one place the
    /// unit is decided.
    pub index_offset: u32,
    pub index_count: u32,
}

/// Per-frame draw counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Statistics {
    pub draw_calls: u32,
    pub quad_count: u32,
}

impl Statistics {
    pub fn total_vertex_count(&self) -> u32 {
        self.quad_count * 4
    }

    pub fn total_index_count(&self) -> u32 {
        self.quad_count * 6
    }
}

/// `translate(position) * scale(size)`, the transform a position/size quad uses.
pub fn quad_transform(position: Vec3, size: Vec2) -> Mat4 {
    Mat4::from_translation(position) * Mat4::from_scale(Vec3::new(size.x, size.y, 1.0))
}

#[derive(Clone, Debug, Default)]
pub struct Batch2D {
    vertices: Vec<SpriteVertex>,
    indices: Vec<u32>,
    texture: Option<Texture>,
    /// Whether the last entry of `batches` is still taking quads.
    open: bool,
    batches: Vec<DrawBatch>,
    stats: Statistics,
}

impl Batch2D {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-size for `quads`, so a frame's first batch does not grow its way up.
    pub fn with_capacity(quads: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(quads * 4),
            indices: Vec::with_capacity(quads * 6),
            ..Self::default()
        }
    }

    /// Start a frame. `Statistics` survives until `reset_stats`.
    pub fn begin(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.batches.clear();
        self.texture = None;
        self.open = false;
    }

    pub fn reset_stats(&mut self) {
        self.stats = Statistics::default();
    }

    pub fn stats(&self) -> Statistics {
        self.stats
    }

    pub fn push_quad(&mut self, transform: Mat4, color: Vec4) {
        self.push_sprite(None, transform, color, 1.0);
    }

    pub fn push_sprite(
        &mut self,
        texture: Option<Texture>,
        transform: Mat4,
        color: Vec4,
        tiling_factor: f32,
    ) {
        self.push_sprite_uv(texture, transform, color, tiling_factor, QUAD_UV);
    }

    /// The unit quad carrying caller-supplied atlas coordinates, in the same
    /// bottom-left, bottom-right, top-right, top-left order as [`QUAD_UV`].
    pub fn push_sprite_uv(
        &mut self,
        texture: Option<Texture>,
        transform: Mat4,
        color: Vec4,
        tiling_factor: f32,
        uv: [[f32; 2]; 4],
    ) {
        self.open_draw(texture);

        let base = self.vertices.len() as u32;
        for corner in 0..QUAD_POSITIONS.len() {
            let local = Vec4::new(
                QUAD_POSITIONS[corner][0],
                QUAD_POSITIONS[corner][1],
                QUAD_POSITIONS[corner][2],
                1.0,
            );
            let position = transform * local;
            self.vertices.push(SpriteVertex {
                position: [position.x, position.y, position.z],
                color: color.to_array(),
                uv: uv[corner],
                tiling_factor,
            });
        }
        for offset in [0u32, 1, 2, 2, 3, 0] {
            self.indices.push(base + offset);
        }

        if let Some(batch) = self.batches.last_mut() {
            batch.index_count += 6;
        }
        self.stats.quad_count += 1;
    }

    /// Ends the draw in progress, so the next `push_*` starts a new one even at
    /// the same texture. Never needed for `batches` to be complete.
    pub fn flush(&mut self) {
        self.open = false;
    }

    /// Bind the pipeline, the buffers and the globals, then draw every batch.
    ///
    /// `texture_group` resolves the texture a batch was recorded against to the
    /// bind group for set 0. It is the pass's, not this type's, because it owns
    /// the sampler and the group cache and only it knows when to build them.
    pub fn record(
        &self,
        list: &mut dyn CommandList,
        pipeline: &Pipeline,
        vertex_buffer: &Buffer,
        index_buffer: &Buffer,
        globals: &BindGroup,
        texture_group: impl Fn(Option<Texture>) -> BindGroup,
    ) {
        if self.batches.is_empty() {
            return;
        }

        list.bind_pipeline(pipeline);
        list.bind_vertex_buffer(0, vertex_buffer, 0);
        list.bind_group(GLOBALS_SET, globals);
        for batch in &self.batches {
            let group = texture_group(batch.texture);
            list.bind_group(TEXTURE_SET, &group);
            // The RHI's `draw_indexed` has no first-index argument, so a batch's
            // offset is the index buffer's.
            list.bind_index_buffer(index_buffer, batch.index_offset as u64 * INDEX_SIZE);
            list.draw_indexed(batch.index_count, 1);
        }
    }

    pub fn batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    pub fn vertices(&self) -> &[SpriteVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    pub fn index_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }

    /// The draw a quad belongs to. A texture change ends the last one and starts
    /// another, which keeps `batches` complete at every point: a caller that never
    /// calls `flush` still has every draw it pushed, and one that asks for fewer
    /// draws can say so with `flush`.
    fn open_draw(&mut self, texture: Option<Texture>) {
        if self.open && self.texture == texture {
            return;
        }
        self.texture = texture;
        self.batches.push(DrawBatch {
            texture,
            index_offset: self.indices.len() as u32,
            index_count: 0,
        });
        self.stats.draw_calls += 1;
        self.open = true;
    }
}

/// The texture and its sampler: set 0 of both 2D pipelines.
///
/// Separate image and sampler bindings, not a combined one, because wgpu has no
/// combined descriptor at all.
pub fn texture_layout(label: &str) -> DescriptorSetLayoutDesc {
    let fragment = StageMask::of(ShaderStage::Fragment);
    DescriptorSetLayoutDesc::new(label)
        .with(BindingDesc::new(
            TEXTURE_IMAGE_BINDING,
            DescriptorKind::SampledImage,
            fragment,
        ))
        .with(BindingDesc::new(
            TEXTURE_SAMPLER_BINDING,
            DescriptorKind::Sampler,
            fragment,
        ))
}

/// The block every 2D draw of a frame shares: set 1.
pub fn globals_layout(label: &str) -> DescriptorSetLayoutDesc {
    let stages = StageMask::of(ShaderStage::Vertex).union(StageMask::of(ShaderStage::Fragment));
    DescriptorSetLayoutDesc::new(label).with(BindingDesc::new(
        GLOBALS_BINDING,
        DescriptorKind::UniformBuffer,
        stages,
    ))
}

/// The bind group set 0 expects: the texture at binding 0, its sampler at 1.
pub fn texture_group(
    resources: &mut dyn Resources,
    layout: &DescriptorSetLayoutDesc,
    texture: Texture,
    sampler: Sampler,
) -> Result<BindGroup, chassis::Error> {
    let view: ImageView = resources.create_image_view(&texture)?;
    resources.create_bind_group(
        layout,
        &[],
        &[(TEXTURE_IMAGE_BINDING, view)],
        &[(TEXTURE_SAMPLER_BINDING, sampler)],
    )
}

/// The two set numbers, here because the shader's `@group` and the pass's
/// `bind_group` call are the two places they appear and they have to agree.
pub(crate) const TEXTURE_SET: u32 = 0;
pub(crate) const GLOBALS_SET: u32 = 1;
pub(crate) const TEXTURE_IMAGE_BINDING: u32 = 0;
pub(crate) const TEXTURE_SAMPLER_BINDING: u32 = 1;
pub(crate) const GLOBALS_BINDING: u32 = 0;

/// `u32` indices, which is what `PipelineDesc::index_format` is set to.
pub(crate) const INDEX_SIZE: u64 = std::mem::size_of::<u32>() as u64;

pub fn sprite_vertex_layout() -> VertexLayout {
    VertexLayout {
        stride: std::mem::size_of::<SpriteVertex>() as u32,
        attributes: vec![
            VertexAttribute {
                location: 0,
                format: TextureFormat::R32G32B32Float,
                offset: std::mem::offset_of!(SpriteVertex, position) as u32,
            },
            VertexAttribute {
                location: 1,
                format: TextureFormat::Rgba32Float,
                offset: std::mem::offset_of!(SpriteVertex, color) as u32,
            },
            VertexAttribute {
                location: 2,
                format: TextureFormat::R32G32Float,
                offset: std::mem::offset_of!(SpriteVertex, uv) as u32,
            },
            VertexAttribute {
                location: 3,
                format: TextureFormat::R32Float,
                offset: std::mem::offset_of!(SpriteVertex, tiling_factor) as u32,
            },
        ],
    }
}

/// What went wrong building a 2D pipeline: the shader it compiles or the RHI it
/// asks for.
#[derive(Debug)]
pub enum PipelineError {
    Wgsl(forge::wgsl::WgslError),
    Rhi(chassis::Error),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Wgsl(error) => write!(f, "{error}"),
            PipelineError::Rhi(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<forge::wgsl::WgslError> for PipelineError {
    fn from(error: forge::wgsl::WgslError) -> Self {
        PipelineError::Wgsl(error)
    }
}

impl From<chassis::Error> for PipelineError {
    fn from(error: chassis::Error) -> Self {
        PipelineError::Rhi(error)
    }
}

/// A 2D pipeline and everything a caller needs to bind it: the set layouts the
/// bind groups must be built from, so the two cannot disagree.
pub struct QuadPipeline {
    pipeline: Pipeline,
    shader: chassis::Shader,
    texture_layout: DescriptorSetLayoutDesc,
    globals_layout: DescriptorSetLayoutDesc,
}

impl QuadPipeline {
    /// `source` is a WGSL file with `vs_main` and `fs_main` in one module.
    pub fn build(
        resources: &mut dyn Resources,
        label: &str,
        source: &str,
        vertex_layout: VertexLayout,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
    ) -> Result<Self, PipelineError> {
        let spirv = forge::wgsl::to_spirv(source, label)?;
        let shader = resources.create_shader(label, &spirv)?;

        let texture_layout = texture_layout(&format!("{label}.texture"));
        let globals_layout = globals_layout(&format!("{label}.globals"));
        let pipeline = resources.create_pipeline(PipelineDesc {
            label: label.to_owned(),
            shaders: vec![
                ShaderStageDesc {
                    stage: ShaderStage::Vertex,
                    shader,
                },
                ShaderStageDesc {
                    stage: ShaderStage::Fragment,
                    shader,
                },
            ],
            set_layouts: vec![texture_layout.clone(), globals_layout.clone()],
            vertex: vertex_layout,
            index_format: Some(IndexFormat::U32),
            // Culling off: a 2D quad's winding is only meaningful once the
            // view-projection is known.
            primitive: PrimitiveState {
                cull_mode: 0,
                ..PrimitiveState::default()
            },
            depth: DepthState {
                format: depth_format,
                write_enabled: depth_format.is_some(),
                // `1` is the value the sibling 3D slice uses; both backends
                // hardcode less-or-equal and ignore this.
                compare: 1,
            },
            blend: BlendState { enabled: true },
            color_targets: vec![ColorTargetState {
                format: color_format,
                blend: true,
                write_mask: 0xF,
            }],
            samples: SampleCount::X1,
        })?;

        Ok(QuadPipeline {
            pipeline,
            shader,
            texture_layout,
            globals_layout,
        })
    }

    pub fn sprite(
        resources: &mut dyn Resources,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
    ) -> Result<Self, PipelineError> {
        Self::build(
            resources,
            "sprite",
            SPRITE_WGSL,
            sprite_vertex_layout(),
            color_format,
            depth_format,
        )
    }

    pub fn pipeline(&self) -> Pipeline {
        self.pipeline
    }

    pub fn shader(&self) -> chassis::Shader {
        self.shader
    }

    pub fn texture_layout(&self) -> &DescriptorSetLayoutDesc {
        &self.texture_layout
    }

    pub fn globals_layout(&self) -> &DescriptorSetLayoutDesc {
        &self.globals_layout
    }

    /// The globals holder for this pipeline, built from the layout it was
    /// created with.
    pub fn globals(&self) -> Globals {
        Globals::new(self.globals_layout.clone())
    }
}

/// Set 1: one uniform block, uploaded per frame.
///
/// NOTE: the buffer and its bind group are re-created on every upload, because
/// `Resources` has no update-in-place call; a frame costs one small allocation.
pub struct Globals {
    layout: DescriptorSetLayoutDesc,
    buffer: Option<Buffer>,
    group: Option<BindGroup>,
}

impl Globals {
    pub fn new(layout: DescriptorSetLayoutDesc) -> Self {
        Self {
            layout,
            buffer: None,
            group: None,
        }
    }

    pub fn layout(&self) -> &DescriptorSetLayoutDesc {
        &self.layout
    }

    /// Replaces the contents and returns the group to bind at set 1.
    pub fn upload<T: bytemuck::Pod>(
        &mut self,
        resources: &mut dyn Resources,
        value: &T,
    ) -> Result<BindGroup, chassis::Error> {
        self.destroy(resources);
        let bytes = bytemuck::bytes_of(value);
        let buffer = resources.create_buffer(
            BufferDesc::new(bytes.len() as u64, BufferUsage::UNIFORM),
            Some(bytes),
        )?;
        let group =
            resources.create_bind_group(&self.layout, &[(GLOBALS_BINDING, buffer, 0)], &[], &[])?;
        self.buffer = Some(buffer);
        self.group = Some(group);
        Ok(group)
    }

    pub fn group(&self) -> Option<BindGroup> {
        self.group
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources) {
        if let Some(group) = self.group.take() {
            resources.destroy_bind_group(group);
        }
        if let Some(buffer) = self.buffer.take() {
            resources.destroy_buffer(buffer);
        }
    }
}

/// The globals the sprite shader reads: one view-projection.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteGlobals {
    pub view_projection: [[f32; 4]; 4],
}

impl SpriteGlobals {
    pub fn new(view_projection: Mat4) -> Self {
        Self {
            view_projection: view_projection.to_cols_array_2d(),
        }
    }
}

/// A vertex and index buffer sized for the batch's current contents.
///
/// NOTE: re-created once per frame, because `Resources` has no update-in-place
/// call; a drawn batch costs one allocation per frame.
pub fn create_buffers(
    resources: &mut dyn Resources,
    batch: &Batch2D,
) -> chassis::Result<(Buffer, Buffer)> {
    let vertex = resources.create_buffer(
        BufferDesc::new(batch.vertex_bytes().len() as u64, BufferUsage::VERTEX),
        Some(batch.vertex_bytes()),
    )?;
    let index = resources.create_buffer(
        BufferDesc::new(batch.index_bytes().len() as u64, BufferUsage::INDEX),
        Some(batch.index_bytes()),
    )?;
    Ok((vertex, index))
}
