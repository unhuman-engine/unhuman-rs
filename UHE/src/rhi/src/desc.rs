use crate::format::{BufferUsage, SampleCount, TextureFormat, TextureUsage};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }

    pub const fn new_3d(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
}

impl BufferDesc {
    pub const fn new(size: u64, usage: BufferUsage) -> Self {
        Self { size, usage }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TextureDesc {
    pub extent: Extent,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    pub mip_levels: u32,
    pub samples: SampleCount,
    /// Initial contents. The backend is free to skip the copy when `None`.
    pub initial_data: Option<Vec<u8>>,
}

impl TextureDesc {
    pub fn new_2d(extent: Extent, format: TextureFormat, usage: TextureUsage) -> Self {
        Self {
            extent,
            format,
            usage,
            mip_levels: 1,
            samples: SampleCount::X1,
            initial_data: None,
        }
    }

    pub fn with_mips(mut self, levels: u32) -> Self {
        self.mip_levels = levels.max(1);
        self
    }

    pub fn with_samples(mut self, samples: SampleCount) -> Self {
        self.samples = samples;
        self
    }

    /// Bytes needed for mip level 0. Mips are derived, not stored.
    pub fn level0_size(&self) -> Option<u64> {
        let bytes = self.format.bytes_per_texel()? as u64;
        let texels =
            self.extent.width as u64 * self.extent.height as u64 * self.extent.depth.max(1) as u64;
        Some(texels * bytes)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum IndexFormat {
    U16,
    U32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: TextureFormat,
    pub offset: u32,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct VertexLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PrimitiveState {
    pub topology: u32,
    pub cull_mode: u32,
    pub front_face: u32,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: 4,   // triangle list
            cull_mode: 1,  // back
            front_face: 0, // counter-clockwise
        }
    }
}

// Vulkan's viewport transform maps NDC +Y to framebuffer -Y, with no GL-style
// Y-flip. A triangle wound counter-clockwise in NDC (front_face: 0) appears
// clockwise in framebuffer coordinates, so expect to set front_face: 1 somewhere.

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DepthState {
    pub format: Option<TextureFormat>,
    pub write_enabled: bool,
    pub compare: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: bool,
    pub write_mask: u8,
}

impl Default for ColorTargetState {
    fn default() -> Self {
        Self {
            format: TextureFormat::Undefined,
            blend: false,
            write_mask: 0xF,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BlendState {
    pub enabled: bool,
}

/// A complete graphics pipeline description.
///
/// One struct rather than a builder: a second backend either consumes this or
/// does not exist yet, and a builder here would be guessing at the shape of a
/// backend nobody has written.
/// A pipeline stage. Named rather than flagged so a `PipelineDesc` cannot
/// contain the same stage twice, which is legal to spell and illegal to create.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessellationControl,
    TessellationEvaluation,
}

impl ShaderStage {
    /// Bits for a shader visibility mask.
    pub const fn bit(self) -> u32 {
        match self {
            ShaderStage::Vertex => 1 << 0,
            ShaderStage::Fragment => 1 << 1,
            ShaderStage::Compute => 1 << 2,
            ShaderStage::Geometry => 1 << 3,
            ShaderStage::TessellationControl => 1 << 4,
            ShaderStage::TessellationEvaluation => 1 << 5,
        }
    }

    /// Stages that can appear in a graphics pipeline. A compute stage in one is
    /// a `vkCreateGraphicsPipelines` error reported as a driver problem.
    pub fn is_graphics(self) -> bool {
        !matches!(self, ShaderStage::Compute)
    }
}

/// Which stages may read one binding.
///
/// A mask rather than a single stage, because one buffer is commonly written by
/// the vertex stage and read by the fragment stage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct StageMask(pub u32);

impl StageMask {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(0b11_1111);

    pub const fn of(stage: ShaderStage) -> Self {
        Self(stage.bit())
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, stage: ShaderStage) -> bool {
        self.0 & stage.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// What one binding points at.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DescriptorKind {
    UniformBuffer,
    /// Read-only storage. What a shader may read, and what a vertex-stage
    /// binding can use without `VERTEX_WRITABLE_STORAGE`.
    StorageBuffer,
    /// Storage a shader writes.
    ///
    /// Distinct from [`Self::StorageBuffer`] because wgpu validates a binding
    /// against the shader's declared access: a read-only layout on a writable
    /// buffer is rejected at pipeline creation. Vulkan uses one descriptor type
    /// for both.
    StorageBufferReadWrite,
    /// Image and sampler declared as two bindings. Vulkan requires
    /// `SAMPLED_IMAGE` and `SAMPLER` to be separate bindings when they are
    /// separate; declaring them as one is `CombinedImageSampler`.
    SampledImage,
    /// A depth texture read as a texture rather than an attachment.
    ///
    /// wgpu gives a depth view `TextureSampleType::Depth`, and a `Float` binding
    /// against one is a layout mismatch the driver refuses; Vulkan takes the same
    /// `SAMPLED_IMAGE` descriptor either way and checks the layout at barrier
    /// time instead.
    SampledDepth,
    Sampler,
    CombinedImageSampler,
    StorageImage,
    /// Subpass input. The backend already had it; the enum was missing the case,
    /// so a shader could declare a layout the RHI could not describe.
    InputAttachment,
}

/// How a sampler filters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Filter {
    Nearest,
    #[default]
    Linear,
}

/// What happens outside 0..1.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum AddressMode {
    #[default]
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

/// A texture sampler.
///
/// Filter and address mode only. Anisotropy, compare and border colour are
/// deliberately absent rather than defaulted, because a silently different
/// sampler is worse than one that cannot be asked for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SamplerDesc {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mip_filter: Filter,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
}

impl SamplerDesc {
    /// Nearest, no mips: what a pixel-art texture wants and what a
    /// non-mipmapped texture needs.
    pub const fn nearest() -> Self {
        Self {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mip_filter: Filter::Nearest,
            address_u: AddressMode::Repeat,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::Repeat,
        }
    }

    /// Trilinear with repeat. The default for a mipmapped colour texture.
    pub const fn trilinear() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mip_filter: Filter::Linear,
            address_u: AddressMode::Repeat,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::Repeat,
        }
    }

    /// Linear, clamped. What a render target wants: sampling outside it should
    /// give the edge pixel, not a wrapped one from the other side of the screen.
    pub const fn clamped() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mip_filter: Filter::Linear,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
        }
    }
}

/// One binding within a set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BindingDesc {
    pub binding: u32,
    pub kind: DescriptorKind,
    /// Array length. 1 for everything but a descriptor array.
    pub count: u32,
    pub stages: StageMask,
}

impl BindingDesc {
    pub fn new(binding: u32, kind: DescriptorKind, stages: StageMask) -> Self {
        Self {
            binding,
            kind,
            count: 1,
            stages,
        }
    }
}

/// One descriptor set's shape.
///
/// A pipeline owns these shapes, not the sets themselves: the layout is fixed at
/// pipeline creation, and the sets are allocated per frame. Getting that
/// backwards is how a set ends up bound to a layout it does not match.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct DescriptorSetLayoutDesc {
    pub label: String,
    pub bindings: Vec<BindingDesc>,
}

impl DescriptorSetLayoutDesc {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            bindings: Vec::new(),
        }
    }

    pub fn with(mut self, binding: BindingDesc) -> Self {
        self.bindings.push(binding);
        self
    }

    /// True when two declarations can share one `VkDescriptorSetLayout`.
    ///
    /// The label is excluded: two layouts that differ only by name are the same.
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.bindings == other.bindings
    }

    /// A cache key with the same exclusions as [`Self::is_compatible`].
    ///
    /// Must agree with `is_compatible`: one dedupes, the other is its cache key,
    /// so a disagreement changes whether a bind group and a pipeline share an
    /// object.
    pub fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.bindings.hash(&mut hasher);
        hasher.finish()
    }
}

/// One stage's module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ShaderStageDesc {
    pub stage: ShaderStage,
    pub shader: crate::handle::Shader,
}

/// A complete compute pipeline description.
///
/// The graphics fields are absent rather than defaulted: a compute pipeline has
/// no vertex input, no rasterisation state and no attachments, and carrying
/// them would invite a backend to read a field that means nothing.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ComputePipelineDesc {
    pub label: String,
    pub shader: crate::handle::Shader,
    /// Descriptor set shapes, indexed by set number. Empty is legal and common.
    pub set_layouts: Vec<DescriptorSetLayoutDesc>,
}

impl ComputePipelineDesc {
    pub fn new(label: impl Into<String>, shader: crate::handle::Shader) -> Self {
        Self {
            label: label.into(),
            shader,
            set_layouts: Vec::new(),
        }
    }

    pub fn with_set(mut self, set: DescriptorSetLayoutDesc) -> Self {
        self.set_layouts.push(set);
        self
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct PipelineDesc {
    pub label: String,
    /// Every stage this pipeline runs.
    ///
    /// A list, not a single handle: Vulkan has no way to compile a vertex and a
    /// fragment module from one file into one module.
    pub shaders: Vec<ShaderStageDesc>,
    /// Descriptor set shapes, indexed by set number. Empty means the pipeline
    /// reads no descriptors, which is legal and common.
    pub set_layouts: Vec<DescriptorSetLayoutDesc>,
    pub vertex: VertexLayout,
    pub index_format: Option<IndexFormat>,
    pub primitive: PrimitiveState,
    pub depth: DepthState,
    pub blend: BlendState,
    pub color_targets: Vec<ColorTargetState>,
    pub samples: SampleCount,
}

#[derive(Clone, PartialEq, Debug)]
pub struct ColorAttachmentDesc {
    /// The attachment. Present because the graph has to name what a pass writes,
    /// not just what shape it is.
    pub texture: crate::Texture,
    pub format: TextureFormat,
    pub load: LoadOp,
    /// Used when `load` is `LoadOp::Clear`. Carried as data, not as a variant,
    /// because the backends take the clear value as a side argument and an enum
    /// payload would make every `Load` construction spell a value it ignores.
    pub clear: [f32; 4],
    pub store: bool,
    /// MSAA resolve target, or `Texture::default()` for none.
    pub resolve: crate::Texture,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoadOp {
    Load,
    Clear,
    /// Valid as a source state on first use of a transient resource.
    DontCare,
}

#[derive(Clone, PartialEq, Debug)]
pub struct DepthAttachmentDesc {
    pub texture: crate::Texture,
    pub format: TextureFormat,
    pub load: LoadOp,
    /// Used when `load` is `LoadOp::Clear`; the far plane, usually.
    pub clear_depth: f32,
    pub store: bool,
}

/// Attachments a pass writes during one `begin_rendering` scope.
///
/// Carries no layouts: transitions and barriers are the render graph's job,
/// derived from the declared read/write sets. A pass gets no barrier API.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct RenderPassDesc {
    pub label: String,
    pub colors: Vec<ColorAttachmentDesc>,
    pub depth: Option<DepthAttachmentDesc>,
}

impl RenderPassDesc {
    pub fn color(format: TextureFormat) -> Self {
        Self {
            label: String::new(),
            colors: vec![ColorAttachmentDesc {
                texture: crate::Texture::null(),
                format,
                load: LoadOp::Clear,
                clear: [0.0; 4],
                store: true,
                resolve: crate::Texture::null(),
            }],
            depth: None,
        }
    }

    pub fn with_depth(mut self, format: TextureFormat) -> Self {
        self.depth = Some(DepthAttachmentDesc {
            texture: crate::Texture::null(),
            format,
            load: LoadOp::Clear,
            clear_depth: 1.0,
            store: true,
        });
        self
    }

    pub fn labeled(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

/// The row pitch a texture-to-buffer copy must use, in bytes.
///
/// wgpu validates the pitch against this alignment
/// (`COPY_BYTES_PER_ROW_ALIGNMENT`) and refuses a tighter one, so a readback of a
/// width that is not a multiple of 64 texels fails validation rather than
/// returning a wrong image. Vulkan has no such rule for a buffer-image copy, and
/// both backends write the same bytes, so the pitch is computed here and used by
/// both: the caller that reads the buffer back strips the padding, and only the
/// caller knows whether it wants to.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// The row pitch for `width` texels of `bytes_per_texel` bytes each.
///
/// At least `width * bytes_per_texel`, rounded up to the copy alignment.
pub const fn copy_row_pitch(width: u32, bytes_per_texel: u32) -> u32 {
    let tight = width * bytes_per_texel;
    tight.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT
}
