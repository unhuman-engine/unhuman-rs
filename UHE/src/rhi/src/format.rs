use std::ops::BitOr;

/// Formats the engine is willing to talk about.
///
/// Deliberately not a mirror of every Vulkan format: this is the set the
/// renderer can actually author and validate against. A second backend maps
/// its own format table onto these.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum TextureFormat {
    Undefined,
    R8Unorm,
    R8Srgb,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8Srgb,
    Bgra8Unorm,
    Bgra8Srgb,
    R16Float,
    Rgba16Float,
    Rgba16Uint,
    R32Float,
    R32G32Float,
    R32G32B32Float,
    Rgba32Float,
    R11b11G11B10Float,
    Depth16,
    Depth24,
    Depth32Float,
    Depth24Stencil8,
}

impl TextureFormat {
    pub fn is_depth(self) -> bool {
        matches!(
            self,
            TextureFormat::Depth16
                | TextureFormat::Depth24
                | TextureFormat::Depth32Float
                | TextureFormat::Depth24Stencil8
        )
    }

    pub fn is_srgb(self) -> bool {
        matches!(
            self,
            TextureFormat::R8Srgb | TextureFormat::Rgba8Srgb | TextureFormat::Bgra8Srgb
        )
    }

    pub fn bytes_per_texel(self) -> Option<u32> {
        Some(match self {
            TextureFormat::Undefined => return None,
            TextureFormat::R8Unorm | TextureFormat::R8Srgb | TextureFormat::Depth16 => 1,
            TextureFormat::Rg8Unorm | TextureFormat::R16Float | TextureFormat::Depth24 => 2,
            TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8Srgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8Srgb
            | TextureFormat::R11b11G11B10Float
            | TextureFormat::Depth32Float
            | TextureFormat::Depth24Stencil8 => 4,
            TextureFormat::R32Float => 4,
            // Vertex formats, and they are here rather than in a second enum
            // because Vulkan uses one format table for both. `Rg32Float` and
            // `Rgb32Float` are the same names wgpu uses.
            TextureFormat::R32G32Float => 8,
            TextureFormat::R32G32B32Float => 12,
            TextureFormat::Rgba16Float | TextureFormat::Rgba16Uint => 8,
            TextureFormat::Rgba32Float => 16,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SampleCount {
    X1,
    X2,
    X4,
    X8,
}

/// Buffer usage flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BufferUsage(pub u32);

impl BufferUsage {
    pub const NONE: BufferUsage = BufferUsage(0);
    pub const VERTEX: BufferUsage = BufferUsage(1 << 0);
    pub const INDEX: BufferUsage = BufferUsage(1 << 1);
    pub const UNIFORM: BufferUsage = BufferUsage(1 << 2);
    pub const STORAGE: BufferUsage = BufferUsage(1 << 3);
    pub const TRANSFER_SRC: BufferUsage = BufferUsage(1 << 4);
    pub const TRANSFER_DST: BufferUsage = BufferUsage(1 << 5);
    pub const INDIRECT: BufferUsage = BufferUsage(1 << 6);
    /// The CPU may map this buffer to read it.
    ///
    /// Needed by nothing on Vulkan and everything on wgpu, which requires
    /// `MAP_READ` at creation and refuses the map otherwise: `Usage flags COPY_SRC
    /// | COPY_DST of Buffer do not contain required usage flags MAP_READ`. Vulkan
    /// reads back by mapping host-visible memory, so it ignores this flag.
    pub const MAP_READ: BufferUsage = BufferUsage(1 << 7);

    pub fn contains(self, other: BufferUsage) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for BufferUsage {
    type Output = BufferUsage;
    fn bitor(self, rhs: BufferUsage) -> BufferUsage {
        BufferUsage(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for BufferUsage {
    fn bitor_assign(&mut self, rhs: BufferUsage) {
        self.0 |= rhs.0;
    }
}

/// Texture usage flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TextureUsage(pub u32);

impl TextureUsage {
    pub const NONE: TextureUsage = TextureUsage(0);
    pub const SAMPLED: TextureUsage = TextureUsage(1 << 0);
    pub const STORAGE: TextureUsage = TextureUsage(1 << 1);
    pub const COLOR_ATTACHMENT: TextureUsage = TextureUsage(1 << 2);
    pub const DEPTH_ATTACHMENT: TextureUsage = TextureUsage(1 << 3);
    pub const TRANSFER_SRC: TextureUsage = TextureUsage(1 << 4);
    pub const TRANSFER_DST: TextureUsage = TextureUsage(1 << 5);

    pub fn contains(self, other: TextureUsage) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for TextureUsage {
    type Output = TextureUsage;
    fn bitor(self, rhs: TextureUsage) -> TextureUsage {
        TextureUsage(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for TextureUsage {
    fn bitor_assign(&mut self, rhs: TextureUsage) {
        self.0 |= rhs.0;
    }
}
