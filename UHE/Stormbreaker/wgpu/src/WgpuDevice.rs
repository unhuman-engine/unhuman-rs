//! Adapter and device.
//!
//! ## Why a second backend exists
//!
//! The Vulkan backend is the one with the control: explicit barriers, explicit
//! memory, an explicit command list per pass. That is what the render graph's
//! ordering is designed around, and it is what makes bare-metal speed possible.
//!
//! wgpu is the opposite trade. It tracks resource transitions itself, it has no
//! per-barrier control at all, and it will run the same SPIR-V on Vulkan, Metal
//! and DX12. It is here for the cases ash cannot serve: a driver the Vulkan
//! loader cannot find, a platform where shipping Vulkan is a fight, and CI.
//!
//! ## What the two backends share, and what they do not
//!
//! Shared: `chassis::Gpu` and `chassis::Resources`, the render graph's *ordering*
//! (culling, versioning, and the ordering that makes parallel recording safe),
//! and everything `forge` produces. A shader compiled once runs on both.
//!
//! Not shared: the barrier lowering. `VulkanBarrierEncoder` turns the graph's
//! backend-neutral barriers into `vkCmdPipelineBarrier2` or
//! `vkCmdPipelineBarrier` per the device's `SyncTier`. wgpu has no equivalent --
//! it inserts its own transitions -- so this backend ignores the graph's barrier
//! output entirely and relies on the graph's *ordering* only.
//!
//! That asymmetry is why the graph emits backend-neutral barriers rather than
//! `VkImageMemoryBarrier`. If the compiler wrote Vulkan structs, this crate
//! could not exist.

use std::collections::HashMap;

use chassis::TextureFormat;

/// An adapter, chosen. A backend needs one before it needs a device.
pub struct Adapter {
    inner: wgpu::Adapter,
    index: u32,
}

impl Adapter {
    /// The best adapter available, preferring a discrete GPU.
    ///
    /// Falls back to whatever exists rather than failing. On a CI runner there
    /// is no discrete GPU and refusing to start would make every backend test
    /// skip, which is how a backend silently stops being tested.
    pub fn request(index: u32) -> Option<Self> {
        Self::request_with(index, &wgpu::Instance::default(), None)
    }

    /// An adapter that can present to `surface`, from the instance that made it.
    ///
    /// The surface has to come from the same instance: an adapter from a second one
    /// cannot even be asked whether it can present to it. The windowed path creates
    /// the instance, the surface and the adapter from one value for that reason.
    pub fn request_for_surface(
        index: u32,
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Option<Self> {
        Self::request_with(index, instance, Some(surface))
    }

    fn request_with(
        index: u32,
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Option<Self> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface,
            ..Default::default()
        }));
        // A `Result`, not an `Option`: wgpu 30 reports *why* there is no
        // adapter, and "no compatible surface" and "no driver" are different
        // problems. `ok()` collapses both into `None`, which is the one thing
        // this function must not do -- a backend that silently finds nothing
        // looks exactly like a backend that was never built.
        let adapter = adapter.ok()?;
        Some(Self {
            inner: adapter,
            index,
        })
    }

    /// Every adapter, so a launcher can offer a choice and a test can skip a
    /// backend that is not present instead of failing.
    pub fn enumerate() -> Vec<Self> {
        let instance = wgpu::Instance::default();
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .enumerate()
            .map(|(index, inner)| Self {
                inner,
                index: index as u32,
            })
            .collect()
    }

    pub fn name(&self) -> String {
        self.inner.get_info().name
    }

    /// wgpu's backend enum as a string, for the log and for a test that asserts
    /// *which* backend ran.
    ///
    /// Worth asserting rather than reading: a test that says "wgpu works" while
    /// silently running on the Vulkan backend proves the Vulkan backend works,
    /// which was already known.
    pub fn backend(&self) -> &'static str {
        match self.inner.get_info().backend {
            wgpu::Backend::Vulkan => "vulkan",
            wgpu::Backend::Metal => "metal",
            wgpu::Backend::Dx12 => "dx12",
            wgpu::Backend::Gl => "gl",
            wgpu::Backend::BrowserWebGpu => "webgpu",
            _ => "other",
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn limits(&self) -> wgpu::Limits {
        self.inner.limits()
    }

    pub fn features(&self) -> wgpu::Features {
        self.inner.features()
    }

    /// What the surface offers on this adapter.
    ///
    /// Asked again on every reconfigure rather than kept: a window moved to another
    /// display changes the answer, and `Surface::configure` panics on a format the
    /// surface no longer offers.
    pub fn capabilities(&self, surface: &wgpu::Surface<'_>) -> wgpu::SurfaceCapabilities {
        surface.get_capabilities(&self.inner)
    }

    pub fn create_device(&self, label: &str) -> Option<(Device, Queue)> {
        // A `Result`, collapsed to `None` here: the caller only needs to know
        // whether it got one, and the reason belongs in a log rather than in a
        // return type the renderer would have to handle.
        let requested = pollster::block_on(self.inner.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            // Features are requested *adaptively*: each one is asked for only
            // when the adapter has it.
            //
            // The earlier rule here was "request nothing", which is right for
            // anything optional and wrong for anything load-bearing.
            // `MAPPABLE_PRIMARY_BUFFERS` is load-bearing: without it wgpu refuses
            // to create a buffer that can be read back, and a readback is how
            // both backends' tests prove anything. Requesting it unconditionally
            // would fail device creation on an adapter without it -- which is the
            // worst failure for a test suite, because it passes on the machine
            // you have and fails on the one that matters.
            //
            // `VERTEX_WRITABLE_STORAGE` is deliberately *not* requested: the
            // frame set's storage buffer is bound read-only (see `layout_type_of`),
            // and read-only storage in the vertex stage does not need it. Asking
            // for it would grant the vertex stage the power to write buffers, to
            // make a layout the engine does not have.
            required_features: self.inner.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        }));
        requested.ok().map(|(device, queue)| {
            (
                Device {
                    inner: device,
                    index: self.index,
                    name: self.name(),
                },
                Queue {
                    inner: queue,
                    family: 0,
                },
            )
        })
    }
}

/// The device. Owns every wgpu object, so it is what `Drop` matters on.
///
/// `Clone`, and cheap: a `wgpu::Device` is a handle to a shared object, so
/// cloning is a refcount bump. Every resource registry needs one and passing a
/// borrow would mean the registries could not outlive the caller's device.
#[derive(Clone)]
pub struct Device {
    inner: wgpu::Device,
    index: u32,
    name: String,
}

impl Device {
    pub fn inner(&self) -> &wgpu::Device {
        &self.inner
    }

    pub fn adapter_index(&self) -> u32 {
        self.index
    }

    pub fn adapter_name(&self) -> &str {
        &self.name
    }

    /// The largest 2D texture this device accepts.
    ///
    /// Reported through the RHI's `GpuInfo::max_texture_size`, so a renderer can
    /// size a shadow map once instead of asking.
    pub fn max_texture_size(&self) -> u32 {
        self.inner.limits().max_texture_dimension_2d
    }
}

/// `Clone`, and cheap: a `wgpu::Queue` is a handle to a shared object.
///
/// The executor needs its own, because it outlives the call that built it and
/// `wgpu::Queue` is not `Clone`. Rather than reach into private fields to make a
/// copy, the type says what it is.
#[derive(Clone)]
pub struct Queue {
    inner: wgpu::Queue,
    family: u32,
}

impl Queue {
    pub fn inner(&self) -> &wgpu::Queue {
        &self.inner
    }

    /// The queue family index. wgpu exposes one queue and does not number them,
    /// so this is a constant rather than a lie about a concept the API lacks.
    pub fn family(&self) -> u32 {
        self.family
    }
}

/// Backend-nothing, for headless.
///
/// wgpu's distinguishing practical property: a device with no surface at all is
/// ordinary. That is why `Gpu::swapchain` is an `Option`, and why this backend
/// can be tested anywhere a driver exists -- which is why its CI is real and the
/// Vulkan backend's Windows job is not.
pub fn headless(index: u32) -> Option<(Device, Queue)> {
    Adapter::request(index)?.create_device("stormglass-headless")
}

/// What the renderer needs to know, filled in from an adapter without creating a
/// device.
///
/// Lets a launcher list backends without paying for device creation, and lets a
/// test assert what *would* be chosen on a machine that has no such device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendInfo {
    pub index: u32,
    pub name: String,
    pub backend: &'static str,
    pub max_texture_size: u32,
}

/// Every backend available, for a launcher or a diagnostic.
pub fn enumerate_backends() -> Vec<BackendInfo> {
    Adapter::enumerate()
        .into_iter()
        .map(|adapter| BackendInfo {
            index: adapter.index,
            name: adapter.name(),
            backend: adapter.backend(),
            max_texture_size: adapter.limits().max_texture_dimension_2d,
        })
        .collect()
}

/// wgpu has no equivalent of a format enum with the RHI's names, so this is the
/// one place the two vocabularies meet.
///
/// wgpu's `TextureFormat` is a superset of the RHI's -- it has sRGB variants and
/// compressed formats the RHI has no name for -- so the mapping is
/// many-to-one in that direction and one-to-many here. An unsupported format is
/// `None` rather than a default: substituting a format silently would render
/// into an image nobody asked for.
pub fn to_wgpu_format(format: TextureFormat) -> Option<wgpu::TextureFormat> {
    Some(match format {
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        // wgpu spells the sRGB variants `...UnormSrgb`. There is no `R8Srgb` at
        // all, and that is not a gap in this mapping: single-channel sRGB is not
        // a thing either API has. `R8Srgb` therefore maps to `None` rather than
        // to the linear format, which would render the right image with the
        // wrong gamma -- an error that looks like a colour bug and is not.
        TextureFormat::Rgba8Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8Srgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        TextureFormat::R16Float => wgpu::TextureFormat::R16Float,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Rgba16Uint => wgpu::TextureFormat::Rgba16Uint,
        TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        TextureFormat::R32G32Float => wgpu::TextureFormat::Rg32Float,
        // wgpu has no 3-channel 32-bit float, so a caller asking for one gets
        // `None` rather than a 4-channel texture it did not ask for.
        TextureFormat::R32G32B32Float => return None,
        TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        TextureFormat::R11b11G11B10Float => wgpu::TextureFormat::Rg11b10Ufloat,
        TextureFormat::Depth16 => wgpu::TextureFormat::Depth16Unorm,
        // The same choice the Vulkan backend makes, for the same reason:
        // `Depth24` is not a real format in either API, and `Depth24Plus` is
        // the portable spelling of "a depth buffer, implementation's choice".
        TextureFormat::Depth24 => wgpu::TextureFormat::Depth24Plus,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        TextureFormat::Depth24Stencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        TextureFormat::R8Srgb | TextureFormat::Undefined => return None,
    })
}

/// The reverse mapping, for a swapchain whose format wgpu chose.
pub fn from_wgpu_format(format: wgpu::TextureFormat) -> Option<TextureFormat> {
    Some(match format {
        wgpu::TextureFormat::R8Unorm => TextureFormat::R8Unorm,
        wgpu::TextureFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
        wgpu::TextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8Srgb,
        wgpu::TextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8Srgb,
        wgpu::TextureFormat::R16Float => TextureFormat::R16Float,
        wgpu::TextureFormat::Rgba16Float => TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba16Uint => TextureFormat::Rgba16Uint,
        wgpu::TextureFormat::R32Float => TextureFormat::R32Float,
        wgpu::TextureFormat::Rg32Float => TextureFormat::R32G32Float,
        wgpu::TextureFormat::Rgba32Float => TextureFormat::Rgba32Float,
        wgpu::TextureFormat::Rg11b10Ufloat => TextureFormat::R11b11G11B10Float,
        wgpu::TextureFormat::Depth16Unorm => TextureFormat::Depth16,
        wgpu::TextureFormat::Depth24Plus => TextureFormat::Depth24,
        wgpu::TextureFormat::Depth32Float => TextureFormat::Depth32Float,
        wgpu::TextureFormat::Depth24PlusStencil8 => TextureFormat::Depth24Stencil8,
        // wgpu carries compressed and video formats the RHI has no name for.
        // They come back as `None` rather than as a default, because a swapchain
        // that silently reports the wrong format makes every colour wrong.
        _ => return None,
    })
}

/// wgpu texture usages, from the RHI's.
pub fn to_wgpu_texture_usage(usage: chassis::TextureUsage) -> wgpu::TextureUsages {
    let mut out = wgpu::TextureUsages::empty();
    if usage.contains(chassis::TextureUsage::SAMPLED) {
        out |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if usage.contains(chassis::TextureUsage::STORAGE) {
        out |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if usage.contains(chassis::TextureUsage::COLOR_ATTACHMENT) {
        out |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if usage.contains(chassis::TextureUsage::DEPTH_ATTACHMENT) {
        out |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if usage.contains(chassis::TextureUsage::TRANSFER_SRC) {
        out |= wgpu::TextureUsages::COPY_SRC;
    }
    if usage.contains(chassis::TextureUsage::TRANSFER_DST) {
        out |= wgpu::TextureUsages::COPY_DST;
    }
    // A sampled texture with no other use is still a valid thing to ask for, and
    // refusing it here would make `TextureUsage::SAMPLED` alone an error.
    if out.is_empty() {
        out |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    out
}

/// Vertex formats. A vertex attribute format is a different enum from a texture
/// format -- `Float32x4` is a vertex format and `Rgba32Float` is a texture
/// format -- so this is a separate mapping and not a reuse of the other one.
pub fn to_vertex_format(format: TextureFormat) -> Option<wgpu::VertexFormat> {
    Some(match format {
        TextureFormat::R32Float => wgpu::VertexFormat::Float32,
        TextureFormat::R32G32Float => wgpu::VertexFormat::Float32x2,
        TextureFormat::R32G32B32Float => wgpu::VertexFormat::Float32x3,
        TextureFormat::Rgba32Float => wgpu::VertexFormat::Float32x4,
        TextureFormat::Rgba16Uint => wgpu::VertexFormat::Uint16x4,
        // The packed unorm formats, which is how a mesh format usually arrives.
        _ => return None,
    })
}

/// wgpu buffer usages, from the RHI's.
pub fn to_wgpu_buffer_usage(usage: chassis::BufferUsage) -> wgpu::BufferUsages {
    let mut out = wgpu::BufferUsages::empty();
    if usage.contains(chassis::BufferUsage::VERTEX) {
        out |= wgpu::BufferUsages::VERTEX;
    }
    if usage.contains(chassis::BufferUsage::INDEX) {
        out |= wgpu::BufferUsages::INDEX;
    }
    if usage.contains(chassis::BufferUsage::UNIFORM) {
        out |= wgpu::BufferUsages::UNIFORM;
    }
    if usage.contains(chassis::BufferUsage::STORAGE) {
        out |= wgpu::BufferUsages::STORAGE;
    }
    if usage.contains(chassis::BufferUsage::INDIRECT) {
        out |= wgpu::BufferUsages::INDIRECT;
    }
    if usage.contains(chassis::BufferUsage::TRANSFER_SRC) {
        out |= wgpu::BufferUsages::COPY_SRC;
    }
    if usage.contains(chassis::BufferUsage::TRANSFER_DST) {
        out |= wgpu::BufferUsages::COPY_DST;
    }
    // `MAP_READ` has no Vulkan equivalent: Vulkan decides readability from the
    // memory type, not from a buffer flag. See the comment on the variant.
    if usage.contains(chassis::BufferUsage::MAP_READ) {
        out |= wgpu::BufferUsages::MAP_READ;
    }
    out
}

/// Everything this backend has created.
///
/// wgpu owns its objects: a `wgpu::Buffer` dies with the device and needs no
/// `destroy_buffer`. That is the single largest reason this backend is smaller
/// than the Vulkan one -- there is no `VulkanDeletionQueue` equivalent because
/// there is nothing to defer.
///
/// Not yet populated. It exists so the type is settled before the registries
/// that fill it are written against it, and so the borrow structure is decided
/// once rather than retrofitted.
#[allow(dead_code)]
pub struct Resources {
    buffers: HashMap<u32, wgpu::Buffer>,
    textures: HashMap<u32, wgpu::Texture>,
    views: HashMap<u32, wgpu::TextureView>,
    shaders: HashMap<u32, wgpu::ShaderModule>,
}
