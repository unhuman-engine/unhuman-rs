//! The wgpu implementation of `chassis::Resources`.
//!
//! ## The same trait, the same handles, a different object behind them
//!
//! This and `VulkanResources` are interchangeable behind `&mut dyn Resources`.
//! The renderer cannot tell them apart, which is the entire point of splitting
//! the RHI out of the engine crate -- and the reason a test that calls the
//! concrete methods proves nothing.
//!
//! ## Where wgpu is genuinely smaller
//!
//! There is no deletion queue. Vulkan needs one because a `VkBuffer` outlives
//! whatever submitted work might still read it, and freeing it early is a
//! use-after-free that only shows up under load. wgpu owns its objects: a
//! `wgpu::Buffer` dies with the device and is reference-counted until then. The
//! absence is not an omission, it is the single largest reason this backend will
//! end up about half the size of the Vulkan one.
//!
//! ## The one place wgpu costs more
//!
//! Format and usage translation. Vulkan names formats and usage bits directly;
//! wgpu has its own vocabulary (`Rg11b10Ufloat` for `R11b11G11B10Float`,
//! `...UnormSrgb` for `...Srgb`), and a mapping that silently substitutes a
//! near-enough format renders the wrong image rather than failing. Every
//! unmappable format is `None` here for that reason.

use std::borrow::Cow;

use chassis::Registry;
// `create_buffer_init` is not on `wgpu::Device`. It is an extension trait in
// `wgpu::util`, which is the crate's way of saying "convenience, not core API"
// -- and forgetting the import is a compile error rather than a silent
// fallback, which is the right way round.
use wgpu::util::DeviceExt;

use crate::wgpu_device::{self, Device};

/// A compiled pipeline.
///
/// wgpu owns the `RenderPipeline`, so as with buffers and textures this only has
/// to exist to keep the handle resolving.
pub use crate::wgpu_pipeline::PipelineResource;

/// What the RHI calls a buffer.
pub struct BufferResource {
    inner: wgpu::Buffer,
    size: u64,
    /// The RHI usage it was created with. wgpu can be asked, but only through
    /// `wgpu::Buffer::usage`, and a readback has to refuse a buffer the caller
    /// never made readable before wgpu turns that into a validation panic.
    usage: chassis::BufferUsage,
    label: String,
}

impl BufferResource {
    pub fn inner(&self) -> &wgpu::Buffer {
        &self.inner
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn usage(&self) -> chassis::BufferUsage {
        self.usage
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

/// A texture and the view made with it.
///
/// The view lives here rather than in a separate registry because wgpu makes no
/// other claim on it and it dies with the texture anyway. The RHI can still ask
/// for a *second* view with a different aspect, and those are tracked
/// separately -- a view must never outlive what it views.
pub struct TextureResource {
    inner: wgpu::Texture,
    view: wgpu::TextureView,
    format: chassis::TextureFormat,
    extent: chassis::Extent,
    label: String,
}

impl TextureResource {
    pub fn inner(&self) -> &wgpu::Texture {
        &self.inner
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn format(&self) -> chassis::TextureFormat {
        self.format
    }

    pub fn extent(&self) -> chassis::Extent {
        self.extent
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

pub struct ShaderResource {
    inner: wgpu::ShaderModule,
    label: String,
}

impl ShaderResource {
    pub fn inner(&self) -> &wgpu::ShaderModule {
        &self.inner
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Owns every wgpu object the RHI creates.
pub struct WgpuResources {
    device: Device,
    /// Kept for `upload_texture`: `queue.write_texture` is the whole of a wgpu
    /// upload, and unlike Vulkan there is no staging buffer and no command buffer
    /// because the driver does the copy.
    /// `None` when built without one; `upload_texture` reports that rather than
    /// dropping the bytes.
    queue: Option<wgpu::Queue>,
    buffers: Registry<BufferResource, chassis::BufferTag>,
    textures: Registry<TextureResource, chassis::TextureTag>,
    views: Registry<wgpu::TextureView, chassis::ImageViewTag>,
    shaders: Registry<ShaderResource, chassis::ShaderTag>,
    pipelines: Registry<PipelineResource, chassis::PipelineTag>,
    pub(crate) bind_groups: Registry<wgpu::BindGroup, chassis::BindGroupTag>,
    samplers: Registry<wgpu::Sampler, chassis::SamplerTag>,
    /// `DescriptorSetLayoutDesc::hash` to the layout built from it.
    ///
    /// Shared between pipeline creation and bind group creation, and it has to
    /// be: wgpu compares a bind group against the pipeline's layout by identity,
    /// so two equal descriptions built separately are two layouts, and binding
    /// one to the other is a validation error at draw time.
    ///
    /// This is the wgpu difference from Vulkan, where the driver compares set
    /// layouts structurally and two separately-created but identical
    /// `vk::DescriptorSetLayout`s are interchangeable. Here they are not, so the
    /// cache is not an optimisation -- it is what makes bind groups work at all.
    bind_group_layouts: std::collections::HashMap<u64, wgpu::BindGroupLayout>,
}

impl WgpuResources {
    pub fn new(device: &Device) -> Self {
        Self::with_queue(device, None)
    }

    /// With a queue, which is what `upload_texture` needs.
    ///
    /// `new` keeps working without one so a caller that only reads and draws
    /// does not have to hold a queue it never uses. `upload_texture` then fails
    /// with a message that says why, rather than silently discarding bytes.
    pub fn with_queue(device: &Device, queue: Option<wgpu::Queue>) -> Self {
        Self {
            device: device.clone(),
            queue,
            buffers: Registry::new(),
            textures: Registry::new(),
            views: Registry::new(),
            shaders: Registry::new(),
            pipelines: Registry::new(),
            bind_groups: Registry::new(),
            samplers: Registry::new(),
            bind_group_layouts: std::collections::HashMap::new(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn get_buffer(&self, handle: chassis::Buffer) -> Option<&BufferResource> {
        self.buffers.get(handle)
    }

    pub fn get_texture(&self, handle: chassis::Texture) -> Option<&TextureResource> {
        self.textures.get(handle)
    }

    pub fn get_shader(&self, handle: chassis::Shader) -> Option<&ShaderResource> {
        self.shaders.get(handle)
    }

    /// Stores a compiled pipeline. Called by `wgpu_pipeline`, which builds one
    /// and needs somewhere to put it.
    pub(crate) fn register_pipeline(
        &mut self,
        pipeline: crate::wgpu_pipeline::PipelineResource,
    ) -> chassis::Pipeline {
        self.pipelines.create(pipeline)
    }

    pub fn get_pipeline(&self, handle: chassis::Pipeline) -> Option<&PipelineResource> {
        self.pipelines.get(handle)
    }

    /// Mints an RHI handle for a texture this backend did not create.
    ///
    /// For swapchain images: the surface owns them and a new one arrives every
    /// frame, so there is no `TextureDesc` to create one from -- but a render-graph
    /// pass attaches its target by handle, so there has to be one. Nothing is
    /// destroyed with the handle; `forget_texture` retires it.
    pub fn register_texture(
        &mut self,
        texture: wgpu::Texture,
        format: chassis::TextureFormat,
        extent: chassis::Extent,
        label: &str,
    ) -> chassis::Texture {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.textures.create(TextureResource {
            inner: texture,
            view,
            format,
            extent,
            label: label.to_owned(),
        })
    }

    /// Mints an RHI handle for a view this backend did not create.
    ///
    /// The wgpu twin of `VulkanResources::register_image_view`, and for the same
    /// caller: a swapchain image's view belongs to the surface, so
    /// `create_image_view` -- which takes a `Texture` from this registry -- cannot
    /// make one.
    pub fn register_image_view(&mut self, view: wgpu::TextureView) -> chassis::ImageView {
        self.views.create(view)
    }

    /// Retires a handle from [`register_texture`].
    pub fn forget_texture(&mut self, handle: chassis::Texture) {
        self.textures.remove(handle);
    }

    /// Retires a handle from [`register_image_view`].
    pub fn forget_image_view(&mut self, handle: chassis::ImageView) {
        self.views.remove(handle);
    }

    /// Live objects, for a leak check in a test.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        (
            self.buffers.len(),
            self.textures.len(),
            self.views.len(),
            self.shaders.len(),
        )
    }

    /// Forgets a handle without destroying anything.
    ///
    /// wgpu needs no destroy, so this is all `destroy_*` has to do -- but the
    /// handle must still stop resolving, or a stale one returns a dropped
    /// object and the next use is a use-after-free rather than a clean error.
    fn forget_buffer(&mut self, handle: chassis::Buffer) {
        self.buffers.remove(handle);
    }

    fn forget_shader(&mut self, handle: chassis::Shader) {
        self.shaders.remove(handle);
    }
}

/// `chassis::Filter` as `wgpu::FilterMode`.
fn filter_of(filter: chassis::Filter) -> wgpu::FilterMode {
    match filter {
        chassis::Filter::Nearest => wgpu::FilterMode::Nearest,
        chassis::Filter::Linear => wgpu::FilterMode::Linear,
    }
}

/// `chassis::AddressMode` as `wgpu::AddressMode`.
fn address_mode_of(mode: chassis::AddressMode) -> wgpu::AddressMode {
    match mode {
        chassis::AddressMode::Repeat => wgpu::AddressMode::Repeat,
        chassis::AddressMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        chassis::AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
    }
}

/// wgpu's `count`, from the RHI's.
///
/// ## `Some(1)` is rejected; `None` is accepted
///
/// In wgpu 30 `count: Some(n)` means a **binding array** and requires the
/// matching `*_BINDING_ARRAY` feature -- including `Some(1)`. A plain binding
/// must be `None`. This is not a thing one guesses: `Some(1)` reads like "one
/// element" and is the natural translation of `count: 1`, and wgpu rejects it
/// with a message about a *feature* rather than about the count.
///
/// So one becomes `None` and only a genuine count above one becomes `Some`.
/// Proven by probing all three spellings on a real device rather than by reading
/// the wgpu changelog, which is where the answer should have come from.
fn binding_count(count: u32) -> Option<std::num::NonZeroU32> {
    (count > 1)
        .then(|| std::num::NonZeroU32::new(count))
        .flatten()
}

impl WgpuResources {
    /// The layout for `desc`, built once and shared.
    ///
    /// Load-bearing, not an optimisation: wgpu compares a bind group against the
    /// pipeline's layout by identity, so two equal descriptions built separately
    /// are two layouts and binding one to the other is a draw-time validation
    /// error naming neither. Vulkan compares structurally and does not care --
    /// that is the backend difference this cache exists to absorb.
    pub fn bind_group_layout(
        &mut self,
        desc: &chassis::DescriptorSetLayoutDesc,
    ) -> chassis::Result<wgpu::BindGroupLayout> {
        let key = desc.hash();
        if let Some(layout) = self.bind_group_layouts.get(&key) {
            return Ok(layout.clone());
        }
        let mut entries = Vec::with_capacity(desc.bindings.len());
        for binding in &desc.bindings {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding.binding,
                visibility: crate::wgpu_pipeline::visibility_of(binding.stages),
                ty: crate::wgpu_pipeline::layout_type_of(binding.kind)?,
                count: binding_count(binding.count),
            });
        }
        let layout =
            self.device
                .inner()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&desc.label),
                    entries: &entries,
                });
        self.bind_group_layouts.insert(key, layout.clone());
        Ok(layout)
    }

    /// Distinct bind group layouts built. A count that climbs per frame means
    /// the cache key does not match `is_compatible`.
    pub fn bind_group_layout_count(&self) -> usize {
        self.bind_group_layouts.len()
    }
}

impl chassis::Resources for WgpuResources {
    fn create_buffer(
        &mut self,
        desc: chassis::BufferDesc,
        initial: Option<&[u8]>,
    ) -> chassis::Result<chassis::Buffer> {
        let usage = wgpu_device::to_wgpu_buffer_usage(desc.usage);
        if usage.is_empty() {
            // Vulkan has the same rule and the same reason: a buffer with no
            // usage is a driver object that can do nothing, and some drivers
            // accept it and then ignore every command recorded against it.
            return Err(chassis::Error::Unsupported(
                "a buffer needs at least one usage".to_owned(),
            ));
        }
        if desc.size == 0 {
            return Err(chassis::Error::Unsupported(
                "a buffer of zero size is not a thing".to_owned(),
            ));
        }

        let label = format!("buffer({} bytes)", desc.size);
        let inner = match initial {
            Some(bytes) => {
                // `create_buffer_init` uploads and keeps the staging copy alive
                // until the first submit that reads it. Without initial data a
                // plain buffer is correct and cheaper, so the two paths are
                // kept apart rather than always paying for the staging one.
                self.device
                    .inner()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&label),
                        contents: bytes,
                        usage,
                    })
            }
            None => self.device.inner().create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: desc.size,
                usage,
                mapped_at_creation: false,
            }),
        };

        Ok(self.buffers.create(BufferResource {
            inner,
            size: desc.size,
            usage: desc.usage,
            label,
        }))
    }

    fn write_buffer(
        &mut self,
        buffer: &chassis::Buffer,
        offset: u64,
        data: &[u8],
    ) -> chassis::Result<()> {
        // Without a queue there is nothing to stage the write through, so refuse
        // by name rather than dropping the bytes.
        let Some(queue) = self.queue.as_ref() else {
            return Err(chassis::Error::Unsupported(
                "write_buffer needs a queue; build the resources with \
                 WgpuResources::with_queue"
                    .to_owned(),
            ));
        };
        let resource = self
            .buffers
            .get(*buffer)
            .ok_or(chassis::Error::InvalidHandle)?;
        // `queue.write_buffer` requires COPY_DST, and a buffer without it is a
        // wgpu validation panic rather than a `Result`.
        if !resource
            .usage()
            .contains(chassis::BufferUsage::TRANSFER_DST)
        {
            return Err(chassis::Error::InvalidDescriptor(format!(
                "write_buffer needs a buffer created with TRANSFER_DST ({})",
                resource.label()
            )));
        }
        if offset + data.len() as u64 > resource.size() {
            return Err(chassis::Error::InvalidDescriptor(format!(
                "write_buffer writes {} bytes at {offset} into a {} byte buffer",
                data.len(),
                resource.size()
            )));
        }
        queue.write_buffer(resource.inner(), offset, data);
        Ok(())
    }

    fn create_texture(&mut self, desc: chassis::TextureDesc) -> chassis::Result<chassis::Texture> {
        let Some(format) = wgpu_device::to_wgpu_format(desc.format) else {
            // No substitution. A near-enough format renders the wrong image,
            // which is a colour bug that sends you looking at the shader.
            return Err(chassis::Error::Unsupported(format!(
                "{:?} has no wgpu equivalent",
                desc.format
            )));
        };
        let usage = wgpu_device::to_wgpu_texture_usage(desc.usage);

        let label = "texture".to_owned();
        let size = wgpu::Extent3d {
            width: desc.extent.width.max(1),
            height: desc.extent.height.max(1),
            depth_or_array_layers: desc.extent.depth.max(1),
        };
        let inner = self
            .device
            .inner()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(&label),
                size,
                mip_level_count: desc.mip_levels.max(1),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });

        let view = inner.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(self.textures.create(TextureResource {
            inner,
            view,
            format: desc.format,
            extent: desc.extent,
            label,
        }))
    }

    fn create_image_view(
        &mut self,
        texture: &chassis::Texture,
    ) -> chassis::Result<chassis::ImageView> {
        let Some(resource) = self.textures.get(*texture) else {
            return Err(chassis::Error::InvalidHandle);
        };
        // A second view, not the one made with the texture. A sampler wants a
        // different aspect or mip range than a render target does, and handing
        // back the first view would be wrong for that without saying so.
        let view = resource
            .inner()
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok(self.views.create(view))
    }

    fn create_shader(&mut self, label: &str, spirv: &[u8]) -> chassis::Result<chassis::Shader> {
        let module = forge::Module::from_bytes(spirv)
            .ok_or_else(|| chassis::Error::InvalidDescriptor("not SPIR-V".to_owned()))?;

        // The magic number is checked above by `forge`, and that check is doing
        // more than tidiness: wgpu's `create_shader_module` is infallible, so a
        // module it dislikes goes to an error scope, and with no scope installed
        // it *panics*. Vulkan's returns `Result` and does not. Validating here is
        // what makes the two backends agree on the common case -- truncated and
        // wrong-magic input.
        //
        // What it does not catch is a module with a correct magic number and a
        // corrupt body. That panics here and is not caught, deliberately: a panic
        // is loud and immediate, and wrapping it in a catch would turn a
        // specific failure into a silent wrong answer.
        let inner = self
            .device
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                // SPIR-V, straight from forge. The same bytes the Vulkan backend
                // takes: one compiler, two consumers. naga validates on the way
                // in, so a malformed module is reported here rather than as an
                // opaque failure at pipeline creation.
                source: wgpu::ShaderSource::SpirV(Cow::Owned(module.words)),
            });
        Ok(self.shaders.create(ShaderResource {
            inner,
            label: label.to_owned(),
        }))
    }

    fn create_pipeline(
        &mut self,
        desc: chassis::PipelineDesc,
    ) -> chassis::Result<chassis::Pipeline> {
        crate::wgpu_pipeline::create(self, &desc)
    }

    fn create_bind_group(
        &mut self,
        layout: &chassis::DescriptorSetLayoutDesc,
        buffers: &[(u32, chassis::Buffer, u64)],
        images: &[(u32, chassis::ImageView)],
        samplers: &[(u32, chassis::Sampler)],
    ) -> chassis::Result<chassis::BindGroup> {
        // The *same* layout object the pipeline was built with, from the shared
        // cache. Building it here as well would give two equal layouts, and wgpu
        // compares by identity -- so the bind group would be unusable with the
        // pipeline and the failure would be at draw time, naming neither.
        let group_layout = self.bind_group_layout(layout)?;

        // Checked here because `create_bind_group` cannot report it.
        //
        // wgpu's `create_bind_group` is infallible: it panics on a binding the
        // layout does not declare ("Unable to find a corresponding declaration
        // for the given binding 9"). Vulkan returns an error for the same mistake
        // and the RHI returns `Result`, so a caller cannot defend itself -- the
        // only way to give it an error instead of a panic is to refuse first.
        //
        // This is the same asymmetry as shader module creation, where wgpu panics
        // on a corrupt SPIR-V body and Vulkan returns `Result`. The rule that
        // follows: where a wgpu call cannot fail, this layer checks what it needs
        // and turns the panic into an error.
        let declared = |binding: u32| -> chassis::Result<chassis::DescriptorKind> {
            layout
                .bindings
                .iter()
                .find(|d| d.binding == binding)
                .map(|d| d.kind)
                .ok_or_else(|| {
                    chassis::Error::InvalidDescriptor(format!(
                        "binding {binding} was written but the layout does not declare it"
                    ))
                })
        };

        let mut entries = Vec::with_capacity(buffers.len() + images.len() + samplers.len());
        for (binding, buffer, offset) in buffers {
            declared(*binding)?;
            let resource = self
                .buffers
                .get(*buffer)
                .ok_or(chassis::Error::InvalidHandle)?;
            let size = resource.size();
            if *offset >= size {
                return Err(chassis::Error::InvalidDescriptor(format!(
                    "binding {binding}: offset {offset} is at or past the end of a \
                     {size}-byte buffer"
                )));
            }
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &resource.inner,
                    offset: *offset,
                    size: wgpu::BufferSize::new(size - *offset),
                }),
            });
        }

        // Images. The declared kind decides the entry shape, because wgpu's
        // `CombinedImageSampler` is one binding carrying both and a
        // `SampledImage` is two -- which is the same pairing the layout described,
        // so the caller writes the same binding numbers on both backends.
        for (binding, view) in images {
            let kind = declared(*binding)?;
            let texture = self.views.get(*view).ok_or(chassis::Error::InvalidHandle)?;
            entries.push(match kind {
                chassis::DescriptorKind::SampledImage
                | chassis::DescriptorKind::SampledDepth
                | chassis::DescriptorKind::StorageImage => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                chassis::DescriptorKind::CombinedImageSampler => {
                    // **wgpu has no combined image sampler.**
                    //
                    // There is no `BindingResource::SamplerAndTextureView`; the
                    // variants are Buffer, Sampler, TextureView, and their array
                    // forms, because WebGPU -- which wgpu implements -- has no
                    // combined descriptor. Vulkan has `COMBINED_IMAGE_SAMPLER` and
                    // uses it by default.
                    //
                    // So the RHI's most convenient descriptor kind does not exist
                    // on this backend, and the honest answer is to say so and name
                    // the alternative. Silently building two entries for one
                    // binding would produce a group the layout cannot accept.
                    return Err(chassis::Error::Unsupported(format!(
                        "binding {binding} is a CombinedImageSampler, which wgpu does \
                         not have; declare SampledImage and Sampler as two bindings"
                    )));
                }
                other => {
                    return Err(chassis::Error::InvalidDescriptor(format!(
                        "binding {binding} is a {other:?}, not an image"
                    )));
                }
            });
        }

        for (binding, sampler) in samplers {
            let kind = declared(*binding)?;
            if kind == chassis::DescriptorKind::CombinedImageSampler {
                // Carried by the image entry above. Adding it again would give one
                // binding two entries, which wgpu rejects.
                continue;
            }
            if kind != chassis::DescriptorKind::Sampler {
                return Err(chassis::Error::InvalidDescriptor(format!(
                    "binding {binding} is a {kind:?}, not a sampler"
                )));
            }
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::Sampler(
                    self.samplers
                        .get(*sampler)
                        .ok_or(chassis::Error::InvalidHandle)?,
                ),
            });
        }

        let group = self
            .device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&layout.label),
                layout: &group_layout,
                entries: &entries,
            });
        Ok(self.bind_groups.create(group))
    }

    fn create_sampler(&mut self, desc: &chassis::SamplerDesc) -> chassis::Result<chassis::Sampler> {
        let sampler = self
            .device
            .inner()
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some("sampler"),
                address_mode_u: address_mode_of(desc.address_u),
                address_mode_v: address_mode_of(desc.address_v),
                address_mode_w: address_mode_of(desc.address_w),
                mag_filter: filter_of(desc.mag_filter),
                min_filter: filter_of(desc.min_filter),
                // `mipmap_filter` is its own enum here and takes a
                // `FilterMode`, not a `Filter` -- so the same three-to-two mistake
                // Vulkan invites does not exist. Worth knowing before porting the
                // Vulkan mapping across.
                // `MipmapFilterMode`, not `FilterMode`. Two enums with the same
                // two variants and different names -- and the one that belongs is
                // the longer. Porting the Vulkan mapping across unchanged is the
                // mistake, and the compiler catches it.
                mipmap_filter: match desc.mip_filter {
                    chassis::Filter::Nearest => wgpu::MipmapFilterMode::Nearest,
                    chassis::Filter::Linear => wgpu::MipmapFilterMode::Linear,
                },
                ..Default::default()
            });
        Ok(self.samplers.create(sampler))
    }

    fn destroy_sampler(&mut self, sampler: chassis::Sampler) {
        // `wgpu::Sampler` is reference counted with no explicit destroy, so
        // releasing the registry's reference is the whole of it. The slot goes
        // so a stale handle cannot resolve to a later sampler.
        self.samplers.remove(sampler);
    }

    fn destroy_buffer(&mut self, buffer: chassis::Buffer) {
        // No `destroy_buffer`: wgpu reference-counts. Only the handle has to
        // stop resolving.
        self.forget_buffer(buffer);
    }

    fn upload_texture(
        &mut self,
        texture: &chassis::Texture,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> chassis::Result<()> {
        let resource = self
            .textures
            .get(*texture)
            .ok_or(chassis::Error::InvalidHandle)?;
        let expected = width as usize * height as usize * 4;
        if data.len() != expected {
            return Err(chassis::Error::InvalidDescriptor(format!(
                "{} bytes for a {width}x{height} RGBA8 texture, which is {expected}",
                data.len()
            )));
        }
        let Some(queue) = self.queue.as_ref() else {
            return Err(chassis::Error::Unsupported(
                "upload_texture needs a queue; build the resources with \
                 WgpuResources::with_queue"
                    .to_owned(),
            ));
        };
        // `wgpu` has no layout parameter for this and no staging step: it stages
        // internally and copies. The whole reason Vulkan needs two barriers and a
        // command buffer to do the same thing.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &resource.inner,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // `Option<u32>` in wgpu 30, not a plain number. `None` means
                // "tightly packed", which is *not* the same as `Some(width * 4)`
                // for the last row -- and wgpu rejects a row pitch that is not a
                // multiple of 256 when there is more than one row.
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    fn read_buffer(&mut self, buffer: &chassis::Buffer) -> chassis::Result<Vec<u8>> {
        let resource = self
            .buffers
            .get(*buffer)
            .ok_or(chassis::Error::InvalidHandle)?;
        // Refused here rather than left to wgpu, which validates the map against
        // the creation usage and reports it through an error scope -- a panic
        // when none is installed, and an unactionable one at that.
        if !resource.usage().contains(chassis::BufferUsage::MAP_READ) {
            return Err(chassis::Error::Unsupported(
                "read_buffer needs a buffer created with BufferUsage::MAP_READ".to_owned(),
            ));
        }

        let slice = resource.inner().slice(..);
        // `map_async`'s callback runs from `poll`, so the poll is not optional:
        // without it the callback never fires and the range is not mapped.
        slice.map_async(wgpu::MapMode::Read, |_| {});

        self.device
            .inner()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| chassis::Error::DeviceLost)?;

        let view = slice.get_mapped_range().map_err(|error| {
            chassis::Error::Unsupported(format!("buffer not mappable: {error}"))
        })?;
        let bytes = view.to_vec();
        drop(view);
        resource.inner().unmap();
        Ok(bytes)
    }

    fn destroy_texture(&mut self, texture: chassis::Texture) {
        self.forget_texture(texture);
    }

    fn destroy_pipeline(&mut self, pipeline: chassis::Pipeline) {
        self.pipelines.remove(pipeline);
    }

    fn destroy_shader(&mut self, shader: chassis::Shader) {
        self.forget_shader(shader);
    }

    fn destroy_bind_group(&mut self, group: chassis::BindGroup) {
        // wgpu objects are reference-counted and have no explicit destroy, so
        // dropping the registry's reference is the whole of it. The slot is
        // released so a stale handle cannot resolve to a later group -- a
        // generational handle that resolved would be worse than one that does
        // not.
        self.bind_groups.remove(group);
    }
}
