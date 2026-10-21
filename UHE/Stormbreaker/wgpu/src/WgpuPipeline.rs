//! Graphics pipelines for the wgpu backend.
//!
//! ## Where this is genuinely simpler than the Vulkan backend
//!
//! No subpasses, no `VkRenderPass`, no `VkPipelineRenderingCreateInfo`. Under
//! dynamic rendering Vulkan has already thrown most of that away; wgpu never had
//! it. Attachment formats sit on the pipeline and the render pass sits on the
//! encoder, which is the whole arrangement.
//!
//! ## Where wgpu is stricter
//!
//! Bind group layouts are mandatory and explicit. Vulkan can create a pipeline
//! layout with zero descriptor sets and simply fail later when something is
//! bound; wgpu requires every layout at pipeline creation and validates the
//! pipeline against them, so a shader reading a binding the layout does not
//! declare fails *here*, naming the binding.
//!
//! ## Borrow lifetimes, which is the only fiddly part
//!
//! [`wgpu::RenderPipelineDescriptor`] borrows its shader modules, its vertex
//! buffer layout, its targets and its pipeline layout. Everything those borrow
//! from has to outlive the call, which means the intermediate `wgpu::ShaderState`
//! carries the layouts and the modules and builds the descriptor borrowing them.
//! Writing it inline instead produces a descriptor borrowing temporaries that are
//! dropped on the same statement -- which compiles nowhere and, where it does
//! compile, is a dangling pointer validation cannot describe.

use chassis::{PipelineDesc, ShaderStage};

use crate::wgpu_device;
use crate::wgpu_resources::WgpuResources;

/// The two pipeline objects wgpu has, behind one RHI handle.
enum PipelineInner {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

/// A compiled pipeline and the layouts it was built against.
pub struct PipelineResource {
    inner: PipelineInner,
    label: String,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

impl PipelineResource {
    pub fn as_render(&self) -> Option<&wgpu::RenderPipeline> {
        match &self.inner {
            PipelineInner::Render(pipeline) => Some(pipeline),
            PipelineInner::Compute(_) => None,
        }
    }

    pub fn as_compute(&self) -> Option<&wgpu::ComputePipeline> {
        match &self.inner {
            PipelineInner::Render(_) => None,
            PipelineInner::Compute(pipeline) => Some(pipeline),
        }
    }

    pub fn is_compute(&self) -> bool {
        matches!(self.inner, PipelineInner::Compute(_))
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// The layouts. A bind group must be made against one of these, and the
    /// encoder does not carry the pipeline's layouts -- so losing them here means
    /// recreating the pipeline layout just to allocate a bind group.
    pub fn bind_group_layouts(&self) -> &[wgpu::BindGroupLayout] {
        &self.bind_group_layouts
    }
}

/// Everything the descriptor borrows, so all of it outlives the call.
struct ShaderState<'a> {
    vertex_module: wgpu::ShaderModule,
    vertex_entry: String,
    fragment_module: Option<wgpu::ShaderModule>,
    fragment_entry: Option<String>,
    /// Built in the caller and stored, because `VertexBufferLayout` borrows its
    /// attributes and a layout built inside a method would borrow a `Vec` that
    /// dies when the method returns.
    buffers: Vec<Option<wgpu::VertexBufferLayout<'a>>>,
    targets: Vec<Option<wgpu::ColorTargetState>>,
}

impl ShaderState<'_> {
    fn vertex_state(&self) -> wgpu::VertexState<'_> {
        wgpu::VertexState {
            module: &self.vertex_module,
            entry_point: Some(&self.vertex_entry),
            buffers: &self.buffers,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }
    }

    fn fragment_state(&self) -> Option<wgpu::FragmentState<'_>> {
        self.fragment_module
            .as_ref()
            .map(|module| wgpu::FragmentState {
                module,
                entry_point: self.fragment_entry.as_deref(),
                targets: &self.targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
    }
}

/// Build a render pipeline from an RHI description.
pub fn create(
    resources: &mut WgpuResources,
    desc: &PipelineDesc,
) -> chassis::Result<chassis::Pipeline> {
    // Cloned rather than borrowed: `bind_group_layout` below takes `&mut
    // resources` to reach the shared cache, and a live `&resources.device()` would
    // make that a conflicting borrow for no reason -- `wgpu::Device` is a
    // reference-counted handle and cloning it bumps a refcount.
    let device = resources.device().inner().clone();

    // Built here and borrowed by the descriptor, so it outlives the call.
    let attributes: Vec<wgpu::VertexAttribute> = desc
        .vertex
        .attributes
        .iter()
        .filter_map(|attribute| {
            Some(wgpu::VertexAttribute {
                format: wgpu_device::to_vertex_format(attribute.format)?,
                offset: attribute.offset as u64,
                shader_location: attribute.location,
            })
        })
        .collect();
    // Every attribute on one buffer. A second binding would need a binding index
    // on each attribute, which `chassis::VertexAttribute` does not have -- the
    // same limitation the Vulkan backend has, for the same reason.
    let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = if attributes.is_empty() {
        Vec::new()
    } else {
        vec![Some(wgpu::VertexBufferLayout {
            array_stride: desc.vertex.stride as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attributes,
        })]
    };
    let state = shader_state(resources, desc, buffers)?;

    // Through the shared cache, not built here.
    //
    // These used to be created fresh per pipeline. That is harmless in Vulkan,
    // where the driver compares set layouts structurally, and fatal in wgpu,
    // which compares by identity: a bind group built against the cached layout
    // would not match a pipeline built against a duplicate of it, and the draw
    // would fail validation naming neither object.
    let mut layouts = Vec::with_capacity(desc.set_layouts.len());
    for set in &desc.set_layouts {
        layouts.push(resources.bind_group_layout(set)?);
    }

    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&desc.label),
        bind_group_layouts: &layout_refs,
        // Push constants in wgpu are one flat byte block rather than Vulkan's
        // ranged list, so this is a size and nothing else. Zero because the RHI
        // has no push-constant declaration yet.
        immediate_size: 0,
    });

    // Infallible in its signature, validated in the driver. The scope is what
    // makes a bad pipeline an error instead of the device handler's panic.
    let pipeline = crate::wgpu_resources::in_validation_scope(resources.device(), || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&desc.label),
            layout: Some(&layout),
            vertex: state.vertex_state(),
            primitive: primitive_of(&desc.primitive),
            depth_stencil: depth_of(&desc.depth),
            multisample: multisample_of(desc.samples),
            fragment: state.fragment_state(),
            multiview_mask: None,
            cache: None,
        })
    })?;

    Ok(resources.register_pipeline(PipelineResource {
        inner: PipelineInner::Render(pipeline),
        label: desc.label.clone(),
        bind_group_layouts: layouts,
    }))
}

/// Build a compute pipeline from an RHI description.
///
/// Same layout cache as a graphics pipeline: wgpu compares bind groups and
/// pipelines by layout identity, so the shared `bind_group_layout` is what
/// makes a group created for one usable with the other.
pub fn create_compute(
    resources: &mut WgpuResources,
    desc: &chassis::ComputePipelineDesc,
) -> chassis::Result<chassis::Pipeline> {
    let device = resources.device().inner().clone();
    let mut layouts = Vec::with_capacity(desc.set_layouts.len());
    for set in &desc.set_layouts {
        layouts.push(resources.bind_group_layout(set)?);
    }
    let Some(shader) = resources.get_shader(desc.shader) else {
        return Err(chassis::Error::InvalidHandle);
    };
    let entry = entry_of(shader, ShaderStage::Compute)?;
    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&desc.label),
        bind_group_layouts: &layout_refs,
        // wgpu has no push constants. The RHI reports that rather than letting
        // a shader declare a block no backend can fill.
        immediate_size: 0,
    });

    let pipeline = crate::wgpu_resources::in_validation_scope(resources.device(), || {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&desc.label),
            layout: Some(&layout),
            module: shader.inner(),
            // The module owns the name, and this is what the parser read.
            entry_point: Some(&entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    })?;

    Ok(resources.register_pipeline(PipelineResource {
        inner: PipelineInner::Compute(pipeline),
        label: desc.label.clone(),
        bind_group_layouts: layouts,
    }))
}

/// Split the declared stages into what wgpu wants, and build the borrowed
/// vertex layout and colour targets.
///
/// A graphics pipeline needs a vertex stage. A fragment stage is optional, and a
/// pipeline without one creates fine and writes nothing -- which is why only the
/// missing *vertex* stage is an error here.
fn shader_state<'a>(
    resources: &WgpuResources,
    desc: &PipelineDesc,
    buffers: Vec<Option<wgpu::VertexBufferLayout<'a>>>,
) -> chassis::Result<ShaderState<'a>> {
    let mut vertex = None;
    let mut vertex_entry = None;
    let mut fragment = None;
    let mut fragment_entry = None;
    for declared in &desc.shaders {
        let Some(shader) = resources.get_shader(declared.shader) else {
            return Err(chassis::Error::InvalidHandle);
        };
        match declared.stage {
            ShaderStage::Vertex => {
                vertex_entry = Some(entry_of(shader, declared.stage)?);
                vertex = Some(shader.inner().clone());
            }
            ShaderStage::Fragment => {
                fragment_entry = Some(entry_of(shader, declared.stage)?);
                fragment = Some(shader.inner().clone());
            }
            other => {
                // Named, not ignored. wgpu's `VertexState`/`FragmentState` pair
                // has no room for a geometry or tessellation stage, and dropping
                // one silently produces a pipeline that draws with fewer stages
                // than its shader expects.
                return Err(chassis::Error::Unsupported(format!(
                    "{other:?} is not wired for wgpu yet"
                )));
            }
        }
    }
    let vertex_module = vertex.ok_or_else(|| {
        chassis::Error::Unsupported("a graphics pipeline needs a vertex stage".to_owned())
    })?;
    let vertex_entry = vertex_entry.ok_or_else(|| {
        chassis::Error::Unsupported("a graphics pipeline needs a vertex stage".to_owned())
    })?;

    // A `ColorTargetState` the shader does not write would be a validation
    // error, so targets come from the description and anything unmappable
    // becomes `None` -- "unused at this location" -- rather than a substitute
    // format.
    let targets: Vec<Option<wgpu::ColorTargetState>> = desc
        .color_targets
        .iter()
        .map(|target| {
            wgpu_device::to_wgpu_format(target.format).map(|format| wgpu::ColorTargetState {
                format,
                // `load` is on the encoder in wgpu, not here: the RHI's colour
                // attachment load op is a render-pass concept and this backend
                // has no render pass.
                blend: target.blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
                // A bitflags, not a mask value: `from_bits_truncate` rather
                // than a cast, so a write_mask with a bit outside the four
                // channels is dropped instead of being a nonsense value. The
                // RHI's mask is `0xf` for "all", which is `ColorWrites::ALL`.
                write_mask: wgpu::ColorWrites::from_bits_truncate(u32::from(target.write_mask)),
            })
        })
        .collect();

    Ok(ShaderState {
        vertex_module,
        vertex_entry,
        fragment_module: fragment,
        fragment_entry,
        buffers,
        targets,
    })
}

/// The module's own entry name for `stage`.
fn entry_of(
    shader: &crate::wgpu_resources::ShaderResource,
    stage: ShaderStage,
) -> chassis::Result<String> {
    shader.entry(stage).map(str::to_owned).ok_or_else(|| {
        chassis::Error::InvalidDescriptor(format!(
            "a {stage:?} stage was declared but the shader module has no entry point for it"
        ))
    })
}

pub(crate) fn visibility_of(mask: chassis::StageMask) -> wgpu::ShaderStages {
    let mut stages = wgpu::ShaderStages::empty();
    if mask.contains(ShaderStage::Vertex) {
        stages |= wgpu::ShaderStages::VERTEX;
    }
    if mask.contains(ShaderStage::Fragment) {
        stages |= wgpu::ShaderStages::FRAGMENT;
    }
    if mask.contains(ShaderStage::Compute) {
        stages |= wgpu::ShaderStages::COMPUTE;
    }
    if stages.is_empty() {
        // wgpu rejects empty visibility and the message names the binding, not
        // what to write here. Everything visible is one redundant barrier at
        // worst.
        stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
    }
    stages
}

pub(crate) fn layout_type_of(kind: chassis::DescriptorKind) -> chassis::Result<wgpu::BindingType> {
    let texture = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    Ok(match kind {
        chassis::DescriptorKind::UniformBuffer => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        // Read-only. The RHI's `StorageBuffer` does not say whether a shader
        // writes it, and wgpu refuses a pipeline whose layout disagrees with the
        // shader: `LOAD | STORE` against a shader's `LOAD` is
        // "Storage class ... doesn't match the shader".
        //
        // The one binding in the engine (the frame set's light list) is read, so
        // read-only is both the least privilege and what matches. It also keeps
        // the vertex-stage binding legal: a *writable* storage buffer visible to
        // the vertex stage needs `VERTEX_WRITABLE_STORAGE`, and this device is
        // not built with it.
        //
        // shortcut: the RHI cannot express a storage buffer a shader writes, so
        // every one maps to read-only; give `DescriptorKind` the distinction
        // when a shader needs to write one.
        chassis::DescriptorKind::StorageBuffer => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        // The writable half. wgpu validates this against the shader's access,
        // so a compute shader writing through a read-only layout is refused at
        // pipeline creation rather than at dispatch.
        chassis::DescriptorKind::StorageBufferReadWrite => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        chassis::DescriptorKind::SampledImage => texture,
        // Not `texture`: a depth view needs `TextureSampleType::Depth`, and
        // the bind group layout is checked against the view's format at
        // `create_bind_group`. `multisampled: false` because wgpu requires
        // that for a depth binding.
        chassis::DescriptorKind::SampledDepth => wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        chassis::DescriptorKind::CombinedImageSampler => texture,
        chassis::DescriptorKind::StorageImage => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        chassis::DescriptorKind::Sampler => {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        }
        chassis::DescriptorKind::InputAttachment => {
            // Not a gap so much as a difference: an input attachment is a
            // subpass concept, and wgpu has no subpasses. The equivalent is
            // sampling the texture, which is `SampledImage`.
            return Err(chassis::Error::Unsupported(
                "input attachments need subpasses, which wgpu has not got; sample the texture instead"
                    .to_owned(),
            ));
        }
    })
}

fn primitive_of(primitive: &chassis::PrimitiveState) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: match primitive.topology {
            0 => wgpu::PrimitiveTopology::PointList,
            1 => wgpu::PrimitiveTopology::LineList,
            2 => wgpu::PrimitiveTopology::LineStrip,
            3 => wgpu::PrimitiveTopology::TriangleStrip,
            _ => wgpu::PrimitiveTopology::TriangleList,
        },
        strip_index_format: None,
        front_face: if primitive.front_face == 1 {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        },
        // 0 none, 2 front, anything else back -- the same mapping as the Vulkan
        // backend, and the same Vulkan Y-flip caveat: wgpu also flips, so a
        // counter-clockwise NDC triangle needs `front_face: 1` or no culling.
        cull_mode: match primitive.cull_mode {
            0 => None,
            2 => Some(wgpu::Face::Front),
            _ => Some(wgpu::Face::Back),
        },
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    }
}

fn depth_of(depth: &chassis::DepthState) -> Option<wgpu::DepthStencilState> {
    let format = wgpu_device::to_wgpu_format(depth.format?)?;
    Some(wgpu::DepthStencilState {
        format,
        // `Option<bool>` in wgpu 30, and the three are genuinely different:
        // `None` leaves the value untouched, `Some(false)` writes it as 1.0.
        // The RHI has two states, so it maps onto `Some` -- and says so, because
        // the mapping from "no depth" to `None` happens earlier.
        depth_write_enabled: Some(depth.write_enabled),
        // `Option` in wgpu 30: `None` means "no depth test". The RHI expresses
        // the same thing by having no depth attachment at all, which returned
        // early above, so reaching here means there is a test.
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

fn multisample_of(samples: chassis::SampleCount) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: match samples {
            chassis::SampleCount::X1 => 1,
            chassis::SampleCount::X2 => 2,
            chassis::SampleCount::X4 => 4,
            chassis::SampleCount::X8 => 8,
        },
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}

#[cfg(test)]
mod tests {
    use super::layout_type_of;
    use chassis::DescriptorKind;
    use wgpu::BindingType;

    /// The whole mapping, one assertion per variant.
    ///
    /// `layout_type_of` is the only place a `DescriptorKind` becomes a wgpu type,
    /// so it is the only place a new kind can be forgotten -- and a forgotten
    /// kind is not a compile error here, it is a binding that does not work.
    #[test]
    fn every_descriptor_kind_maps_to_its_binding_type() {
        assert!(matches!(
            layout_type_of(DescriptorKind::UniformBuffer),
            Ok(BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            })
        ));
        // Read-only, not writable: the shader reads this binding, and wgpu
        // refuses a pipeline whose layout disagrees with the shader. See the
        // comment in `layout_type_of`.
        assert!(matches!(
            layout_type_of(DescriptorKind::StorageBuffer),
            Ok(BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                ..
            })
        ));
        assert!(matches!(
            layout_type_of(DescriptorKind::StorageBufferReadWrite),
            Ok(BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                ..
            })
        ));
        assert!(matches!(
            layout_type_of(DescriptorKind::SampledImage),
            Ok(BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                ..
            })
        ));
        assert!(matches!(
            layout_type_of(DescriptorKind::SampledDepth),
            Ok(BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            })
        ));
        assert!(matches!(
            layout_type_of(DescriptorKind::Sampler),
            Ok(BindingType::Sampler(wgpu::SamplerBindingType::Filtering))
        ));
        assert!(matches!(
            layout_type_of(DescriptorKind::CombinedImageSampler),
            Ok(BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                ..
            })
        ));
        assert!(matches!(
            layout_type_of(DescriptorKind::StorageImage),
            Ok(BindingType::StorageTexture { .. })
        ));
        assert!(
            layout_type_of(DescriptorKind::InputAttachment).is_err(),
            "wgpu has no subpasses, so an input attachment is refused",
        );
    }

    /// A depth texture is not a sampled image here, and the whole reason
    /// `DescriptorKind::SampledDepth` exists.
    ///
    /// wgpu checks the sample type of a bind group's view against the layout at
    /// `create_bind_group`, so a `Float` binding holding a depth view is refused
    /// by name. Vulkan takes one `SAMPLED_IMAGE` for both.
    #[test]
    fn a_depth_texture_is_not_a_float_texture() {
        let depth = layout_type_of(DescriptorKind::SampledDepth).expect("a depth texture");
        let image = layout_type_of(DescriptorKind::SampledImage).expect("a colour texture");

        assert_ne!(depth, image);
    }
}
