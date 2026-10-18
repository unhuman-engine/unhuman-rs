//! Graphics pipelines.
//!
//! The shared description, its structural hash and the pipeline cache live in
//! `vulkan_pipeline_state.rs`; compute is in `vulkan_compute_pipeline.rs`.
//!
//! Created through **dynamic rendering** (`VkPipelineRenderingCreateInfo`)
//! rather than `VkRenderPass`. `chassis::RenderPassDesc` *is* the rendering info
//! and there is no render-pass object to create or destroy, so there is no
//! `VkRenderPass`-shaped attachment state to keep in sync with it.

use ash::vk;
use chassis::Pipeline;

use crate::vulkan_error::{BackendError, Result};
use crate::vulkan_pipeline_state::{PipelineHash, PipelineResource, Pipelines};
use crate::vulkan_shader::Shaders;
use crate::vulkan_types;

pub fn create(
    pipelines: &mut Pipelines,
    descriptors: &mut crate::vulkan_descriptor_manager::Descriptors,
    desc: &chassis::PipelineDesc,
    shaders: &Shaders,
) -> Result<Pipeline> {
    let hash = PipelineHash::of(desc);
    if let Some(existing) = pipelines.by_description(hash) {
        pipelines.mark_cache_hit(existing);
        return Ok(existing);
    }

    let layout = pipelines.create_layout(&desc.set_layouts, descriptors)?;
    let handle = build(pipelines, desc, shaders, layout)?;

    Ok(pipelines.insert(hash, PipelineResource::new(handle, layout, &desc.label)))
}

/// Compiles one graphics pipeline. Dynamic rendering, no `VkRenderPass`.
fn build(
    pipelines: &Pipelines,
    desc: &chassis::PipelineDesc,
    shaders: &Shaders,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let device = pipelines.device();

    // Every declared stage, in the order they were declared. A graphics pipeline
    // with no vertex stage is a `vkCreateGraphicsPipelines` error, and one with
    // no fragment stage creates fine and writes nothing, so both are caught here
    // with a message that says which.
    if !desc
        .shaders
        .iter()
        .any(|s| s.stage == chassis::ShaderStage::Vertex)
    {
        return Err(BackendError::InvalidHandle);
    }

    // Every stage's module, as one array.
    //
    // One module is one entry point, named `forge::ENTRY_POINT`: Slang renames
    // the source's entry point to that, and so does `forge::wgsl::to_spirv_entry`.
    // Naming the source's own name here creates no pipeline at all -- and the
    // driver reports it as a shader error, which points at the wrong file.
    let entry_name = std::ffi::CString::new(forge::ENTRY_POINT).expect("no interior NUL");
    let mut stage_infos: Vec<vk::PipelineShaderStageCreateInfo> =
        Vec::with_capacity(desc.shaders.len());
    for declared in &desc.shaders {
        let Some(shader) = shaders.get(declared.shader) else {
            return Err(BackendError::InvalidHandle);
        };
        stage_infos.push(vk::PipelineShaderStageCreateInfo {
            stage: shader_stage_of(declared.stage),
            module: shader.module(),
            p_name: entry_name.as_ptr(),
            ..Default::default()
        });
    }
    let stages = stage_infos;

    let attributes: Vec<vk::VertexInputAttributeDescription> = desc
        .vertex
        .attributes
        .iter()
        .map(|attribute| vk::VertexInputAttributeDescription {
            location: attribute.location,
            binding: 0,
            format: vulkan_types::vk_format(attribute.format),
            offset: attribute.offset,
        })
        .collect();

    // One binding, because the layout is flat: a second binding would need a
    // binding index on every attribute, and nothing needs one yet.
    let bindings = if attributes.is_empty() {
        Vec::new()
    } else {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: desc.vertex.stride,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    };

    // Counts written next to the slices they describe. ash's builder for these
    // structs sets the pointer but not the count, which is a trap: the count is a
    // separate field and `Default` leaves it at zero.
    // An empty slice has a dangling-but-aligned pointer; the count of zero says
    // it is empty, and the spec says the pointer is ignored then. `.as_ptr()`
    // on an empty Vec is non-null, so this is safe, but say so rather than leave
    // a reader wondering.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count: bindings.len() as u32,
        p_vertex_binding_descriptions: bindings.as_ptr(),
        vertex_attribute_description_count: attributes.len() as u32,
        p_vertex_attribute_descriptions: attributes.as_ptr(),
        ..Default::default()
    };

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(topology_of(desc.primitive.topology))
        .primitive_restart_enable(false);

    // Dynamic viewport and scissor: every pass reads the extent from the
    // swapchain, and baking one in means recompiling on every resize.
    //
    // The `pDynamicStates` list is what makes them dynamic. Without it the
    // counts below are a claim that two static arrays exist that do not, and
    // validation says so precisely:
    // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04130 and -04131.
    // Setting the counts and forgetting the dynamic states reads exactly like
    // setting them correctly, which is why it is worth naming.
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    // Struct literal again: ash's builder sets the pointer and not the count,
    // and a zero count here is the same class of bug as a zero `stage_count`.
    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states: dynamic_states.as_ptr(),
        ..Default::default()
    };

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(cull_mode_of(desc.primitive.cull_mode))
        .front_face(front_face_of(desc.primitive.front_face))
        .depth_bias_enable(false)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vulkan_types::vk_samples(desc.samples))
        .sample_shading_enable(false);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(desc.depth.format.is_some())
        .depth_write_enable(desc.depth.write_enabled)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = desc
        .color_targets
        .iter()
        .map(|target| blend_state(target.blend, target.write_mask))
        .collect();

    let color_blend = vk::PipelineColorBlendStateCreateInfo {
        logic_op_enable: vk::FALSE,
        attachment_count: color_blend_attachments.len() as u32,
        p_attachments: color_blend_attachments.as_ptr(),
        ..Default::default()
    };

    let color_formats: Vec<vk::Format> = desc
        .color_targets
        .iter()
        .map(|target| vulkan_types::vk_format(target.format))
        .collect();
    let depth_format = desc
        .depth
        .format
        .map(vulkan_types::vk_format)
        .unwrap_or(vk::Format::UNDEFINED);

    let rendering = vk::PipelineRenderingCreateInfo {
        color_attachment_count: color_formats.len() as u32,
        p_color_attachment_formats: color_formats.as_ptr(),
        depth_attachment_format: depth_format,
        ..Default::default()
    };

    // Every `*_count` beside a pointer in this struct.
    //
    // `stage_count` was the one missed, and it is the worst kind of omission: the
    // driver read zero stages, accepted an empty `pStages` pointer, and then
    // dereferenced garbage. It surfaced as a SIGSEGV inside RADV with no
    // diagnostic, because the count and the pointer disagreed and nothing checks
    // a pointer it is not asked to dereference. Validation names it in one line:
    // VUID-VkGraphicsPipelineCreateInfo-stageCount-09530.
    let create_info = vk::GraphicsPipelineCreateInfo {
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_viewport_state: &viewport_state,
        p_rasterization_state: &rasterization,
        p_multisample_state: &multisample,
        p_depth_stencil_state: &depth_stencil,
        p_color_blend_state: &color_blend,
        p_dynamic_state: &dynamic_state_info,
        layout,
        // Null render pass: dynamic rendering supplies the attachments.
        render_pass: vk::RenderPass::null(),
        subpass: 0,
        // `push_next` rather than a raw `p_next` cast, because the chained struct
        // has to carry its own `s_type` and the builder is what sets it. The
        // chain must outlive the call, which `rendering` does.
        ..Default::default()
    };
    let mut create_info = create_info;
    create_info.p_next = &rendering as *const _ as *const _;

    let pipelines_out =
        unsafe { device.create_graphics_pipelines(pipelines.driver_cache(), &[create_info], None) }
            .map_err(|(partial, error)| {
                BackendError::vk("vkCreateGraphicsPipelines", error).with_partial(partial.len())
            })?;

    let mut pipelines_out = pipelines_out;
    Ok(pipelines_out.remove(0))
}

fn blend_state(blend: bool, write_mask: u8) -> vk::PipelineColorBlendAttachmentState {
    // Standard alpha blend when asked for, otherwise a plain write.
    vk::PipelineColorBlendAttachmentState {
        blend_enable: if blend { vk::TRUE } else { vk::FALSE },
        src_color_blend_factor: if blend {
            vk::BlendFactor::SRC_ALPHA
        } else {
            vk::BlendFactor::ONE
        },
        dst_color_blend_factor: if blend {
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        } else {
            vk::BlendFactor::ZERO
        },
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::from_raw(write_mask as u32),
    }
}

fn topology_of(value: u32) -> vk::PrimitiveTopology {
    match value {
        0 => vk::PrimitiveTopology::POINT_LIST,
        1 => vk::PrimitiveTopology::LINE_LIST,
        2 => vk::PrimitiveTopology::LINE_STRIP,
        3 => vk::PrimitiveTopology::TRIANGLE_STRIP,
        _ => vk::PrimitiveTopology::TRIANGLE_LIST,
    }
}

fn cull_mode_of(value: u32) -> vk::CullModeFlags {
    match value {
        0 => vk::CullModeFlags::NONE,
        2 => vk::CullModeFlags::FRONT,
        _ => vk::CullModeFlags::BACK,
    }
}

fn front_face_of(value: u32) -> vk::FrontFace {
    if value == 1 {
        vk::FrontFace::CLOCKWISE
    } else {
        vk::FrontFace::COUNTER_CLOCKWISE
    }
}

fn shader_stage_of(stage: chassis::ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        chassis::ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        chassis::ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        chassis::ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        chassis::ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        chassis::ShaderStage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        chassis::ShaderStage::TessellationEvaluation => {
            vk::ShaderStageFlags::TESSELLATION_EVALUATION
        }
    }
}
