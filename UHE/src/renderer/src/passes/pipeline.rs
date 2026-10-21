//! Pipelines: what a pipeline is keyed by, and where the shaders come from.
//!
//! The model pipeline and its double-sided variant differ only in cull mode, so
//! cull mode is part of the cache key rather than state a draw can change.
//!
//! Shaders are WGSL, compiled in process by `forge::wgsl`, so the renderer's own
//! shaders need no external toolchain and both backends take the same SPIR-V.

use std::collections::HashMap;

use chassis::{
    BlendState, ColorTargetState, DepthState, DescriptorKind, DescriptorSetLayoutDesc, IndexFormat,
    Pipeline, PipelineDesc, PrimitiveState, Resources, SampleCount, Shader, ShaderStage,
    ShaderStageDesc, StageMask, TextureFormat, VertexLayout,
};

use crate::mesh::{self, VertexLayoutKey};

pub const FRAME_SET: u32 = 0;
pub const MATERIAL_SET: u32 = 1;
pub const DRAW_SET: u32 = 2;
/// The model pipeline's skin palette, after the per-draw set.
pub const PALETTE_SET: u32 = 3;

/// The shader sources, embedded rather than resolved at run time.
///
/// `include_str!` makes a missing source a compile error rather than a silent null
/// handle, and a test compiles every one of them to SPIR-V.
pub const MODEL_WGSL: &str = include_str!("../../assets/shaders/model.wgsl");
pub const DEPTH_WGSL: &str = include_str!("../../assets/shaders/depth.wgsl");
pub const SSAO_WGSL: &str = include_str!("../../assets/shaders/ssao.wgsl");
pub const SSAO_UPSAMPLE_WGSL: &str = include_str!("../../assets/shaders/ssao_upsample.wgsl");
pub const GRID_WGSL: &str = include_str!("../../assets/shaders/grid.wgsl");
pub const LINE_WGSL: &str = include_str!("../../assets/shaders/line.wgsl");
pub const COMPOSITE_WGSL: &str = include_str!("../../assets/shaders/composite.wgsl");

pub const VS_MAIN: &str = "vs_main";
pub const FS_MAIN: &str = "fs_main";

fn graphics_stages() -> StageMask {
    StageMask::of(ShaderStage::Vertex).union(StageMask::of(ShaderStage::Fragment))
}

/// Set 0: the frame.
///
/// The view-projection, the light list, and the shadow map's depth texture. Push
/// constants and a bindless array are not expressible through this RHI, so the
/// frame is a set like any other.
pub fn frame_layout() -> DescriptorSetLayoutDesc {
    DescriptorSetLayoutDesc::new("frame")
        .with(binding(0, DescriptorKind::UniformBuffer, graphics_stages()))
        .with(binding(1, DescriptorKind::StorageBuffer, graphics_stages()))
        // Two bindings, not a combined sampler: wgpu has no combined descriptor
        // and refuses one. The same pair is what a WGSL `texture_depth_2d` and
        // `sampler` compile to, so one layout serves both backends.
        .with(binding(
            2,
            DescriptorKind::SampledDepth,
            StageMask::of(ShaderStage::Fragment),
        ))
        .with(binding(
            3,
            DescriptorKind::Sampler,
            StageMask::of(ShaderStage::Fragment),
        ))
}

/// Set 1: one material's parameters and its maps.
///
/// A map the material does not declare binds the 1x1 white texture, so no slot is
/// ever unbound.
pub fn material_layout() -> DescriptorSetLayoutDesc {
    let fragment = StageMask::of(ShaderStage::Fragment);
    let mut layout = DescriptorSetLayoutDesc::new("material")
        .with(binding(0, DescriptorKind::UniformBuffer, fragment))
        .with(binding(1, DescriptorKind::Sampler, fragment));
    for slot in 0..crate::material::SLOT_COUNT {
        layout = layout.with(binding(
            2 + slot as u32,
            DescriptorKind::SampledImage,
            fragment,
        ));
    }
    layout
}

/// Set 2: one draw's model matrix, id and tint.
pub fn draw_layout() -> DescriptorSetLayoutDesc {
    DescriptorSetLayoutDesc::new("draw").with(binding(
        0,
        DescriptorKind::UniformBuffer,
        graphics_stages(),
    ))
}

/// The skin palette: every bone matrix the frame's skinned draws can index.
pub fn palette_layout() -> DescriptorSetLayoutDesc {
    DescriptorSetLayoutDesc::new("palette").with(binding(
        0,
        DescriptorKind::StorageBuffer,
        StageMask::of(ShaderStage::Vertex),
    ))
}

/// The prepass's set: the light-space view-projection. Vertex stage only; the
/// depth pipeline has no fragment stage at all.
pub fn shadow_layout() -> DescriptorSetLayoutDesc {
    DescriptorSetLayoutDesc::new("shadow").with(binding(
        0,
        DescriptorKind::UniformBuffer,
        StageMask::of(ShaderStage::Vertex),
    ))
}

/// The line pass's set: the camera's matrix and the batch's own.
pub fn line_layout() -> DescriptorSetLayoutDesc {
    DescriptorSetLayoutDesc::new("line").with(binding(
        0,
        DescriptorKind::UniformBuffer,
        graphics_stages(),
    ))
}

/// The grid pass's set: the view-projection and its inverse.
pub fn grid_layout() -> DescriptorSetLayoutDesc {
    DescriptorSetLayoutDesc::new("grid").with(binding(
        0,
        DescriptorKind::UniformBuffer,
        graphics_stages(),
    ))
}

/// The SSAO pass's set: the kernel, and the depth buffer it reads.
pub fn ssao_layout() -> DescriptorSetLayoutDesc {
    let fragment = StageMask::of(ShaderStage::Fragment);
    DescriptorSetLayoutDesc::new("ssao")
        .with(binding(0, DescriptorKind::UniformBuffer, fragment))
        .with(binding(1, DescriptorKind::SampledDepth, fragment))
        .with(binding(2, DescriptorKind::Sampler, fragment))
}

/// The upsample pass's set: the same kernel and depth as SSAO, plus the
/// half-resolution occlusion it reads.
pub fn ssao_upsample_layout() -> DescriptorSetLayoutDesc {
    let fragment = StageMask::of(ShaderStage::Fragment);
    DescriptorSetLayoutDesc::new("ssao_upsample")
        .with(binding(0, DescriptorKind::UniformBuffer, fragment))
        .with(binding(1, DescriptorKind::SampledDepth, fragment))
        .with(binding(2, DescriptorKind::Sampler, fragment))
        .with(binding(3, DescriptorKind::SampledImage, fragment))
}

/// The composite pass's set: the scene colour, the ambient occlusion, the
/// sampler both are read with, and the exposure/tonemap parameters.
pub fn composite_layout() -> DescriptorSetLayoutDesc {
    let fragment = StageMask::of(ShaderStage::Fragment);
    DescriptorSetLayoutDesc::new("composite")
        .with(binding(0, DescriptorKind::UniformBuffer, fragment))
        .with(binding(1, DescriptorKind::SampledImage, fragment))
        .with(binding(2, DescriptorKind::SampledImage, fragment))
        .with(binding(3, DescriptorKind::Sampler, fragment))
}

fn binding(binding: u32, kind: DescriptorKind, stages: StageMask) -> chassis::BindingDesc {
    chassis::BindingDesc::new(binding, kind, stages)
}

/// The shader modules every pipeline is built from.
///
/// One module per source: the module carries every entry point the source
/// declares, and a pipeline stage names the one it needs from that module.
pub struct ShaderSet {
    pub model: Shader,
    pub depth: Shader,
    pub ssao: Shader,
    pub ssao_upsample: Shader,
    pub grid: Shader,
    pub line: Shader,
    pub composite: Shader,
}

impl ShaderSet {
    pub fn load(resources: &mut dyn Resources) -> Result<Self, chassis::Error> {
        Ok(Self {
            model: module(resources, "model", MODEL_WGSL)?,
            depth: module(resources, "depth", DEPTH_WGSL)?,
            ssao: module(resources, "ssao", SSAO_WGSL)?,
            ssao_upsample: module(resources, "ssao_upsample", SSAO_UPSAMPLE_WGSL)?,
            grid: module(resources, "grid", GRID_WGSL)?,
            line: module(resources, "line", LINE_WGSL)?,
            composite: module(resources, "composite", COMPOSITE_WGSL)?,
        })
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources) {
        for shader in self.all() {
            resources.destroy_shader(shader);
        }
    }

    fn all(&self) -> [Shader; 7] {
        [
            self.model,
            self.depth,
            self.ssao,
            self.ssao_upsample,
            self.grid,
            self.line,
            self.composite,
        ]
    }

    fn vertex_for(&self, variant: ShaderVariant) -> Shader {
        match variant {
            ShaderVariant::Model
            | ShaderVariant::ModelDoubleSided
            | ShaderVariant::ModelTransparent
            | ShaderVariant::ModelTransparentDoubleSided => self.model,
            ShaderVariant::Shadow => self.depth,
            ShaderVariant::Ssao => self.ssao,
            ShaderVariant::SsaoUpsample => self.ssao_upsample,
            ShaderVariant::Grid => self.grid,
            ShaderVariant::Line => self.line,
            ShaderVariant::Composite => self.composite,
        }
    }

    fn fragment_for(&self, variant: ShaderVariant) -> Option<Shader> {
        match variant {
            ShaderVariant::Model
            | ShaderVariant::ModelDoubleSided
            | ShaderVariant::ModelTransparent
            | ShaderVariant::ModelTransparentDoubleSided => Some(self.model),
            // A depth-only pipeline has no fragment stage: the prepass is the same
            // draw with no colour target, and Vulkan allows a pipeline with only a
            // vertex stage.
            ShaderVariant::Shadow => None,
            ShaderVariant::Ssao => Some(self.ssao),
            ShaderVariant::SsaoUpsample => Some(self.ssao_upsample),
            ShaderVariant::Grid => Some(self.grid),
            ShaderVariant::Line => Some(self.line),
            ShaderVariant::Composite => Some(self.composite),
        }
    }
}

/// Compiles one embedded source to SPIR-V, keeping every entry point it declares.
fn module(
    resources: &mut dyn Resources,
    label: &str,
    source: &str,
) -> Result<Shader, chassis::Error> {
    let spirv = forge::wgsl::to_spirv(source, label)
        .map_err(|error| chassis::Error::InvalidDescriptor(error.to_string()))?;
    resources.create_shader(label, &spirv)
}

/// Which pipeline a draw wants.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShaderVariant {
    Model,
    ModelDoubleSided,
    ModelTransparent,
    ModelTransparentDoubleSided,
    Shadow,
    Ssao,
    SsaoUpsample,
    Grid,
    Line,
    Composite,
}

impl ShaderVariant {
    pub fn blends(self) -> bool {
        matches!(
            self,
            ShaderVariant::ModelTransparent | ShaderVariant::ModelTransparentDoubleSided
        )
    }

    pub fn culls_none(self) -> bool {
        matches!(
            self,
            ShaderVariant::ModelDoubleSided
                | ShaderVariant::ModelTransparentDoubleSided
                // The grid is a full-screen quad whose winding depends on the
                // view, and a gizmo line has no facing at all.
                | ShaderVariant::Grid
                | ShaderVariant::Line
                | ShaderVariant::Ssao
                | ShaderVariant::SsaoUpsample
                | ShaderVariant::Composite
        )
    }
}

/// Everything a pipeline is built from.
///
/// A pipeline is immutable state, so every field that changes it has to be in
/// the key or two draws with different state share one pipeline. This is the
/// list `PipelineDesc` declares, minus the label.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PipelineKey {
    pub variant: ShaderVariant,
    pub vertex: VertexLayoutKey,
    pub color_targets: Vec<TextureFormat>,
    pub depth: Option<TextureFormat>,
    pub depth_write: bool,
    pub depth_compare: u32,
    pub samples: SampleCount,
    pub index_format: Option<IndexFormat>,
}

impl PipelineKey {
    /// An opaque model pipeline for one colour target and one depth format.
    pub fn model(
        variant: ShaderVariant,
        color: TextureFormat,
        depth: TextureFormat,
        samples: SampleCount,
    ) -> Self {
        Self {
            variant,
            vertex: VertexLayoutKey::of(&mesh::vertex_layout()),
            color_targets: vec![color],
            depth: Some(depth),
            depth_write: true,
            // `VK_COMPARE_OP_LESS`, the default depth test.
            depth_compare: 1,
            samples,
            index_format: Some(mesh::INDEX_FORMAT),
        }
    }

    /// The depth-only prepass: no colour target, so the vertex stage is the
    /// whole pipeline.
    ///
    /// It draws with `depth.wgsl`, which reads position and the skin attributes
    /// alone; declaring normal, UV, tangent or colour would fetch them for
    /// nothing.
    pub fn shadow(depth: TextureFormat, samples: SampleCount) -> Self {
        Self {
            variant: ShaderVariant::Shadow,
            vertex: VertexLayoutKey::of(&mesh::depth_vertex_layout()),
            color_targets: Vec::new(),
            depth: Some(depth),
            depth_write: true,
            depth_compare: 1,
            samples,
            index_format: Some(mesh::INDEX_FORMAT),
        }
    }

    /// A full-screen pass: no vertex buffer at all, the triangle comes from
    /// `@builtin(vertex_index)`. No depth attachment: a pass cannot load the
    /// depth a previous pass wrote (see this module and `GeometryPass`), so a
    /// full-screen pass either declares its own cleared depth or none at all.
    pub fn fullscreen(variant: ShaderVariant, color: TextureFormat, samples: SampleCount) -> Self {
        Self {
            variant,
            vertex: VertexLayoutKey::of(&VertexLayout::default()),
            color_targets: vec![color],
            depth: None,
            depth_write: false,
            depth_compare: 1,
            samples,
            index_format: None,
        }
    }

    /// The grid: six vertices from `@builtin(vertex_index)`, depth-tested against
    /// the scene depth so geometry occludes it. It writes no depth, so it cannot
    /// hide the gizmos drawn in the same pass.
    pub fn grid(color: TextureFormat, depth: TextureFormat, samples: SampleCount) -> Self {
        Self {
            variant: ShaderVariant::Grid,
            vertex: VertexLayoutKey::of(&VertexLayout::default()),
            color_targets: vec![color],
            depth: Some(depth),
            depth_write: false,
            depth_compare: 1,
            samples,
            index_format: None,
        }
    }

    /// A line list, which is what a gizmo is. Depth-tested but not depth-writing:
    /// a gizmo behind geometry is hidden without occluding later draws.
    pub fn lines(color: TextureFormat, depth: TextureFormat, samples: SampleCount) -> Self {
        Self {
            variant: ShaderVariant::Line,
            vertex: VertexLayoutKey::of(&line_vertex_layout()),
            color_targets: vec![color],
            depth: Some(depth),
            depth_write: false,
            depth_compare: 1,
            samples,
            index_format: Some(mesh::INDEX_FORMAT),
        }
    }
}

/// The layout a line vertex is drawn with.
pub fn line_vertex_layout() -> VertexLayout {
    VertexLayout {
        stride: std::mem::size_of::<crate::passes::lines::LineVertex>() as u32,
        attributes: vec![
            chassis::VertexAttribute {
                location: 0,
                format: TextureFormat::R32G32B32Float,
                offset: std::mem::offset_of!(crate::passes::lines::LineVertex, position) as u32,
            },
            chassis::VertexAttribute {
                location: 1,
                format: TextureFormat::Rgba32Float,
                offset: std::mem::offset_of!(crate::passes::lines::LineVertex, color) as u32,
            },
        ],
    }
}

/// The pipeline description for a key. Pure, so the state a key implies is
/// testable with no device.
pub fn pipeline_desc(key: &PipelineKey, shaders: &ShaderSet) -> PipelineDesc {
    let mut stages = vec![ShaderStageDesc {
        stage: ShaderStage::Vertex,
        shader: shaders.vertex_for(key.variant),
    }];
    if let Some(fragment) = shaders.fragment_for(key.variant) {
        stages.push(ShaderStageDesc {
            stage: ShaderStage::Fragment,
            shader: fragment,
        });
    }

    let set_layouts = match key.variant {
        ShaderVariant::Model
        | ShaderVariant::ModelDoubleSided
        | ShaderVariant::ModelTransparent
        | ShaderVariant::ModelTransparentDoubleSided => {
            vec![
                frame_layout(),
                material_layout(),
                draw_layout(),
                palette_layout(),
            ]
        }
        // The prepass writes shadow depth alone: it needs the light's matrix in
        // set 0 and the model matrix in set 1, and nothing else. Declaring a
        // read of the shadow map here would be a pass that reads the resource it
        // writes, which the graph refuses.
        ShaderVariant::Shadow => vec![shadow_layout(), draw_layout(), palette_layout()],
        ShaderVariant::Line => vec![line_layout()],
        ShaderVariant::Ssao => vec![ssao_layout()],
        ShaderVariant::SsaoUpsample => vec![ssao_upsample_layout()],
        ShaderVariant::Composite => vec![composite_layout()],
        // The grid's matrices are a uniform block, because this RHI cannot declare
        // push constants.
        ShaderVariant::Grid => vec![grid_layout()],
    };

    let vertex = match key.variant {
        ShaderVariant::Model
        | ShaderVariant::ModelDoubleSided
        | ShaderVariant::ModelTransparent
        | ShaderVariant::ModelTransparentDoubleSided => mesh::vertex_layout(),
        ShaderVariant::Shadow => mesh::depth_vertex_layout(),
        ShaderVariant::Line => line_vertex_layout(),
        ShaderVariant::Ssao
        | ShaderVariant::SsaoUpsample
        | ShaderVariant::Grid
        | ShaderVariant::Composite => VertexLayout::default(),
    };

    PipelineDesc {
        label: format!("{:?}", key.variant),
        shaders: stages,
        set_layouts,
        vertex,
        index_format: key.index_format,
        primitive: PrimitiveState {
            // Triangle list (4) or line list (1), the same numbering
            // `chassis::PrimitiveState::default` documents.
            topology: if key.variant == ShaderVariant::Line {
                1
            } else {
                4
            },
            cull_mode: if key.variant.culls_none() { 0 } else { 1 },
            front_face: 0,
        },
        depth: DepthState {
            format: key.depth,
            write_enabled: key.depth_write,
            compare: key.depth_compare,
        },
        blend: BlendState {
            enabled: key.variant.blends(),
        },
        color_targets: key
            .color_targets
            .iter()
            .map(|format| ColorTargetState {
                format: *format,
                blend: key.variant.blends(),
                write_mask: 0xF,
            })
            .collect(),
        samples: key.samples,
    }
}

/// Pipelines, keyed by what they were built from.
pub struct PipelineCache {
    entries: HashMap<PipelineKey, Pipeline>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn get(&self, key: &PipelineKey) -> Option<Pipeline> {
        self.entries.get(key).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The pipeline for this key, created once.
    ///
    /// `create_pipeline` is expensive, so caching by key keeps a scene with forty
    /// double-sided materials from creating forty pipelines.
    pub fn get_or_create(
        &mut self,
        resources: &mut dyn Resources,
        key: PipelineKey,
        shaders: &ShaderSet,
    ) -> Result<Pipeline, chassis::Error> {
        if let Some(pipeline) = self.entries.get(&key) {
            return Ok(*pipeline);
        }
        let pipeline = resources.create_pipeline(pipeline_desc(&key, shaders))?;
        self.entries.insert(key, pipeline);
        Ok(pipeline)
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources, key: &PipelineKey) -> bool {
        let Some(pipeline) = self.entries.remove(key) else {
            return false;
        };
        resources.destroy_pipeline(pipeline);
        true
    }

    pub fn clear(&mut self, resources: &mut dyn Resources) {
        for (_, pipeline) in self.entries.drain() {
            resources.destroy_pipeline(pipeline);
        }
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}
