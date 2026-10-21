//! Real pipeline creation, on real hardware.
//!
//! The pipeline is the only place where a shader, a vertex layout, a set layout
//! and a dynamic rendering format list all have to agree. Every failure there is
//! reported as a driver error naming a number, so these tests exist to make the
//! message ours instead.

use ash::vk;
use ash::vk::Handle;
use chassis::{DescriptorKind, ShaderStage, ShaderStageDesc, StageMask};
use forge::cache::ShaderCache;
use forge::slangc::SlangcCompiler;
use forge::{CompileRequest, Slangc, Stage};
use std::path::{Path, PathBuf};
use stormbreaker::vulkan_descriptor_manager::Descriptors;
use stormbreaker::{create_device, device_available, graphic::create, BackendError, DeviceOptions};

/// A path inside the workspace, found by walking up from this crate.
///
/// Never a fixed `../../assets`. The crate moved when the workspace gained its
/// the `UHE/src` and `UHE/Stormbreaker` shape, and a depth-sensitive path that resolves
/// to nothing makes every shader test skip *while reporting success* -- the most
/// expensive bug in this project's history, because it hid a `stage_count` bug
/// that segfaulted the driver the first time a pipeline was ever created.
///
/// This is the same mistake as resolving a toolchain relative to the process's
/// working directory: `cargo test` runs from the crate, not the workspace root.
fn asset(relative: &str) -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        let candidate = ancestor.join(relative);
        if candidate.is_file() {
            return candidate;
        }
    }
    panic!("could not find {relative} above {}", manifest.display());
}

fn shader() -> PathBuf {
    asset("assets/shaders/Triangle.slang")
}

fn open_device() -> Option<stormbreaker::Context> {
    if !device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    match create_device(DeviceOptions::from_env(c"stormbreaker-pipelines")) {
        Ok(context) => Some(context),
        Err(BackendError::Loader(why)) => {
            eprintln!("SKIP: loader unavailable: {why}");
            None
        }
        Err(other) => panic!("device creation failed: {other}"),
    }
}

/// Compiles both stages and returns them.
///
/// Panics when there is no compiler, which is the policy `docs/WORK.md` states:
/// "the shader-dependent backend tests compile `.slang` at run time and panic
/// rather than skip when there is no compiler, so a job that cannot find one
/// fails loudly instead of going green having tested nothing". Every test in this
/// file is about a pipeline built from these modules, so a silent skip here is a
/// test that asserts nothing.
fn stages(
    context: &stormbreaker::Context,
    shaders: &mut stormbreaker::vulkan_shader::Shaders,
) -> (chassis::Shader, chassis::Shader) {
    let compiler = SlangcCompiler::new(Slangc::default()).unwrap_or_else(|error| {
        panic!("no slangc: {error}. Set SLANGC=/path/to/slangc; the vendored copy lives in a C++ checkout that is not in this tree")
    });
    let forge = forge::Forge::new(compiler, ShaderCache::disabled());
    let directory = std::env::temp_dir().join("stormbreaker-pipeline");
    std::fs::create_dir_all(&directory).expect("a scratch directory");
    let (vertex, fragment) = forge
        .compile_pair(&shader(), "vs_main", "fs_main", &directory)
        .expect("the test shader compiles");

    let vertex = shaders
        .create_from_words(&context.device, &vertex.words)
        .expect("a vertex module");
    let fragment = shaders
        .create_from_words(&context.device, &fragment.words)
        .expect("a fragment module");
    (vertex, fragment)
}

/// `float3 position` and `float4 color`, as the shader declares them.
///
/// Both attributes are `Rgba32Float` and the stride is 32 rather than 28,
/// because there is no packed RGB float format. Vulkan allows a format with more
/// components than the shader input, so a `float3` input fed from RGBA is
/// legal and the fourth component is dropped.
fn vertex_layout() -> chassis::VertexLayout {
    chassis::VertexLayout {
        stride: 32,
        attributes: vec![
            chassis::VertexAttribute {
                location: 0,
                format: chassis::TextureFormat::Rgba32Float,
                offset: 0,
            },
            chassis::VertexAttribute {
                location: 1,
                format: chassis::TextureFormat::Rgba32Float,
                offset: 16,
            },
        ],
    }
}

fn desc_with(
    vertex: chassis::Shader,
    fragment: Option<chassis::Shader>,
    set_layouts: Vec<chassis::DescriptorSetLayoutDesc>,
) -> chassis::PipelineDesc {
    let mut shaders = vec![ShaderStageDesc {
        stage: ShaderStage::Vertex,
        shader: vertex,
    }];
    if let Some(fragment) = fragment {
        shaders.push(ShaderStageDesc {
            stage: ShaderStage::Fragment,
            shader: fragment,
        });
    }
    chassis::PipelineDesc {
        label: "triangle".to_owned(),
        shaders,
        set_layouts,
        vertex: vertex_layout(),
        index_format: None,
        primitive: chassis::PrimitiveState::default(),
        depth: chassis::DepthState::default(),
        blend: chassis::BlendState::default(),
        color_targets: vec![chassis::ColorTargetState {
            format: chassis::TextureFormat::Rgba8Unorm,
            blend: false,
            write_mask: 0xf,
        }],
        samples: chassis::SampleCount::X1,
    }
}

/// A vertex and a fragment module in one pipeline. Before this existed the
/// backend hardcoded `stages: [vertex]`, so a pipeline could never shade.
#[test]
fn a_vertex_and_a_fragment_stage_create_one_pipeline() {
    let Some(context) = open_device() else { return };
    let mut shaders = stormbreaker::vulkan_shader::Shaders::with_device(Some(&context.device));
    let (vertex, fragment) = stages(&context, &mut shaders);

    let mut pipelines = stormbreaker::vulkan_pipeline_state::Pipelines::new(
        &context.device,
        vk::PipelineCache::null(),
    );
    let mut descriptors = Descriptors::new(&context.device);
    let handle = create(
        &mut pipelines,
        &mut descriptors,
        &desc_with(vertex, Some(fragment), Vec::new()),
        &shaders,
    )
    .expect("a two-stage pipeline creates");

    let resource = pipelines.get(handle).expect("handle resolves");
    assert!(!resource.handle().is_null());
    assert!(!resource.layout().is_null(), "and it has a real layout");
}

/// The pipeline layout used to be built with zero descriptor sets, so a shader
/// that read any descriptor could not be bound. This creates a layout with a real
/// set and checks the driver agrees on the shape.
#[test]
fn a_pipeline_with_a_descriptor_set_gets_a_matching_layout() {
    let Some(context) = open_device() else { return };
    let mut shaders = stormbreaker::vulkan_shader::Shaders::with_device(Some(&context.device));
    let (vertex, fragment) = stages(&context, &mut shaders);

    let set = chassis::DescriptorSetLayoutDesc::new("frame")
        .with(chassis::BindingDesc::new(
            0,
            DescriptorKind::UniformBuffer,
            StageMask::of(ShaderStage::Vertex).union(StageMask::of(ShaderStage::Fragment)),
        ))
        .with(chassis::BindingDesc::new(
            1,
            DescriptorKind::CombinedImageSampler,
            StageMask::of(ShaderStage::Fragment),
        ));

    let mut pipelines = stormbreaker::vulkan_pipeline_state::Pipelines::new(
        &context.device,
        vk::PipelineCache::null(),
    );
    let description = desc_with(vertex, Some(fragment), vec![set]);
    let mut descriptors = Descriptors::new(&context.device);
    create(&mut pipelines, &mut descriptors, &description, &shaders)
        .expect("a pipeline with one set");

    assert_eq!(
        descriptors.layout_count(),
        1,
        "one set layout created, not zero",
    );
}

/// Two sets with the same shape are one `VkDescriptorSetLayout`. Keying on the
/// label would make a rename change the driver objects for no reason.
#[test]
fn identically_shaped_sets_share_one_layout() {
    let Some(context) = open_device() else { return };
    let mut shaders = stormbreaker::vulkan_shader::Shaders::with_device(Some(&context.device));
    let (vertex, fragment) = stages(&context, &mut shaders);

    let shape = |label: &str| {
        chassis::DescriptorSetLayoutDesc::new(label).with(chassis::BindingDesc::new(
            0,
            DescriptorKind::UniformBuffer,
            StageMask::of(ShaderStage::Vertex),
        ))
    };

    let mut pipelines = stormbreaker::vulkan_pipeline_state::Pipelines::new(
        &context.device,
        vk::PipelineCache::null(),
    );
    let mut descriptors = Descriptors::new(&context.device);
    create(
        &mut pipelines,
        &mut descriptors,
        &desc_with(vertex, Some(fragment), vec![shape("first")]),
        &shaders,
    )
    .expect("first pipeline");
    create(
        &mut pipelines,
        &mut descriptors,
        &desc_with(vertex, Some(fragment), vec![shape("second")]),
        &shaders,
    )
    .expect("second pipeline");

    assert_eq!(
        descriptors.layout_count(),
        1,
        "the same shape under two labels is still one driver layout",
    );
}

/// The description hash decides whether a pipeline is reused. It covered one
/// shader index and nothing else, so swapping the fragment module would have
/// returned the cached pipeline and drawn with the old shader.
#[test]
fn a_different_fragment_shader_is_a_different_pipeline() {
    let Some(context) = open_device() else { return };
    let mut shaders = stormbreaker::vulkan_shader::Shaders::with_device(Some(&context.device));
    let (vertex, fragment) = stages(&context, &mut shaders);

    let compiler = SlangcCompiler::new(Slangc::default()).expect("slangc");
    let forge = forge::Forge::new(compiler, ShaderCache::disabled());
    let directory = std::env::temp_dir().join("stormbreaker-pipeline-swap");
    std::fs::create_dir_all(&directory).expect("scratch");

    // The same source, a different compile: different words, different handle.
    let alternate = forge
        .compile(
            &CompileRequest::new(shader(), "fs_main", Stage::Fragment)
                .optimization(forge::Optimization::O2),
            &directory.join("alt.frag.spv"),
        )
        .expect("alternate fragment stage");
    let other_fragment = shaders
        .create_from_words(&context.device, &alternate.words)
        .expect("second fragment module");

    let mut pipelines = stormbreaker::vulkan_pipeline_state::Pipelines::new(
        &context.device,
        vk::PipelineCache::null(),
    );
    let mut descriptors = Descriptors::new(&context.device);
    let first = create(
        &mut pipelines,
        &mut descriptors,
        &desc_with(vertex, Some(fragment), Vec::new()),
        &shaders,
    )
    .expect("first pipeline");
    let second = create(
        &mut pipelines,
        &mut descriptors,
        &desc_with(vertex, Some(other_fragment), Vec::new()),
        &shaders,
    )
    .expect("second pipeline");

    assert_ne!(
        first, second,
        "a different fragment module must not hit the description cache",
    );
    let (_, hits) = pipelines.stats();
    assert_eq!(hits, 0, "and the cache must not have been consulted");
}

/// The same description twice is one pipeline. A hash that misses here recompiles
/// every frame.
#[test]
fn the_same_description_is_compiled_once() {
    let Some(context) = open_device() else { return };
    let mut shaders = stormbreaker::vulkan_shader::Shaders::with_device(Some(&context.device));
    let (vertex, fragment) = stages(&context, &mut shaders);

    let mut pipelines = stormbreaker::vulkan_pipeline_state::Pipelines::new(
        &context.device,
        vk::PipelineCache::null(),
    );
    let description = desc_with(vertex, Some(fragment), Vec::new());
    let mut descriptors = Descriptors::new(&context.device);
    let first = create(&mut pipelines, &mut descriptors, &description, &shaders).expect("first");
    let second = create(&mut pipelines, &mut descriptors, &description, &shaders).expect("second");

    assert_eq!(first, second, "the same description is the same pipeline");
    let (created, hits) = pipelines.stats();
    assert_eq!(created, 1, "compiled once");
    assert_eq!(hits, 1, "and the second request was a cache hit");
}

/// A pipeline with no vertex stage cannot be created. Catching it here gives a
/// message that names the stage; the driver's names a number.
#[test]
fn a_pipeline_without_a_vertex_stage_is_refused() {
    let Some(context) = open_device() else { return };
    let mut shaders = stormbreaker::vulkan_shader::Shaders::with_device(Some(&context.device));
    let (vertex, fragment) = stages(&context, &mut shaders);

    let mut description = desc_with(vertex, Some(fragment), Vec::new());
    description
        .shaders
        .retain(|stage| stage.stage != ShaderStage::Vertex);

    let mut pipelines = stormbreaker::vulkan_pipeline_state::Pipelines::new(
        &context.device,
        vk::PipelineCache::null(),
    );
    let mut descriptors = Descriptors::new(&context.device);
    assert!(
        create(&mut pipelines, &mut descriptors, &description, &shaders).is_err(),
        "a pipeline with no vertex stage must not be created",
    );
}
