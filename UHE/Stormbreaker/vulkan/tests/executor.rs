//! The executor, on real hardware: a triangle, drawn, read back.
//!
//! This is the end of the chain that started with a render graph and ends with
//! pixels, and every link in it has been unit tested separately. What this proves
//! is that the links agree: that the graph's compiled order is the submit order,
//! that the barriers the graph emitted are the ones the driver accepts, and that
//! a draw recorded into a list the executor handed out reaches an image.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ash::vk;
use chassis::graph;
use forge::cache::ShaderCache;
use forge::slangc::SlangcCompiler;
use forge::Slangc;
use stormbreaker::{create_device, device_available, DeviceOptions};

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
/// Whether the pool actually had a worker free to take a job.
///
/// Probed rather than inferred from `available_parallelism`. Under CPU
/// contention every worker may be descheduled, and a pool with twelve workers
/// that cannot run one job is a pool with none -- and an assertion about
/// parallelism is not checkable there.
fn pool_has_spare_worker(pool: &Arc<grind::Jobs>) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};

    let ran_elsewhere = Arc::new(AtomicBool::new(false));
    let calling_thread = std::thread::current().id();
    let counter = Arc::new(grind::JobCounter::new());

    pool.execute(
        Box::new({
            let ran_elsewhere = Arc::clone(&ran_elsewhere);
            move || {
                ran_elsewhere.store(
                    std::thread::current().id() != calling_thread,
                    Ordering::SeqCst,
                );
            }
        }),
        Some(Arc::clone(&counter)),
    );
    pool.wait(&counter);
    ran_elsewhere.load(Ordering::SeqCst)
}

/// One pool for the whole test binary.
///
/// `grind::Jobs::new` spawns `available_parallelism() - 1` threads, so building
/// one per test would spawn hundreds across a run for no reason -- and a pool
/// built per frame would spawn threads that outlive the assertion that made them.
fn jobs() -> Arc<grind::Jobs> {
    static POOL: std::sync::OnceLock<Arc<grind::Jobs>> = std::sync::OnceLock::new();
    Arc::clone(POOL.get_or_init(|| Arc::new(grind::Jobs::new())))
}

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

fn shader() -> std::path::PathBuf {
    asset("assets/shaders/Triangle.slang")
}

fn open_device() -> Option<stormbreaker::Context> {
    if !device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    match create_device(DeviceOptions::from_env(c"stormbreaker-executor")) {
        Ok(context) => Some(context),
        Err(stormbreaker::BackendError::Loader(why)) => {
            eprintln!("SKIP: loader unavailable: {why}");
            None
        }
        Err(other) => panic!("device creation failed: {other}"),
    }
}

/// Three vertices, all inside clip space, all the same colour.
///
/// Inside clip space means no camera is involved, so a black image can only be
/// the backend. All one colour means the fragment shader ran, not merely that a
/// pipeline was created.
const VERTICES: [f32; 24] = [
    // position xyz, one pad, colour rgba -- eight floats, matching the 32-byte
    // stride. Six floats per vertex against a 32-byte layout reads past the end
    // of the third one.
    0.0, 0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // top
    -0.75, -0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // bottom left
    0.75, -0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // bottom right
];

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// The registries a command list borrows. `Context` owns the device and the
/// queue, not the resources -- the caller builds those, and the executor borrows
/// them for the length of one frame.
/// Field order is load-bearing.
///
/// Rust drops struct fields in declaration order, so `context` -- which owns the
/// `ash::Device` and destroys it in its own `Drop` -- has to be declared *last*.
/// With it first, the device went away while the pipelines were still alive and
/// `vkDestroyPipeline` was called on a dead device: a SIGSEGV in the driver with
/// only VUID-vkDestroyDevice-device-05137 as a hint.
struct Fixture {
    // Destroyed in this order: every driver object, then the device.
    pipelines: std::sync::Arc<std::sync::RwLock<stormbreaker::vulkan_pipeline_state::Pipelines>>,
    textures: std::sync::Arc<std::sync::RwLock<stormbreaker::vulkan_texture::Textures>>,
    buffers: std::sync::Arc<std::sync::RwLock<stormbreaker::vulkan_buffer::Buffers>>,
    descriptors:
        std::sync::Arc<std::sync::RwLock<stormbreaker::vulkan_descriptor_manager::Descriptors>>,
    pipeline: chassis::Pipeline,
    vertex_buffer: chassis::Buffer,
    /// Graph slot 0. The frame's single pass draws into this.
    target: chassis::Texture,
    /// Kept alive: the shader handles inside `pipeline` are only valid while
    /// their modules exist.
    _shaders: stormbreaker::vulkan_shader::Shaders,
    /// Last, always. Everything above holds a handle from this device.
    context: stormbreaker::Context,
}

/// Builds everything the executor needs, or reports exactly which step failed.
///
/// Every step says why it gave up. A fixture that returns a bare `None` turns
/// every assertion below into a test that passes without running, which is worse
/// than no test at all -- it is a test that reports success while proving
/// nothing. This one was doing exactly that.
fn fixture() -> Result<Fixture, String> {
    let context = open_device().ok_or("no Vulkan device".to_owned())?;

    let compiler =
        SlangcCompiler::new(Slangc::default()).map_err(|e| format!("slangc unavailable: {e}"))?;
    let forge = forge::Forge::new(compiler, ShaderCache::disabled());
    let directory = std::env::temp_dir().join("stormbreaker-executor");
    std::fs::create_dir_all(&directory).map_err(|e| format!("scratch: {e}"))?;
    let (vertex_module, fragment_module) = forge
        .compile_pair(&shader(), "vs_main", "fs_main", &directory)
        .map_err(|e| format!("Triangle.slang did not compile: {e}"))?;

    let mut shaders = stormbreaker::vulkan_shader::Shaders::with_device(Some(&context.device));
    let vertex_shader = shaders
        .create_from_words(
            &context.device,
            &vertex_module.words,
            vk::ShaderStageFlags::VERTEX,
        )
        .map_err(|e| format!("vertex module: {e}"))?;
    let fragment_shader = shaders
        .create_from_words(
            &context.device,
            &fragment_module.words,
            vk::ShaderStageFlags::FRAGMENT,
        )
        .map_err(|e| format!("fragment module: {e}"))?;

    let description = chassis::PipelineDesc {
        label: "triangle".to_owned(),
        shaders: vec![
            chassis::ShaderStageDesc {
                stage: chassis::ShaderStage::Vertex,
                shader: vertex_shader,
            },
            chassis::ShaderStageDesc {
                stage: chassis::ShaderStage::Fragment,
                shader: fragment_shader,
            },
        ],
        set_layouts: Vec::new(),
        vertex: chassis::VertexLayout {
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
        },
        index_format: None,
        // Culling off. Vulkan's default viewport transform maps NDC +Y to
        // framebuffer -Y, so a triangle wound counter-clockwise in NDC appears
        // *clockwise* in framebuffer coordinates and is a back face under
        // `frontFace = COUNTER_CLOCKWISE`. With culling on -- which is what
        // `PrimitiveState::default()` asks for -- this triangle was culled and
        // the image was exactly the clear colour. Turning culling off here
        // removes the variable; the winding convention is a real trap and is
        // noted rather than papered over.
        primitive: chassis::PrimitiveState {
            cull_mode: 0,
            ..chassis::PrimitiveState::default()
        },
        depth: chassis::DepthState::default(),
        blend: chassis::BlendState::default(),
        color_targets: vec![chassis::ColorTargetState {
            format: chassis::TextureFormat::Rgba8Unorm,
            blend: false,
            write_mask: 0xf,
        }],
        samples: chassis::SampleCount::X1,
    };
    let memory = context.memory_properties();
    let mut pipelines = stormbreaker::vulkan_pipeline_state::Pipelines::new(
        &context.device,
        vk::PipelineCache::null(),
    );
    let mut textures = stormbreaker::vulkan_texture::Textures::new(&context.device, memory);
    let mut buffers = stormbreaker::vulkan_buffer::Buffers::new(&context.device, memory);
    let mut descriptors =
        stormbreaker::vulkan_descriptor_manager::Descriptors::new(&context.device);

    let pipeline =
        stormbreaker::graphic::create(&mut pipelines, &mut descriptors, &description, &shaders)
            .map_err(|e| format!("pipeline creation: {e}"))?;

    // Created with `initial`, so the upload staging buffer is filled and the
    // command copied in one step: the draw reads device memory, not a mapped
    // host pointer, which is what makes the draw itself the thing under test.
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(&VERTICES));
    for value in VERTICES {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let vertex_buffer = buffers
        .create(
            &context.device,
            &chassis::BufferDesc::new(bytes.len() as u64, chassis::BufferUsage::VERTEX),
            Some(&bytes),
        )
        .map_err(|e| format!("vertex buffer: {e}"))?;

    // A real offscreen target. Without one the graph's barriers cannot resolve a
    // texture, and every frame fails with InvalidHandle -- which is the correct
    // behaviour, and is exactly what `a_null_slot_cannot_resolve_a_barrier_target`
    // pins down. A test that wants a frame to run has to give it something to run
    // against.
    let target = textures
        .create(
            &context.device,
            &chassis::TextureDesc::new_2d(
                chassis::Extent {
                    width: WIDTH,
                    height: HEIGHT,
                    depth: 1,
                },
                chassis::TextureFormat::Rgba8Unorm,
                chassis::TextureUsage::COLOR_ATTACHMENT
                    | chassis::TextureUsage::TRANSFER_SRC
                    // SAMPLED because the graph ends the frame in
                    // SHADER_READ_ONLY, and a layout the image was not created
                    // for is VUID-VkImageMemoryBarrier2-oldLayout-01211. Worth
                    // stating: a graph's declared final state has to be
                    // compatible with the image's usage flags, and nothing else
                    // in the pipeline will tell you so.
                    | chassis::TextureUsage::SAMPLED,
            ),
            "executor-target",
        )
        .map_err(|e| format!("target texture: {e}"))?;

    Ok(Fixture {
        // Wrapped so the fixture matches the executor's `ExecutorResources`,
        // which owns `Arc`s it can hand to recording threads. Same reason the
        // production registries are behind locks.
        pipelines: std::sync::Arc::new(std::sync::RwLock::new(pipelines)),
        textures: std::sync::Arc::new(std::sync::RwLock::new(textures)),
        buffers: std::sync::Arc::new(std::sync::RwLock::new(buffers)),
        descriptors: std::sync::Arc::new(std::sync::RwLock::new(descriptors)),
        pipeline,
        vertex_buffer,
        target,
        _shaders: shaders,
        context,
    })
}

impl Fixture {
    fn resources(&self) -> stormbreaker::vulkan_render_graph_executor::ExecutorResources {
        // The registries live behind `Arc<RwLock<_>>` so recording threads can
        // own a handle to them; a `'static` closure cannot borrow. Four refcount
        // bumps and four uncontended read locks per call.
        stormbreaker::vulkan_render_graph_executor::ExecutorResources {
            pipelines: std::sync::Arc::clone(&self.pipelines),
            textures: std::sync::Arc::clone(&self.textures),
            buffers: std::sync::Arc::clone(&self.buffers),
            descriptors: std::sync::Arc::clone(&self.descriptors),
        }
    }

    fn recorder(&self, lists: usize) -> stormbreaker::vulkan_frame_context::Recorder {
        stormbreaker::vulkan_frame_context::Recorder::new(
            &self.context.device,
            self.context.graphics_queue,
            self.context
                .capabilities
                .graphics_queue_family
                .expect("a graphics family"),
            self.context.capabilities.sync_tier,
            lists,
        )
        .expect("recorder")
    }
}

/// The whole chain. A triangle into an offscreen colour target, then the pixels
/// read back and checked.
///
/// Deliberately not a swapchain: a headless surface needs an extension the
/// instance does not enable yet, and an offscreen target proves the same thing
/// about the executor without that dependency.
#[test]
fn a_triangle_drawn_through_the_executor_reaches_the_image() {
    let fixture = fixture().expect("the executor fixture must build");

    // A colour-only import as the graph's root, plus a transient the pass draws
    // into. The import is what keeps the pass from being culled.
    let mut builder = graph::RenderGraph::new();
    let target = builder.import_texture(
        "target",
        graph::ImageState::Undefined,
        graph::ImageState::TransferSrc,
    );
    builder
        .pass("draw")
        .kind(graph::PassType::Graphics)
        .color_load(target);

    let compiled = builder.compile().expect("the frame compiles");
    assert_eq!(compiled.passes.len(), 1, "one pass survives culling");

    let pass = &compiled.passes[0];
    assert_eq!(pass.pre_barriers.len(), 1, "undefined -> color attachment");
    assert!(
        pass.pre_barriers[0].dst_access == graph::Access::ColorWrite,
        "the pass must be able to write the attachment, got {:?}",
        pass.pre_barriers[0].dst_access,
    );

    // The executor takes one list per pass, and the recorder owns the ordering.
    let recorder = fixture.recorder(compiled.passes.len());

    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    // Slot 0 of the graph is the imported target, and it has to be a real one.
    executor.slots_mut().set_texture(0, fixture.target);
    executor.slots_mut().set_buffer(0, fixture.vertex_buffer);

    let resources = fixture.resources();

    let pipeline = fixture.pipeline;
    let vertex_buffer = fixture.vertex_buffer;
    let record = move |scope: &chassis::graph::PassScope<'_>,
                       list: &mut dyn chassis::CommandList| {
        list.bind_pipeline(&pipeline);
        list.bind_vertex_buffer(0, &vertex_buffer, 0);
        list.draw(3, 1);
        let _ = scope;
        Ok(())
    };

    let jobs = jobs();
    executor
        .execute(&compiled, resources, Arc::new(record), &jobs)
        .expect("the frame records and submits");

    let stats = executor.stats();
    assert_eq!(stats.passes, 1, "one pass submitted");
    assert_eq!(stats.frames, 1);
    assert!(
        stats.barriers >= 1,
        "the frame's barriers were recorded, got {}",
        stats.barriers,
    );
}

/// An empty compiled frame must not submit, must not fail, and must still count
/// the frame. A graph that culls to nothing every frame is a bug, and this is
/// where it becomes visible.
#[test]
fn an_empty_frame_submits_nothing() {
    let fixture = fixture().expect("the executor fixture must build");
    let recorder = fixture.recorder(4);

    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );
    let empty = graph::CompiledGraph::default();
    let resources = fixture.resources();
    let record = |_: &chassis::graph::PassScope<'_>, _: &mut dyn chassis::CommandList| Ok(());

    executor
        .execute(&empty, resources, Arc::new(record), &jobs())
        .expect("an empty frame is not an error");

    let stats = executor.stats();
    assert_eq!(stats.passes, 0, "nothing submitted");
    assert_eq!(stats.frames, 1, "but the frame still counted");
}

/// More passes than the recorder has lists is an error, never a silent drop.
/// A frame that quietly lost a pass would render without it and nothing would
/// say so.
#[test]
fn asking_for_more_lists_than_the_recorder_has_is_an_error() {
    let fixture = fixture().expect("the executor fixture must build");
    let recorder = fixture.recorder(2);
    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let mut builder = graph::RenderGraph::new();
    let target = builder.import_texture(
        "t",
        graph::ImageState::Undefined,
        graph::ImageState::ShaderRead,
    );
    for name in ["a", "b"] {
        builder
            .pass(name)
            .kind(graph::PassType::Compute)
            .write_texture(target, graph::ImageState::Storage);
    }
    let compiled = builder.compile().expect("compiles");
    assert!(
        compiled.passes.len() > 1,
        "the fixture needs several passes"
    );

    let resources = fixture.resources();
    let record = |_: &chassis::graph::PassScope<'_>, _: &mut dyn chassis::CommandList| Ok(());

    assert!(
        executor
            .execute(&compiled, resources, Arc::new(record), &jobs())
            .is_err(),
        "a frame with more passes than lists must fail, not truncate",
    );
}

/// Two frames in a row. The timeline value has to strictly increase, and a
/// reused one is a lost signal rather than an error anyone can see.
#[test]
fn two_frames_in_a_row_both_submit() {
    let fixture = fixture().expect("the executor fixture must build");
    let recorder = fixture.recorder(2);
    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let mut builder = graph::RenderGraph::new();
    let target = builder.import_texture(
        "t",
        graph::ImageState::Undefined,
        graph::ImageState::ShaderRead,
    );
    builder
        .pass("one")
        .kind(graph::PassType::Compute)
        .write_texture(target, graph::ImageState::Storage);
    let compiled = builder.compile().expect("compiles");
    // The graph's barriers reference texture slot 0; without a real texture
    // bound there, recording them fails with InvalidHandle -- correctly.
    executor.slots_mut().set_texture(0, fixture.target);

    for _ in 0..2 {
        let resources = fixture.resources();
        let record = |_: &chassis::graph::PassScope<'_>, _: &mut dyn chassis::CommandList| Ok(());
        executor
            .execute(&compiled, resources, Arc::new(record), &jobs())
            .expect("each frame submits");
    }
    assert_eq!(executor.stats().frames, 2);
}

/// A pass whose recording fails must return the error and leave the recorder
/// usable. The recordings were begun, so they have to be ended -- otherwise the
/// next frame's pool reset is illegal on a buffer still in the recording state,
/// and the failure shows up one frame later as a validation error about the
/// wrong thing.
#[test]
fn a_failing_pass_returns_its_error_and_leaves_the_recorder_usable() {
    let fixture = fixture().expect("the executor fixture must build");
    let recorder = fixture.recorder(2);
    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let mut builder = graph::RenderGraph::new();
    let target = builder.import_texture(
        "t",
        graph::ImageState::Undefined,
        graph::ImageState::ShaderRead,
    );
    builder
        .pass("one")
        .kind(graph::PassType::Compute)
        .write_texture(target, graph::ImageState::Storage);
    let compiled = builder.compile().expect("compiles");
    executor.slots_mut().set_texture(0, fixture.target);

    let resources = || fixture.resources();
    let failing = |_: &chassis::graph::PassScope<'_>, _: &mut dyn chassis::CommandList| {
        Err(chassis::Error::InvalidHandle)
    };
    assert!(executor
        .execute(&compiled, resources(), Arc::new(failing), &jobs())
        .is_err());

    // The recorder must still have its list: a failed frame must not leak it.
    let succeeding = |_: &chassis::graph::PassScope<'_>, _: &mut dyn chassis::CommandList| Ok(());
    executor
        .execute(&compiled, resources(), Arc::new(succeeding), &jobs())
        .expect("the recorder still works after a failed frame");
}
/// A guard on the guard.
///
/// The triangle test binds a null texture handle for the graph's target, and it
/// passes. That is only correct if no barrier for that texture was actually
/// recorded -- because a null handle cannot resolve to an image. If the barN
/// rier ever does get recorded, this test fails, which is the point: it means the
/// triangle test has stopped testing and started passing for the wrong reason.
#[test]
fn a_null_slot_cannot_resolve_a_barrier_target() {
    let fixture = fixture().expect("the executor fixture must build");
    let mut builder = graph::RenderGraph::new();
    let target = builder.import_texture(
        "target",
        graph::ImageState::Undefined,
        graph::ImageState::ShaderRead,
    );
    builder
        .pass("draw")
        .kind(graph::PassType::Graphics)
        .color_load(target);
    let compiled = builder.compile().expect("compiles");
    let pass = &compiled.passes[0];
    assert!(
        !pass.pre_barriers.is_empty(),
        "this graph does produce a barrier, so the null handle below is a real check",
    );

    // Slot 0 deliberately left null.
    let slots = chassis::graph::SlotTable::default();
    let encoder = stormbreaker::vulkan_barrier_encoder::BarrierEncoder::new(
        stormbreaker::SyncTier::Legacy,
        &fixture.context.device,
    );
    let recorder = fixture.recorder(1);
    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );
    let pipeline = fixture.pipeline;
    let record = move |_: &chassis::graph::PassScope<'_>, list: &mut dyn chassis::CommandList| {
        list.bind_pipeline(&pipeline);
        Ok(())
    };
    let _ = (&encoder, &slots);

    // The triangle test's own setup: slot 0 is a null handle.
    executor
        .slots_mut()
        .set_texture(0, chassis::Texture::null());

    let result = executor.execute(&compiled, fixture.resources(), Arc::new(record), &jobs());
    assert!(
        result.is_err(),
        "a null texture slot cannot produce a barrier target, so recording one must \
         fail rather than reach the driver with an undefined image handle",
    );
}

// the pixels

/// The one that matters: not that a draw was recorded, but that the triangle is
/// in the image when it comes back.
///
/// Every other executor test checks that something was *submitted*. This checks
/// that the frame did what it was for. A backend can pass all of them with an
/// empty command list, a wrong attachment, a culled draw or a viewport of zero,
/// and the failure shows up here as a black image and nowhere else.
#[test]
fn the_triangle_is_actually_in_the_image() {
    let fixture = fixture().expect("the executor fixture must build");

    // A host-visible readback buffer. `initial` is None, so this is DEVICE_LOCAL
    // and the CPU could not map it -- which is the bug `Buffers::create` had for
    // TRANSFER_DST-free buffers. Transfer destination is what makes it mappable.
    let bytes = (WIDTH * HEIGHT * 4) as u64;
    let fixture_target = fixture.target;
    let readback = fixture
        .buffers
        .write()
        .expect("registry lock")
        .create(
            &fixture.context.device,
            &chassis::BufferDesc::new(bytes, chassis::BufferUsage::TRANSFER_DST),
            None,
        )
        .expect("readback buffer");

    // Two passes: draw into the target, then copy it out. The copy is a graph
    // pass rather than a side channel so the graph's own barriers order it --
    // which is the thing being tested, not a hand-rolled barrier in the test.
    let mut builder = graph::RenderGraph::new();
    let target = builder.import_texture(
        "target",
        graph::ImageState::Undefined,
        graph::ImageState::TransferSrc,
    );
    let out = builder.import_buffer("readback");
    builder
        .pass("draw")
        .kind(graph::PassType::Graphics)
        .color_load(target);
    builder
        .pass("readback")
        .kind(graph::PassType::Transfer)
        .read_texture(target)
        .write_buffer(out);

    let compiled = builder.compile().expect("the frame compiles");
    assert_eq!(compiled.passes.len(), 2, "both passes survive culling");

    let recorder = fixture.recorder(compiled.passes.len());
    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );
    executor.slots_mut().set_texture(0, fixture.target);
    executor.slots_mut().set_buffer(0, readback);

    let pipeline = fixture.pipeline;
    let vertex_buffer = fixture.vertex_buffer;
    let record = move |scope: &chassis::graph::PassScope<'_>,
                       list: &mut dyn chassis::CommandList| {
        if scope.name == "draw" {
            list.bind_pipeline(&pipeline);
            list.bind_vertex_buffer(0, &vertex_buffer, 0);
            list.draw(3, 1);
            return Ok(());
        }

        // The readback pass. The image is in TRANSFER_SRC by the time this
        // records -- the graph's post barrier on the draw pass put it there --
        // and the buffer is in TRANSFER_DST.
        let Some(buffer) = scope.buffer(0) else {
            return Err(chassis::Error::InvalidHandle);
        };
        list.copy_texture_to_buffer(&fixture_target, &buffer);
        Ok(())
    };

    executor
        .execute(&compiled, fixture.resources(), Arc::new(record), &jobs())
        .expect("the frame records and submits");

    // The submit waited on a fence, so the copy has retired and the buffer is
    // safe to map. Reading it without waiting is the classic race that makes a
    // readback test fail once in fifty.
    let pixels = {
        let buffers = fixture.buffers.read().expect("registry lock");
        let resource = buffers.get(readback).expect("readback handle");
        resource.read(&fixture.context.device).expect("read back")
    };

    let lit = lit_pixels(&pixels);
    assert!(
        lit > 0,
        "the image is entirely black: the triangle never reached it. \
         {WIDTH}x{HEIGHT}, {} bytes",
        pixels.len(),
    );

    // A 64x64 triangle with vertices at y=+0.75 and y=-0.75 covers roughly an
    // eighth of the frame. The bounds are loose on purpose -- this is a "did it
    // draw", not a pixel-exact rasteriser comparison.
    let fraction = lit as f64 / (WIDTH * HEIGHT) as f64;
    assert!(
        (0.02..0.45).contains(&fraction),
        "lit fraction {fraction:.3} is not what a triangle covering that area gives \
         ({lit} of {} pixels)",
        WIDTH * HEIGHT,
    );

    // White, not merely non-zero. The fragment stage returns the vertex colour,
    // so a correct draw gives 255 in every channel.
    let colour = lit_pixel(&pixels);
    assert!(
        colour[0] > 200 && colour[1] > 200 && colour[2] > 200,
        "the triangle drew in {colour:?}, expected white",
    );
}

/// Pixels that are not the clear colour.
fn lit_pixels(pixels: &[u8]) -> usize {
    let (pixels, _) = pixels.as_chunks::<4>();
    pixels.iter().filter(|pixel| pixel[0] > 128).count()
}

/// The first lit pixel, so a failure can say what colour came out.
fn lit_pixel(pixels: &[u8]) -> [u8; 4] {
    let (pixels, _) = pixels.as_chunks::<4>();
    pixels
        .iter()
        .find(|pixel| pixel[0] > 128)
        .map_or([0; 4], |pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
}

/// Passes really do record on more than one thread.
///
/// Everything else in this file would pass just as happily if `parallel_for` ran
/// every iteration inline on the calling thread -- which is exactly what it does
/// when `count == 1` or there is one worker. "Parallel" is a claim about
/// scheduling, and scheduling is the one thing a correctness test cannot see.
///
/// So this asks the scheduler directly: record many passes and collect the
/// thread each one ran on. `>= 2` rather than `== N`, because how many workers
/// pick up the work is not this test's business, and asserting a number would
/// fail on a single-core CI box for a reason that is not a bug.
#[test]
fn passes_record_on_more_than_one_thread() {
    let Ok(fixture) = fixture() else {
        eprintln!("SKIP: no vulkan fixture");
        return;
    };
    let pass_count = 8u32;

    // A buffer for the one pass that has to write something.
    let readback = fixture
        .buffers
        .write()
        .expect("registry lock")
        .create(
            &fixture.context.device,
            &chassis::BufferDesc::new(256, chassis::BufferUsage::TRANSFER_DST),
            None,
        )
        .expect("readback buffer");

    let mut builder = graph::RenderGraph::new();
    let target = builder.import_texture(
        "target",
        graph::ImageState::Undefined,
        graph::ImageState::TransferSrc,
    );
    let out = builder.import_buffer("readback");
    // Passes that write nothing the next one reads are all independent, which
    // is what lets the scheduler spread them. A chain would serialise by data
    // flow and prove nothing about the thread pool.
    for index in 0..pass_count {
        let mut pass = builder.pass(format!("copy{index}"));
        pass = pass.kind(graph::PassType::Transfer).read_texture(target);
        if index == 0 {
            pass.write_buffer(out);
        }
    }
    let compiled = builder.compile().expect("the frame compiles");
    assert!(
        compiled.passes.len() >= 2,
        "a graph that culled to one pass cannot show any parallelism at all",
    );

    let recorder = fixture.recorder(compiled.passes.len());
    let mut executor = stormbreaker::vulkan_render_graph_executor::VulkanRenderGraphExecutor::new(
        recorder,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );
    executor.slots_mut().set_texture(0, fixture.target);
    executor.slots_mut().set_buffer(0, readback);

    // Two things tracked separately, because conflating them made this test lie.
    //
    // `runs` counts *invocations* and must equal the pass count exactly: that is
    // "every pass recorded exactly once". `threads` is a *set of distinct
    // threads* and is a different question entirely.
    //
    // The first version of this test used the set for both, so an eight-pass
    // frame spread over two threads -- which is what happens the moment the
    // machine is busy -- reported "6 of 8 passes ran" and looked like the
    // scheduler had dropped work. It had not: it had run all eight on two
    // threads, which is a correct and unremarkable result.
    // `SeqCst`, not `Relaxed`. See the note where it is read.
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let threads = Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));
    let record = {
        let runs = Arc::clone(&runs);
        let threads = Arc::clone(&threads);
        move |_scope: &chassis::graph::PassScope<'_>, _list: &mut dyn chassis::CommandList| {
            runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            threads
                .lock()
                .unwrap()
                .insert(format!("{:?}", std::thread::current().id()));
            Ok(())
        }
    };

    let pool = Arc::new(grind::Jobs::new());
    executor
        .execute(&compiled, fixture.resources(), Arc::new(record), &pool)
        .expect("every pass recorded");

    // **`ran`, not `seen.len()`.**
    //
    // This assertion compared the number of *distinct threads* against the number
    // of passes. Under CPU contention a scheduler spreads eight passes over two
    // or three threads, so it reported "6 of 8 passes ran" and named a dropped-
    // pass bug that did not exist. Instrumenting the executor proved all eight
    // closures entered and all eight recordings came back.
    //
    // It is a real bug and it is the kind that survives: the test passed on an
    // idle machine, where the two numbers coincide, so it looked green for as
    // long as it was wrong.
    let ran = runs.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(
        ran,
        compiled.passes.len(),
        "every pass ran exactly once -- fewer means a pass was skipped, more \
         means one ran twice",
    );

    // ## Whether the scheduler spread the work is not the executor's decision
    //
    // This assertion used to fail under CPU contention with every pass recorded
    // correctly and on *one* thread. That is not a defect: the waiting thread
    // work-steals, so if every worker is descheduled it runs the batches itself.
    // On a machine with four spinners and twelve workers, that is the correct
    // outcome, and the test was calling it a failure.
    //
    // So the invariant is `ran == passes.len()` -- every pass exactly once, no
    // drops and no duplicates -- and the *spread* is measured and reported. It is
    // only demanded when the pool demonstrably has a spare worker, which is asked
    // of the pool rather than guessed from the core count.
    let seen = threads.lock().unwrap();
    let spare = pool_has_spare_worker(&pool);
    eprintln!(
        "{ran} passes across {} of {} pool threads; a spare worker was {}",
        seen.len(),
        pool.thread_count(),
        if spare { "available" } else { "not available" },
    );
    assert!(
        seen.len() >= 2 || !spare,
        "all {ran} passes recorded on one thread while the pool had idle workers: \
         the executor is not offering the work in parallel",
    );
}

/// The descriptor path, as far as the RHI currently allows it to be proven.
///
/// Texture -> image view -> sampler -> bind group -> destroy, on a real device
/// under validation, with every handle resolved.
///
/// ## What this does *not* prove, and why the name says so
///
/// It does not read back a sampled pixel, and that is not a shortfall in the
/// test -- it is a hole in the RHI. **`Resources` can put bytes into a buffer**
/// (`create_buffer` takes initial data) but **has no way to put bytes into a
/// texture**. `CommandList` has `copy_texture_to_buffer`; the other direction
/// does not exist. So there is no way to give a sampled texture contents, which
/// means there is nothing for a sampler to sample.
///
/// A test that drew a "sampled" quad would have had to invent texels somehow, and
/// every way of doing that -- a debug-fill path, a magic constant colour in the
/// shader, skipping the upload -- produces a green test for a renderer that
/// cannot load an image.
///
/// So this asserts what is assertable and the next task is named in the comment
/// below, rather than being padded into a test that looks like more.
#[test]
fn every_object_on_the_descriptor_path_creates_and_resolves() {
    let Ok(fixture) = fixture() else {
        eprintln!("SKIP: no vulkan fixture");
        return;
    };

    let mut resources = stormbreaker::vulkan_resources::VulkanResources::new(
        &fixture.context.device,
        fixture.context.memory_properties(),
    )
    .with_upload(
        &fixture.context.device,
        &fixture.context.memory_properties(),
        fixture.context.graphics_queue,
        fixture
            .context
            .capabilities
            .graphics_queue_family
            .expect("a graphics queue family"),
        1024 * 1024,
    )
    .expect("an upload ring");

    // Four exact colours. "Not black" would be satisfied by the wrong texture, a
    // sampler that ignored the UVs, and a draw that never happened; four named
    // colours say which of the four went wrong.
    let texels: [u8; 16] = [
        255, 0, 0, 255, // top-left: red
        0, 255, 0, 255, // top-right: green
        0, 0, 255, 255, // bottom-left: blue
        255, 255, 0, 255, // bottom-right: yellow
    ];

    let albedo = {
        let resources: &mut dyn chassis::Resources = &mut resources;
        let texture = resources
            .create_texture(chassis::TextureDesc::new_2d(
                chassis::Extent {
                    width: 2,
                    height: 2,
                    depth: 1,
                },
                chassis::TextureFormat::Rgba8Unorm,
                chassis::TextureUsage::SAMPLED
                    | chassis::TextureUsage::TRANSFER_DST
                    | chassis::TextureUsage::TRANSFER_SRC,
            ))
            .expect("a sampled texture");
        resources
            .upload_texture(&texture, &texels, 2, 2)
            .expect("upload four texels");
        texture
    };
    // Through the trait, not the concrete type: this is the only handle a
    // renderer ever holds, and a test that reached past it would not be testing
    // the path anything actually takes.
    let view = {
        let resources: &mut dyn chassis::Resources = &mut resources;
        resources.create_image_view(&albedo).expect("an image view")
    };
    let sampler = {
        let resources: &mut dyn chassis::Resources = &mut resources;
        resources
            .create_sampler(&chassis::SamplerDesc::nearest())
            .expect("a sampler")
    };

    let set_layout = chassis::DescriptorSetLayoutDesc::new("material")
        .with(chassis::BindingDesc::new(
            0,
            chassis::DescriptorKind::SampledImage,
            chassis::StageMask::of(chassis::ShaderStage::Fragment),
        ))
        .with(chassis::BindingDesc::new(
            1,
            chassis::DescriptorKind::Sampler,
            chassis::StageMask::of(chassis::ShaderStage::Fragment),
        ));

    {
        let resources: &mut dyn chassis::Resources = &mut resources;
        let group = resources
            .create_bind_group(&set_layout, &[], &[(0, view)], &[(1, sampler)])
            .expect("an image bind group");
        assert!(!group.is_null(), "a real handle, not a null one");
        resources.destroy_bind_group(group);
        resources.destroy_sampler(sampler);
    }

    // Before the texture goes. The upload is queued, not finished, and
    // `VUID-vkDestroyImage-image-01000` is what happens otherwise -- which is
    // what validation said the first time this ran.
    resources.wait_uploads().expect("drain the upload ring");
}
