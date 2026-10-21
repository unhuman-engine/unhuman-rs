//! A repeatable micro-benchmark for the Vulkan per-frame CPU path.
//!
//! `#[ignore]` so the correctness suites pay nothing for it. Run it by hand:
//!
//! ```text
//! UHE_VALIDATION=1 cargo test -p stormbreaker --test bench_recording \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! It prints three numbers, because "recording is slow" is three different
//! questions:
//!
//!  * `record-only` -- `take` + barrier lowering + scope + pass bodies + `end`,
//!    with no submit and no fence. This is the CPU recording cost proper.
//!  * `record_frame` -- the whole submission through the executor, including the
//!    fence wait, which is what the renderer times.
//!  * `submit+wait` -- the queue submit plus the fence wait on its own, measured
//!    on a second device so its timeline does not collide with the executor's.
//!
//! The frame is a stand-in for `bifrost`'s: shadow, opaque, SSAO, editor,
//! composite and readback, at the extent the issue measured. The draw load is a
//! few hundred `draw(3, 1)` calls recorded per frame through a real pipeline, so
//! the per-call costs a scene actually pays are on the clock.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chassis::graph::pass::{PassRecorder, PassScope};
use chassis::graph::{CompiledGraph, ImageState, PassType, SlotTable};
use chassis::{CommandList as _, Gpu, Resources as _};
use stormbreaker::vulkan_command_buffer::{CommandList as VulkanCommandList, Recording};
use stormbreaker::vulkan_device::{create_device, DeviceOptions};
use stormbreaker::vulkan_frame_context::Recorder;
use stormbreaker::vulkan_gpu::VulkanGpu;
use stormbreaker::vulkan_render_graph_barriers::{
    open_render_scope, record_post_barriers, record_pre_barriers,
};
use stormbreaker::vulkan_submission_ring::{SubmissionToken, FRAMES_IN_FLIGHT};
use stormbreaker::BarrierEncoder;

const WIDTH: u32 = 1998;
const HEIGHT: u32 = 1135;
const SHADOW: u32 = 1024;
const FRAMES: u32 = 60;
/// Draws recorded per colour pass. A scene of this size is a few hundred objects.
/// `BENCH_DRAWS` overrides it, for sweeping the per-draw cost.
fn draws() -> u32 {
    std::env::var("BENCH_DRAWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(300)
}

/// Three vertices inside clip space, matching the 32-byte vertex layout.
const VERTICES: [f32; 24] = [
    0.0, 0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // top
    -0.75, -0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // bottom left
    0.75, -0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // bottom right
];

fn extent() -> chassis::Extent2d {
    chassis::Extent2d {
        width: WIDTH,
        height: HEIGHT,
    }
}

/// A workspace path, found by walking up from this crate. `cargo test` runs from
/// the crate, not the workspace root, so a fixed `../../assets` resolves to
/// nothing and the shader silently goes missing.
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

fn open_gpu() -> Option<VulkanGpu> {
    if !stormbreaker::device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    let context = match create_device(DeviceOptions::from_env(c"stormbreaker-bench")) {
        Ok(context) => context,
        Err(error) => {
            eprintln!("SKIP: no vulkan device: {error}");
            return None;
        }
    };
    match VulkanGpu::headless(context, extent()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("SKIP: no gpu: {error}");
            None
        }
    }
}

/// The shape of `bifrost::render`'s frame, without the shader/pipeline work.
fn frame() -> (CompiledGraph, SlotTable) {
    use chassis::graph::RenderGraph;
    let mut graph = RenderGraph::new();
    let scene_color =
        graph.import_texture("scene.color", ImageState::Undefined, ImageState::ShaderRead);
    let scene_depth = graph.import_texture(
        "scene.depth",
        ImageState::Undefined,
        ImageState::DepthAttachment,
    );
    let shadow = graph.import_texture("shadow.map", ImageState::Undefined, ImageState::ShaderRead);
    let ssao = graph.import_texture("ssao.ao", ImageState::Undefined, ImageState::ShaderRead);
    let backbuffer = graph.import_texture(
        "backbuffer",
        ImageState::Undefined,
        ImageState::ColorAttachment,
    );

    graph
        .pass("shadow")
        .kind(PassType::Graphics)
        .depth(shadow, 1.0);
    graph
        .pass("opaque")
        .kind(PassType::Graphics)
        .color(scene_color, [0.0, 0.0, 0.0, 1.0])
        .depth(scene_depth, 1.0)
        .read_texture(shadow);
    graph
        .pass("ssao")
        .kind(PassType::Graphics)
        .color(ssao, [1.0; 4])
        .read_texture(scene_depth);
    graph
        .pass("editor")
        .kind(PassType::Graphics)
        .color_load(scene_color)
        .depth_load(scene_depth);
    graph
        .pass("composite")
        .kind(PassType::Graphics)
        .color(backbuffer, [0.0; 4])
        .read_texture(scene_color)
        .read_texture(ssao);
    let out = graph.import_buffer("readback");
    graph
        .pass("readback")
        .kind(PassType::Transfer)
        .read_texture(backbuffer)
        .write_buffer(out);

    let compiled = graph.compile().expect("the bench frame compiles");
    let mut slots = SlotTable::default();
    slots.set_texture(scene_color, chassis::Texture::null());
    (compiled, slots)
}

/// Real textures for every slot the frame names, at the bench extent.
fn bind_textures(gpu: &mut VulkanGpu, slots: &mut SlotTable) {
    use chassis::{Extent, TextureDesc, TextureFormat, TextureUsage};
    let desc = |width, height, format, usage| {
        TextureDesc::new_2d(
            Extent {
                width,
                height,
                depth: 1,
            },
            format,
            usage,
        )
    };

    let color = TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED | TextureUsage::TRANSFER_SRC;
    let depth = TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED;
    let buffers = [
        (WIDTH, HEIGHT, TextureFormat::Rgba8Unorm, color),
        (WIDTH, HEIGHT, TextureFormat::Depth32Float, depth),
        (SHADOW, SHADOW, TextureFormat::Depth32Float, depth),
        (WIDTH, HEIGHT, TextureFormat::Rgba8Unorm, color),
        (WIDTH, HEIGHT, TextureFormat::Rgba8Unorm, color),
    ];
    let resources = gpu.resources();
    for (index, (width, height, format, usage)) in buffers.into_iter().enumerate() {
        let texture = resources
            .create_texture(desc(width, height, format, usage))
            .expect("a bench texture");
        slots.set_texture(index as u32, texture);
    }
}

/// A pipeline and vertex buffer to draw with, so the recording loop pays the
/// per-draw costs a scene actually pays. Built through the RHI, the renderer's
/// own path; the shaders are the precompiled triangle, so the bench does not
/// need `slangc`.
fn drawing_kit(gpu: &mut VulkanGpu) -> (chassis::Pipeline, chassis::Buffer) {
    let vertex_spirv = std::fs::read(asset("assets/shaders/Triangle.vert.spv")).expect("vs spv");
    let fragment_spirv = std::fs::read(asset("assets/shaders/Triangle.frag.spv")).expect("fs spv");
    let resources = gpu.resources();
    let vertex_shader = resources
        .create_shader("bench.vert", &vertex_spirv)
        .expect("vertex shader");
    let fragment_shader = resources
        .create_shader("bench.frag", &fragment_spirv)
        .expect("fragment shader");

    let description = chassis::PipelineDesc {
        label: "bench.triangle".to_owned(),
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
        // The winding convention is a trap: Vulkan's viewport maps NDC +Y to
        // framebuffer -Y, so a CCW triangle reads as CW and a default cull would
        // cull it. Off here, where the goal is load, not correctness.
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
    let pipeline = resources
        .create_pipeline(description)
        .expect("bench pipeline");

    let mut bytes = Vec::with_capacity(std::mem::size_of_val(&VERTICES));
    for value in VERTICES {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let vertex = resources
        .create_buffer(
            chassis::BufferDesc::new(bytes.len() as u64, chassis::BufferUsage::VERTEX),
            Some(&bytes),
        )
        .expect("vertex buffer");

    (pipeline, vertex)
}

/// The recorder body: bind the kit and issue `DRAWS` draws in the two colour
/// passes that have no depth attachment, so the pipeline and the pass agree.
fn drawing(pipeline: chassis::Pipeline, vertex: chassis::Buffer) -> impl PassRecorder {
    move |scope: &PassScope<'_>, list: &mut dyn chassis::CommandList| {
        if matches!(scope.name, "ssao" | "composite") {
            list.bind_pipeline(&pipeline);
            list.bind_vertex_buffer(0, &vertex, 0);
            for _ in 0..draws() {
                list.draw(3, 1);
            }
        }
        Ok(())
    }
}

/// Record one frame's lists and hand them back ended. No submit, so no fence.
#[allow(clippy::too_many_arguments)]
fn record_once(
    recorder: &mut Recorder,
    encoder: &BarrierEncoder,
    compiled: &CompiledGraph,
    slots: &SlotTable,
    resources: &stormbreaker::vulkan_render_graph_executor::ExecutorResources,
    record: &dyn PassRecorder,
) -> Vec<Recording> {
    let pipelines = resources.pipelines.read().expect("registry lock");
    let textures = resources.textures.read().expect("registry lock");
    let buffers = resources.buffers.read().expect("registry lock");
    let descriptors = resources.descriptors.read().expect("registry lock");

    let mut recordings = recorder.take(compiled.passes.len()).expect("lists");
    let mut ended = Vec::with_capacity(recordings.len());
    for (mut recording, pass) in recordings.drain(..).zip(compiled.passes.iter()) {
        {
            let mut list = VulkanCommandList::new(
                &mut recording,
                &pipelines,
                &textures,
                &buffers,
                &descriptors,
            );
            record_pre_barriers(encoder, pass, slots, &textures, &buffers, &mut list)
                .expect("pre barriers");
            let pass_extent = extent();
            let rendering =
                open_render_scope(pass, slots, &textures, &pass_extent, &mut list).expect("scope");
            let scope = PassScope {
                index: pass.index,
                name: &pass.name,
                extent: pass_extent,
                slots,
            };
            record.record(&scope, &mut list).expect("pass body");
            if rendering {
                list.end_rendering();
            }
            record_post_barriers(encoder, pass, slots, &textures, &mut list)
                .expect("post barriers");
        }
        ended.push(recording.end().expect("end"));
    }
    ended
}

/// Time the CPU recording loop with no submit: `take` through `end`, then the
/// buffers go back without ever reaching the queue.
fn measure_record_only(
    gpu: &mut VulkanGpu,
    compiled: &CompiledGraph,
    slots: &SlotTable,
    record: &dyn PassRecorder,
) -> Duration {
    let resources = gpu.resources().executor_resources();
    let executor = gpu.executor_mut();
    let recorder = executor.recorder_mut();
    let encoder = BarrierEncoder::new(recorder.tier(), &recorder.device().clone());

    let start = Instant::now();
    for _ in 0..FRAMES {
        let ended = record_once(recorder, &encoder, compiled, slots, &resources, record);
        recorder.recycle(ended);
    }
    start.elapsed() / FRAMES
}

/// Time the executor's whole `record_frame`: parallel recording, submit, fence.
fn measure_record_frame(
    gpu: &mut VulkanGpu,
    compiled: &CompiledGraph,
    slots: &SlotTable,
    record: Arc<dyn PassRecorder>,
) -> Duration {
    let start = Instant::now();
    for _ in 0..FRAMES {
        gpu.record_frame(compiled, slots, Arc::clone(&record))
            .expect("a bench frame records");
    }
    start.elapsed() / FRAMES
}

/// Record one frame and submit it, returning the fence token.
fn submit_once(
    recorder: &mut Recorder,
    encoder: &BarrierEncoder,
    compiled: &CompiledGraph,
    slots: &SlotTable,
    resources: &stormbreaker::vulkan_render_graph_executor::ExecutorResources,
    record: &dyn PassRecorder,
) -> SubmissionToken {
    let ended = record_once(recorder, encoder, compiled, slots, resources, record);
    recorder.submit_batched(ended).expect("submit")
}

/// Time the queue submit on its own, with recording outside the timed region.
///
/// The submit still waits when it reuses the oldest fence in the ring -- that
/// wait is inherent to a finite ring -- but it does not wait for the frame it
/// just queued. The readback column is the separate wait for the slot read
/// `READBACK_LAG` frames later.
fn measure_submit(
    recorder: &mut Recorder,
    encoder: &BarrierEncoder,
    compiled: &CompiledGraph,
    slots: &SlotTable,
    resources: &stormbreaker::vulkan_render_graph_executor::ExecutorResources,
    record: &dyn PassRecorder,
) -> (Duration, Duration) {
    const READBACK_LAG: usize = FRAMES_IN_FLIGHT - 1;
    let mut submit_total = Duration::ZERO;
    let mut readback_total = Duration::ZERO;
    let mut tokens: VecDeque<SubmissionToken> = VecDeque::with_capacity(FRAMES_IN_FLIGHT);

    for _ in 0..FRAMES {
        let start = Instant::now();
        let token = submit_once(recorder, encoder, compiled, slots, resources, record);
        submit_total += start.elapsed();

        tokens.push_back(token);
        if tokens.len() > READBACK_LAG {
            let old = tokens.pop_front().expect("the ring holds the slot");
            let start = Instant::now();
            recorder.wait(old).expect("readback wait");
            readback_total += start.elapsed();
        }
    }
    for token in tokens {
        recorder.wait(token).expect("drain the ring");
    }
    (submit_total / FRAMES, readback_total / FRAMES)
}

#[test]
#[ignore = "micro-benchmark; run with --ignored"]
fn profile_the_frame_path() {
    let Some(mut gpu) = open_gpu() else { return };
    let (compiled, mut slots) = frame();
    bind_textures(&mut gpu, &mut slots);
    let (pipeline, vertex) = drawing_kit(&mut gpu);
    assert_eq!(
        compiled.passes.len(),
        6,
        "the bench frame is the six-pass bifrost shape",
    );
    let record: Arc<dyn PassRecorder> = Arc::new(drawing(pipeline, vertex));

    for _ in 0..5 {
        gpu.record_frame(&compiled, &slots, Arc::clone(&record))
            .expect("warmup frame");
    }

    let record_only = measure_record_only(&mut gpu, &compiled, &slots, record.as_ref());
    let frame_time = measure_record_frame(&mut gpu, &compiled, &slots, Arc::clone(&record));

    let Some(mut submit_gpu) = open_gpu() else {
        return;
    };
    let mut submit_slots = SlotTable::default();
    bind_textures(&mut submit_gpu, &mut submit_slots);
    let (submit_pipeline, submit_vertex) = drawing_kit(&mut submit_gpu);
    let submit_record: Arc<dyn PassRecorder> = Arc::new(drawing(submit_pipeline, submit_vertex));
    let resources = submit_gpu.resources().executor_resources();
    let executor = submit_gpu.executor_mut();
    let recorder = executor.recorder_mut();
    let device = recorder.device().clone();
    let encoder = BarrierEncoder::new(recorder.tier(), &device);
    let (submit, readback) = measure_submit(
        recorder,
        &encoder,
        &compiled,
        &submit_slots,
        &resources,
        submit_record.as_ref(),
    );

    eprintln!(
        "bench {WIDTH}x{HEIGHT}, {} passes, {} draws/pass, {FRAMES} frames:\n  \
         record-only    {:>8.3} ms\n  submit (no wait){:>8.3} ms (no recording)\n  \
         readback wait  {:>8.3} ms\n  record_frame   {:>8.3} ms",
        compiled.passes.len(),
        draws(),
        record_only.as_secs_f64() * 1e3,
        submit.as_secs_f64() * 1e3,
        readback.as_secs_f64() * 1e3,
        frame_time.as_secs_f64() * 1e3,
    );
}

/// A pending fence is never reset for a new submission.
///
/// The slow pass leaves the first submission on the GPU while the next two use
/// the other slots. The fourth must wait that first fence before reusing it, and
/// its generation must move on.
#[test]
fn a_pending_fence_is_waited_before_reuse() {
    let Some(mut gpu) = open_gpu() else { return };
    let (compiled, mut slots) = frame();
    bind_textures(&mut gpu, &mut slots);
    let (pipeline, vertex) = drawing_kit(&mut gpu);

    // Long enough that the first frame is still queued when the ring wraps,
    // short enough that the test is not a benchmark itself.
    const SLOW_DRAWS: u32 = 50_000;
    let slow: Arc<dyn PassRecorder> = Arc::new(
        move |scope: &chassis::graph::PassScope<'_>, list: &mut dyn chassis::CommandList| {
            if scope.name == "ssao" {
                list.bind_pipeline(&pipeline);
                list.bind_vertex_buffer(0, &vertex, 0);
                for _ in 0..SLOW_DRAWS {
                    list.draw(3, 1);
                }
            }
            Ok(())
        },
    );
    let fast: Arc<dyn PassRecorder> =
        Arc::new(|_: &chassis::graph::PassScope<'_>, _: &mut dyn chassis::CommandList| Ok(()));

    let resources = gpu.resources().executor_resources();
    let executor = gpu.executor_mut();
    let recorder = executor.recorder_mut();
    let device = recorder.device().clone();
    let encoder = BarrierEncoder::new(recorder.tier(), &device);

    let first = submit_once(
        recorder,
        &encoder,
        &compiled,
        &slots,
        &resources,
        slow.as_ref(),
    );
    assert!(
        recorder.is_pending(first),
        "the slow frame finished before the ring could wrap"
    );

    let mut fresh = Vec::new();
    for _ in 0..(FRAMES_IN_FLIGHT - 1) {
        let token = submit_once(
            recorder,
            &encoder,
            &compiled,
            &slots,
            &resources,
            fast.as_ref(),
        );
        assert_ne!(
            token.index(),
            first.index(),
            "a slot was reused before the ring walked every fence"
        );
        fresh.push(token);
    }

    let reused = submit_once(
        recorder,
        &encoder,
        &compiled,
        &slots,
        &resources,
        fast.as_ref(),
    );
    assert_eq!(
        reused.index(),
        first.index(),
        "the ring wraps to its oldest fence"
    );
    assert!(
        reused.generation() > first.generation(),
        "a reused fence must have a new generation"
    );
    assert!(
        !recorder.is_pending(first),
        "the pending fence was reused without waiting for it"
    );

    recorder.wait(reused).expect("drain the reused fence");
    for token in fresh {
        recorder.wait(token).expect("drain the ring");
    }
}
