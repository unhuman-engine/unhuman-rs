//! The wgpu executor, on a real compiled frame.
//!
//! The graph is `chassis::graph`, built once and used by both backends. That is
//! the claim being tested: one frame description, two very different executors.

use chassis::graph as rg;
use chassis::graph::{NoRecording, PassRecorder, PassScope, SlotTable};
use chassis::{CommandList, Resources};
use stormglass::wgpu_device::Adapter;
use stormglass::wgpu_render_graph_executor::{
    backbuffer_slot, resolve_backbuffer, ExecutorStats, WgpuRenderGraphExecutor as WgpuExecutor,
};
use stormglass::wgpu_resources::WgpuResources;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// A frame the executor can run, built with the shared graph.
///
/// Two declared passes, which compile to two. The frame's *final* transition --
/// the image to `TransferSrc` for readback -- is a post-barrier on the last
/// writer, not a pass of its own. That is the difference between a barrier and a
/// pass, and it is why the count here is 2 and not 3.
fn two_pass_frame() -> (rg::RenderGraph, rg::TextureId) {
    let mut graph = rg::RenderGraph::new();
    let target = graph.import_texture(
        "target",
        rg::ImageState::Undefined,
        rg::ImageState::TransferSrc,
    );
    graph
        .pass("draw")
        .kind(rg::PassType::Graphics)
        .color_load(target);
    graph
        .pass("resolve")
        .kind(rg::PassType::Graphics)
        .color_load(target);
    (graph, target)
}

/// Bind a real colour target to `slot`, so the executor can open a render scope
/// for a pass that writes it. The executor opens the scope from the compiled
/// pass's attachments, so an unbound one is refused before the recorder runs.
fn bind_target(executor: &mut WgpuExecutor, slot: rg::TextureId) {
    let texture = executor
        .resources_mut()
        .create_texture(chassis::TextureDesc::new_2d(
            chassis::Extent {
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
            },
            chassis::TextureFormat::Rgba8Unorm,
            chassis::TextureUsage::COLOR_ATTACHMENT,
        ))
        .expect("the target texture");
    executor.slots_mut().set_texture(slot, texture);
}

/// One compiled frame, submitted. No pixels yet -- the pipeline and the command
/// list are covered elsewhere; this is the executor's own job.
#[test]
fn a_compiled_frame_runs_through_the_executor() {
    let Some(adapter) = Adapter::request(0) else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-executor") else {
        return;
    };
    let resources = WgpuResources::new(&device);
    let mut executor = WgpuExecutor::new(
        queue,
        resources,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let (graph, target) = two_pass_frame();
    let compiled = graph.compile().expect("the frame compiles");
    assert_eq!(compiled.passes.len(), 2, "both passes survive culling");
    bind_target(&mut executor, target);

    executor
        .execute(&compiled, &NoRecording)
        .expect("a frame with no recordings still submits");
    executor.wait_idle().expect("the queue drains");

    let stats = executor.stats();
    assert_eq!(
        stats,
        ExecutorStats {
            passes: 2,
            buffers: 2,
            frames: 1,
        }
    );
}

/// Two frames, because a second frame is where ordering bugs show up.
///
/// Nothing carries over between frames in wgpu -- each submit is independent and
/// the queue orders them -- so this is cheap to assert and catches an executor
/// that reused something it should have rebuilt.
#[test]
fn two_frames_in_a_row_both_run() {
    let Some(adapter) = Adapter::request(0) else {
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-two-frames") else {
        return;
    };
    let resources = WgpuResources::new(&device);
    let mut executor = WgpuExecutor::new(
        queue,
        resources,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );
    let (graph, target) = two_pass_frame();
    let compiled = graph.compile().expect("compiles");
    bind_target(&mut executor, target);

    for _ in 0..2 {
        executor
            .execute(&compiled, &NoRecording)
            .expect("each frame runs");
    }
    executor.wait_idle().expect("drains");
    assert_eq!(executor.stats().frames, 2);
}

/// An empty frame is not an error and submits nothing.
///
/// A frame that culls to nothing is a real condition -- a window minimised, a
/// camera off screen -- and treating it as a failure is how a frame loop ends up
/// logging an error every time the user alt-tabs.
#[test]
fn an_empty_frame_submits_nothing() {
    let Some(adapter) = Adapter::request(0) else {
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-empty") else {
        return;
    };
    let resources = WgpuResources::new(&device);
    let mut executor = WgpuExecutor::new(
        queue,
        resources,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let graph = rg::RenderGraph::new();
    let compiled = graph.compile().expect("an empty graph compiles");
    executor
        .execute(&compiled, &NoRecording)
        .expect("not an error");
    executor.wait_idle().expect("drains");

    let stats = executor.stats();
    assert_eq!(stats.passes, 0);
    assert_eq!(stats.frames, 1, "the frame still counted");
}

/// The seam, stated: a renderer implements `PassRecorder` once and the executor
/// drives it.
///
/// This compiles only because `PassRecorder` and `PassScope` live in `chassis`,
/// not in either backend. Before the move this file could not have existed
/// without wgpu depending on stormbreaker.
#[test]
fn a_renderer_writes_one_pass_recorder_for_either_backend() {
    let Some(adapter) = Adapter::request(0) else {
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-seam") else {
        return;
    };
    let resources = WgpuResources::new(&device);
    let mut executor = WgpuExecutor::new(
        queue,
        resources,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let (graph, target) = two_pass_frame();
    let compiled = graph.compile().expect("compiles");
    bind_target(&mut executor, target);

    // A plain closure: no backend type, no backend import, nothing but chassis.
    //
    // Behind a `Mutex`, because `PassRecorder`'s blanket impl takes `Fn` and a
    // closure that accumulates state is `FnMut`. That is the shape every renderer
    // has to use and it is worth being blunt about: a plain `Vec` push would not
    // compile, so a renderer that wants to collect per-pass state reaches for
    // interior mutability immediately. A `Cell` would do for a single thread and
    // would silently stop the cross-thread recording the bound exists to allow.
    let visited = std::sync::Mutex::new(Vec::new());
    let record = |scope: &PassScope<'_>, _list: &mut dyn CommandList| {
        visited
            .lock()
            .expect("not poisoned")
            .push(scope.name.to_owned());
        assert_eq!(scope.extent.width, WIDTH);
        Ok(())
    };

    executor.execute(&compiled, &record).expect("runs");
    let visited = visited.into_inner().expect("not poisoned");
    assert_eq!(
        visited,
        vec!["draw", "resolve"],
        "every pass was visited, in compiled order",
    );
}

/// A pass that ends while still inside a render scope is refused, naming it.
///
/// Submitted as-is, wgpu ends the scope at `finish` and the tail of the pass is
/// quietly lost. That is the worst possible failure: a frame that renders, just
/// not all of it.
#[test]
fn a_pass_that_never_closes_its_render_scope_is_refused() {
    let Some(adapter) = Adapter::request(0) else {
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-unclosed") else {
        return;
    };
    let resources = WgpuResources::new(&device);
    let mut executor = WgpuExecutor::new(
        queue,
        resources,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let mut graph = rg::RenderGraph::new();
    let target = graph.import_texture(
        "target",
        rg::ImageState::Undefined,
        rg::ImageState::ShaderRead,
    );
    // No attachments, so the executor opens no scope of its own and this is the
    // recorder's to open -- which is exactly the bug being tested.
    graph
        .pass("leaky")
        .kind(rg::PassType::Graphics)
        .read_texture(target);
    let compiled = graph.compile().expect("compiles");

    let record = |scope: &PassScope<'_>, list: &mut dyn CommandList| {
        // Opens a scope and never ends it.
        let mut desc = chassis::RenderPassDesc::color(chassis::TextureFormat::Rgba8Unorm);
        desc.label = scope.name.to_owned();
        list.begin_rendering(&desc);
        Ok(())
    };

    let error = executor
        .execute(&compiled, &record)
        .expect_err("an unclosed scope must not be submitted");
    assert!(
        error.to_string().contains("leaky"),
        "the error names the pass, got {error}",
    );
}

/// The recorder is Send + Sync, which is what lets a backend record passes on
/// worker threads.
///
/// Asserted at compile time rather than described in a comment, because the whole
/// justification for `PassRecorder` living in a shared crate is that both
/// backends may call it from anywhere.
#[test]
fn the_pass_recorder_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NoRecording>();
    // A closure capturing nothing is both, which is the trivial case; the trait
    // bound on `execute` is what actually matters and the compiler checks it.
    fn takes_recorder(_: &dyn PassRecorder) {}
    takes_recorder(&NoRecording);
}

// -- the swapchain image: the one slot the frontend cannot fill
//
// No device in any of these. The decision is a function of the compiled frame and
// the handle the acquire produced, and both are plain data -- `chassis::graph::compile`
// needs no driver.

/// Two handles that are not equal, minted the way the backend mints them.
fn handles() -> (chassis::Texture, chassis::Texture) {
    let mut registry = chassis::Registry::<u8, chassis::TextureTag>::new();
    (registry.create(0), registry.create(1))
}

/// A frame as the renderer declares it: its own textures and the swapchain image
/// it composites into, whose slot is therefore the fourth import.
fn compiled_frame() -> rg::CompiledGraph {
    let mut graph = rg::RenderGraph::new();
    let scene_color = graph.import_texture(
        "scene.color",
        rg::ImageState::Undefined,
        rg::ImageState::ShaderRead,
    );
    let scene_depth = graph.import_texture(
        "scene.depth",
        rg::ImageState::Undefined,
        rg::ImageState::DepthAttachment,
    );
    let shadow = graph.import_texture(
        "shadow.map",
        rg::ImageState::Undefined,
        rg::ImageState::ShaderRead,
    );
    let backbuffer = graph.import_texture(
        "backbuffer",
        rg::ImageState::Undefined,
        rg::ImageState::Present,
    );
    assert_eq!(backbuffer, 3);

    graph
        .pass("shadow")
        .kind(rg::PassType::Graphics)
        .depth(shadow, 1.0);
    graph
        .pass("opaque")
        .kind(rg::PassType::Graphics)
        .color_load(scene_color)
        .depth(scene_depth, 1.0)
        .read_texture(shadow);
    graph
        .pass("composite")
        .kind(rg::PassType::Graphics)
        .color(backbuffer, [0.0, 0.0, 0.0, 1.0])
        .read_texture(scene_color);
    graph.compile().expect("the frame compiles")
}

/// A frame that renders to a texture of its own and never to the swapchain.
fn offscreen_frame() -> rg::CompiledGraph {
    let mut graph = rg::RenderGraph::new();
    let target = graph.import_texture(
        "target",
        rg::ImageState::Undefined,
        rg::ImageState::TransferSrc,
    );
    graph
        .pass("draw")
        .kind(rg::PassType::Graphics)
        .color_load(target);
    graph.compile().expect("the frame compiles")
}

// -- the rule, which reads the compiled frame

/// The swapchain image is the resource that has to leave the frame presentable.
///
/// That is the whole rule, and it is the graph's own words: `import_texture`'s doc
/// says the two states differ for the swapchain image, which enters undefined and
/// has to leave in `Present`, and the compiler carries that declaration into the
/// pass list as the epilogue transition.
#[test]
fn the_swapchain_image_is_the_slot_that_has_to_leave_present() {
    let frame = compiled_frame();
    assert_eq!(backbuffer_slot(&frame), Some(3), "slot 3");

    // A frame that renders offscreen ends its target in `TransferSrc` for the
    // readback, which is what every other import looks like: no slot for the
    // backend to fill.
    assert_eq!(backbuffer_slot(&offscreen_frame()), None);
}

/// The acquired image is bound to that slot, and nothing else is touched.
#[test]
fn the_acquired_image_is_bound_to_the_swapchain_slot() {
    let frame = compiled_frame();
    let (scene_color, acquired) = handles();

    let mut slots = SlotTable::default();
    slots.set_texture(0, scene_color);

    resolve_backbuffer(&frame, &mut slots, Some(acquired)).expect("bound");

    assert_eq!(slots.texture(3), Some(acquired));
    assert_eq!(
        slots.texture(0),
        Some(scene_color),
        "the frontend's own registrations stay where they were",
    );
    assert_eq!(
        slots.texture(1),
        None,
        "and a slot nobody bound is still empty rather than filled in",
    );
}

/// A resize mints a new handle for a new image, and the slot has to hold the new
/// one.
#[test]
fn a_new_image_replaces_the_previous_one_in_the_slot() {
    let frame = compiled_frame();
    let (first, second) = handles();

    let mut slots = SlotTable::default();
    resolve_backbuffer(&frame, &mut slots, Some(first)).expect("bound");
    resolve_backbuffer(&frame, &mut slots, Some(second)).expect("rebound");

    assert_eq!(slots.texture(3), Some(second));
    assert_ne!(slots.texture(3), Some(first), "not the retired handle");
}

/// No image, and a frame that draws the swapchain image: refused here, naming the
/// slot, rather than at record time with a slot that resolves to nothing.
#[test]
fn a_frame_with_no_acquired_image_is_refused_and_the_slot_cleared() {
    let frame = compiled_frame();
    let (first, _) = handles();

    let mut slots = SlotTable::default();
    resolve_backbuffer(&frame, &mut slots, Some(first)).expect("bound");
    assert_eq!(slots.texture(3), Some(first), "the previous frame's image");

    // Headless, or an acquire that was skipped. Either way the frame cannot be
    // recorded, and the handle left over from the last frame names an image the
    // surface has already released.
    let error = resolve_backbuffer(&frame, &mut slots, None).expect_err("no image to bind");
    let message = error.to_string();
    assert!(message.contains("slot 3"), "names the slot: {message}");
    assert!(message.contains("swapchain"), "names the image: {message}");
    assert_eq!(
        slots.texture(3),
        None,
        "the retired handle must not survive into a recording",
    );
}

/// A frame that never draws the swapchain image needs no surface at all.
///
/// An offscreen or transfer-only frame is ordinary headless work, and refusing it
/// would make a headless device useless.
#[test]
fn a_frame_that_never_draws_the_swapchain_image_needs_no_image() {
    let mut slots = SlotTable::default();
    resolve_backbuffer(&offscreen_frame(), &mut slots, None).expect("not an error");

    assert_eq!(
        slots.texture_count(),
        0,
        "and nothing was bound, not even a null in a slot",
    );
}

/// The whole path on a device: the slot resolves, through the same `PassScope` a
/// pass records with, to the image the backend bound to it.
///
/// Ignored, not skipped: it needs an adapter, and the image in the slot is a
/// texture standing in for the swapchain's. A swapchain image comes from a
/// `Surface`, a `Surface` comes from a window, and this crate names no windowing
/// library -- the acquire half is driven by `bins/sandbox`, which has one. This is
/// what a person runs when the frame path stops working.
#[test]
#[ignore = "needs a GPU adapter; the swapchain image itself needs a window"]
fn the_recorded_scope_sees_the_bound_swapchain_image() {
    let Some(adapter) = Adapter::request(0) else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-backbuffer") else {
        return;
    };
    let resources = WgpuResources::new(&device);
    let mut executor = WgpuExecutor::new(
        queue,
        resources,
        chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    let acquired = chassis::Resources::create_texture(
        executor.resources_mut(),
        chassis::TextureDesc::new_2d(
            chassis::Extent {
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
            },
            chassis::TextureFormat::Rgba8Unorm,
            chassis::TextureUsage::COLOR_ATTACHMENT,
        ),
    )
    .expect("a stand-in for the acquired image");

    let compiled = compiled_frame();
    let backbuffer = backbuffer_slot(&compiled).expect("the frame draws it");

    executor
        .resolve_backbuffer(&compiled, Some(acquired))
        .expect("the acquired image is bound");

    let seen = std::sync::Mutex::new(None);
    let record = |scope: &PassScope<'_>, _list: &mut dyn CommandList| {
        if scope.name == "composite" {
            *seen.lock().expect("not poisoned") = scope.texture(backbuffer);
        }
        Ok(())
    };
    executor.execute(&compiled, &record).expect("records");

    assert_eq!(
        *seen.lock().expect("not poisoned"),
        Some(acquired),
        "the composite pass can name the backbuffer and get the bound handle",
    );

    // No image at all: refused when it is resolved, and refused again at record
    // time for a caller that never asked -- naming the slot, rather than
    // beginning a render pass with no attachment.
    let error = executor
        .resolve_backbuffer(&compiled, None)
        .expect_err("no image to bind");
    assert!(error.to_string().contains("slot 3"));

    let error = executor
        .execute(&compiled, &record)
        .expect_err("nothing is bound to the swapchain slot");
    assert!(error.to_string().contains("slot 3"));
}
