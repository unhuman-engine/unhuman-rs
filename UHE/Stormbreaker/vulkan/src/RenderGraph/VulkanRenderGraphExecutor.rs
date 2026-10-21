//! Executes a compiled frame: one command list per pass, recorded in parallel,
//! submitted in compiled order.
//!
//! One command list per pass, so one recording thread per pass. A single shared
//! buffer would force recording onto one thread; here recording is parallel and
//! only submission is serial.
//!
//! ## The two decisions that make this different
//!
//! **The executor owns the [`Recorder`].** The graph's compiled order is the
//! submit order, and there is no second source of truth. `Recorder::take` hands
//! out one list per slot and `submit` orders by slot, so slot `i` is compiled
//! pass `i` by construction -- there is no ordering code here to get wrong, and
//! nothing that can reorder a pass relative to its dependencies.
//!
//! **Passes are recorded in parallel.** Each list has its own command pool,
//! which is what makes that legal: the pool, not the buffer, is the external
//! synchronisation object for recording, and a shared pool across worker threads
//! is invalid usage that validation reports as a data race rather than as the
//! missing synchronisation it is. One pool per list exists for exactly this.
//!
//! ## Why the barriers live inside each pass
//!
//! Every pass records its own `pre_barriers` and `post_barriers` into its own
//! command list. That is what keeps a list self-contained: list `i` can be
//! recorded on thread `i` with no knowledge of any other list, because the
//! transitions it needs are the ones it declared. Wrapping a *range* of passes
//! in one buffer would emit barriers only at the range boundary, which is
//! exactly the structure that cannot be parallel.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use chassis::CommandList;

use crate::vulkan_barrier_encoder::BarrierEncoder;
use crate::vulkan_buffer::Buffers;
use crate::vulkan_command_buffer::{CommandList as VulkanCommandList, Recording};
use crate::vulkan_descriptor_manager::Descriptors;
use crate::vulkan_error::{BackendError, Result};
use crate::vulkan_frame_context::Recorder;
use crate::vulkan_gpu_profiler::{self, GpuPassTiming, GpuProfiler, PassQueries};
use crate::vulkan_pipeline_state::Pipelines;
use crate::vulkan_render_graph_barriers::{
    open_render_scope, record_post_barriers, record_pre_barriers,
};
use crate::vulkan_render_graph_resources::FrameRegistrations;
use crate::vulkan_texture::Textures;
use chassis::graph::pass::{PassRecorder, PassScope, SlotTable};
use chassis::graph::CompiledGraph;
use chassis::graph::RenderGraph as Graph;

/// The registries a command list needs to resolve handles.
///
/// Bundled into one borrow so a pass closure takes one argument, and so the
/// executor can check at construction that the registries outlive it.
/// The registries a command list needs, as owned handles.
///
/// ## Why owned rather than borrowed
///
/// `grind::parallel_for` takes an `'static` closure, and a pool spawns its
/// workers at construction, so a recording thread cannot borrow from the frame
/// that queued it. These are `Arc` clones, and the lock is taken inside the
/// recording thread -- once per pass, not once per handle lookup.
///
/// ## Why that is sound
///
/// Creation needs `&mut VulkanResources`, which the recording loop does not have
/// and does not ask for, and `Resources::create_*` takes `&mut self` so no
/// creation can overlap a frame. The write side is therefore uncontended
/// whenever these read guards are held.
pub struct ExecutorResources {
    pub pipelines: Arc<RwLock<Pipelines>>,
    pub textures: Arc<RwLock<Textures>>,
    pub buffers: Arc<RwLock<Buffers>>,
    pub descriptors: Arc<RwLock<Descriptors>>,
}

/// Compile-time proof of the sentence above.
///
/// A comment claiming a registry is read-only during recording is an argument,
/// not a proof. If one ever grows a `Cell`, a `RefCell` or a raw pointer, the
/// recording loop stops compiling here rather than becoming a data race that
/// happens not to fire today.
/// The four registries, as one capturable value.
struct SharedRegistries {
    pipelines: Arc<RwLock<Pipelines>>,
    textures: Arc<RwLock<Textures>>,
    buffers: Arc<RwLock<Buffers>>,
    descriptors: Arc<RwLock<Descriptors>>,
}

impl SharedRegistries {
    fn new(resources: &ExecutorResources) -> Self {
        Self {
            pipelines: Arc::clone(&resources.pipelines),
            textures: Arc::clone(&resources.textures),
            buffers: Arc::clone(&resources.buffers),
            descriptors: Arc::clone(&resources.descriptors),
        }
    }

    /// Another handle on the same four registries.
    ///
    /// Four refcount bumps, not four `Arc`s' worth of anything: each worker takes
    /// one, and drops it when its pass finishes.
    fn shared(&self) -> Self {
        Self {
            pipelines: Arc::clone(&self.pipelines),
            textures: Arc::clone(&self.textures),
            buffers: Arc::clone(&self.buffers),
            descriptors: Arc::clone(&self.descriptors),
        }
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExecutorResources>();
    assert_send_sync::<Pipelines>();
    assert_send_sync::<Textures>();
    assert_send_sync::<Buffers>();
    assert_send_sync::<Descriptors>();
};

/// What recording one pass produced.
struct RecordedPass {
    outcome: Result<()>,
    barriers: u32,
}

/// The size a pass records at: its first attachment's.
///
/// Not the frame's. A shadow pass renders into a square map of its own while the
/// frame is a wide rectangle, so recording it at the frame's extent draws into a
/// corner of the map -- and a depth-only pass with no colour attachment has no
/// colour extent to fall back on, which is a zero-area render pass on Vulkan.
fn attachment_extent(
    pass: &chassis::graph::CompiledPass,
    slots: &SlotTable,
    textures: &Textures,
    frame: chassis::Extent2d,
) -> chassis::Extent2d {
    let color = pass
        .colors
        .first()
        .and_then(|color| slots.texture(color.texture))
        .and_then(|handle| textures.get(handle))
        .map(|texture| chassis::Extent2d {
            width: texture.extent().width,
            height: texture.extent().height,
        });

    color
        .or_else(|| {
            pass.depth
                .as_ref()
                .and_then(|depth| slots.texture(depth.texture))
                .and_then(|handle| textures.get(handle))
                .map(|texture| chassis::Extent2d {
                    width: texture.extent().width,
                    height: texture.extent().height,
                })
        })
        // The backbuffer has no `chassis::Texture` on Vulkan -- the swapchain owns
        // it -- so a pass writing to it falls back to the frame's own extent.
        .unwrap_or(frame)
}

/// Record one pass into its own list.
///
/// A named function rather than an inline block, for one reason: a closure handed
/// to `grind::parallel_for` cannot borrow, so the body had to be reachable
/// without capturing anything from the frame. That it is now separately testable
/// is a side benefit.
///
#[allow(clippy::too_many_arguments)]
fn record_pass(
    pass: &chassis::graph::CompiledPass,
    position: u32,
    pass_count: u32,
    queries: Option<PassQueries>,
    encoder: &BarrierEncoder,
    slots: &SlotTable,
    textures: &Textures,
    buffers: &Buffers,
    extent: &chassis::Extent2d,
    list: &mut VulkanCommandList<'_>,
    record: &dyn PassRecorder,
) -> RecordedPass {
    let mut barriers = 0u32;
    let outcome = (|| -> Result<()> {
        if let Some(queries) = queries {
            if position == 0 {
                list.raw().reset_query_pool(queries.pool, queries.capacity);
                vulkan_gpu_profiler::write_timestamp(list.raw(), queries, 0);
            }
        }

        barriers += record_pre_barriers(encoder, pass, slots, textures, buffers, list)?;

        if let Some(queries) = queries {
            let index = 2 + position * 2;
            vulkan_gpu_profiler::write_timestamp(list.raw(), queries, index);
        }

        // Tell the list what layouts the graph says are current, so a transfer in
        // this pass copies from the right layout instead of guessing one.
        // Declared rather than tracked: the graph is the only thing that knows,
        // and a running total across a list would need the previous list's
        // ending state, which does not exist while they record in parallel.
        for barrier in &pass.pre_barriers {
            if let Some(handle) = slots.texture(barrier.texture) {
                list.declare_layout(handle, barrier.dst_state);
            }
        }

        // The executor opens the render scope, not the pass.
        //
        // The graph already knows the attachments -- that is what
        // `CompiledPass::colors` and `.depth` are -- so a pass closure that had
        // to open the scope could open it on the wrong attachments, and
        // validation would say `vkCmdDraw must be issued inside an active render
        // pass` with nothing naming who was responsible.
        let pass_extent = attachment_extent(pass, slots, textures, *extent);
        let rendering = open_render_scope(pass, slots, textures, &pass_extent, list)?;

        let scope = PassScope {
            index: pass.index,
            name: &pass.name,
            extent: pass_extent,
            slots,
        };

        record
            .record(&scope, list)
            .map_err(|error| BackendError::PassRecording(error.to_string()))?;

        if rendering {
            list.end_rendering();
        }

        // After the body and after the render scope, which is the only place a
        // post barrier can go.
        record_post_barriers(encoder, pass, slots, textures, list)?;

        if let Some(queries) = queries {
            if position + 1 == pass_count {
                vulkan_gpu_profiler::write_timestamp(list.raw(), queries, 1);
            }
            vulkan_gpu_profiler::write_timestamp(list.raw(), queries, 3 + position * 2);
        }
        Ok(())
    })();

    RecordedPass { outcome, barriers }
}

/// What the executor did, for a frame dump and for tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutorStats {
    /// Passes submitted.
    pub passes: u32,
    /// Barriers recorded across the frame.
    pub barriers: u32,
    /// Frames this executor has run.
    pub frames: u64,
}

/// Owns the recorder and the graph's execution.
///
/// Owning the [`Recorder`] is the whole point of the type existing: submit order
/// has exactly one source, and it is the order the graph compiled.
pub struct VulkanRenderGraphExecutor {
    /// Handle-to-driver-object pairings for the frame in flight.
    registrations: FrameRegistrations,
    recorder: Recorder,
    resources: SlotTable,
    extent: chassis::Extent2d,
    /// The timeline value the next frame's chain starts from. Advanced by the
    /// chain length every frame, because a timeline value must strictly
    /// increase and reusing one is a validation error that reads as a hang.
    value_base: u64,
    stats: ExecutorStats,
    last_frame: usize,
    /// `Some` only when a caller asked for per-pass GPU times. Recording the
    /// timestamps otherwise would be a cost every frame pays for a benchmark.
    profiler: Option<GpuProfiler>,
}

impl VulkanRenderGraphExecutor {
    pub fn new(recorder: Recorder, extent: chassis::Extent2d) -> Self {
        Self {
            registrations: FrameRegistrations::new(),
            recorder,
            resources: SlotTable::default(),
            extent,
            value_base: 1,
            stats: ExecutorStats::default(),
            last_frame: 0,
            profiler: None,
        }
    }

    /// Turns on per-pass timestamp queries. Returns false when the queue cannot
    /// write timestamps, which is a driver property and not an error.
    pub fn enable_gpu_timing(&mut self, period: f32, valid_bits: u32) -> bool {
        self.profiler = GpuProfiler::new(self.recorder.device(), period, valid_bits);
        self.profiler.is_some()
    }

    /// The last collected per-pass times, empty before a frame is collected.
    pub fn gpu_timings(&self) -> &[GpuPassTiming] {
        self.profiler.as_ref().map_or(&[], GpuProfiler::timings)
    }

    /// Reads the timestamps of the frame just submitted. The caller must have
    /// waited for its fence; without that the values are not available yet.
    pub fn collect_gpu_timings(&mut self) -> Result<()> {
        match self.profiler.as_mut() {
            Some(profiler) => profiler.collect(),
            None => Ok(()),
        }
    }

    pub fn gpu_frame_ms(&self) -> Option<f32> {
        self.profiler.as_ref().and_then(GpuProfiler::frame_ms)
    }

    pub fn slots(&self) -> &SlotTable {
        &self.resources
    }

    /// Registrations for the frame being recorded.
    ///
    /// Cleared at the start of every `execute`, so the table's lifetime is one
    /// frame. See `vulkan_render_graph_resources` for why that matters.
    pub fn registrations(&mut self) -> &mut FrameRegistrations {
        &mut self.registrations
    }

    pub fn slots_mut(&mut self) -> &mut SlotTable {
        &mut self.resources
    }

    /// Tell the executor how big the frame is, so passes get a viewport without
    /// every one of them having to ask the swapchain.
    pub fn set_extent(&mut self, extent: chassis::Extent2d) {
        self.extent = extent;
    }

    pub fn stats(&self) -> ExecutorStats {
        self.stats
    }

    pub fn recorder(&self) -> &Recorder {
        &self.recorder
    }

    pub fn recorder_mut(&mut self) -> &mut Recorder {
        &mut self.recorder
    }

    /// Record and submit a compiled frame.
    ///
    /// Takes the compiled graph, which is pure data: it cannot change underneath
    /// this, so there is nothing to hold a lock against while passes record.
    pub fn execute(
        &mut self,
        frame: &CompiledGraph,
        scope_resources: ExecutorResources,
        record: Arc<dyn PassRecorder>,
        jobs: &grind::Jobs,
    ) -> Result<crate::vulkan_submission_ring::SubmissionToken> {
        let passes = &frame.passes;
        if passes.is_empty() {
            // A frame with nothing in it still advances the frame counter, so a
            // graph that compiles to nothing every frame -- a real bug, and one
            // the graph tests cover -- is visible here rather than silent.
            self.stats.frames += 1;
            return Ok(crate::vulkan_submission_ring::SubmissionToken::NONE);
        }

        // The query pool is reset by the first command list, so the previous
        // frame's reset must have retired before this one records over it.
        if self.profiler.is_some() {
            self.recorder.submissions().wait_all()?;
            let names: Vec<&str> = passes.iter().map(|pass| pass.name.as_str()).collect();
            if let Some(profiler) = self.profiler.as_mut() {
                profiler.begin_frame(&names);
            }
        }

        // Resolve the whole frame *before* a single command buffer is recorded.
        //
        // A slot pointing at nothing is then one error, reported on this thread,
        // naming a handle -- rather than a failure on whichever worker reached
        // that pass, naming a thread. Resolve is separated from record for the
        // same reason.
        self.registrations.clear();
        {
            let textures = scope_resources.textures.read().expect("registry lock");
            let buffers = scope_resources.buffers.read().expect("registry lock");
            FrameRegistrations::resolve_frame(
                &self.registrations,
                &self.slots().clone(),
                &textures,
                &buffers,
                frame,
            )?;
        }

        let mut recordings = self.recorder.take(passes.len())?;
        let tier = self.recorder.tier();
        let device = self.recorder.device().clone();
        let queries = self
            .profiler
            .as_ref()
            .map(|profiler| profiler.queries(tier.uses_sync2()));

        // A pass recorder returns `chassis::Error`; the rest of this function
        // speaks the backend's. Bridged once, here, rather than by making every
        // call site convert.

        // One encoder for the frame, shared by every pass.
        //
        // It owns its `ash::Device` rather than borrowing, because the passes
        // run on worker threads and a closure handed to `grind::parallel_for` is
        // `'static`. An `ash::Device` clone is two words and one driver refcount
        // bump, once per frame.

        // ## Recording in parallel
        //
        // List `i` is a distinct command buffer out of a distinct command pool,
        // so pass `i` needs nothing from pass `j`. That is not an accident of the
        // arena -- it is what a per-pass command pool is *for*. The spec's
        // external-synchronisation object for recording is the **pool**, not the
        // buffer, so one pool per pass is exactly what makes concurrent recording
        // sound. A shared pool would be the same threading error as sharing one
        // pool across threads, which is the mistake this layout avoids.
        //
        // The registries are read-locked once per pass, not once per handle
        // lookup: four uncontended atomics per pass, against a per-lookup cost on
        // the hottest path in recording.
        let pass_count = passes.len();
        // Cloned so the workers can hold it: `parallel_for` takes an `'static`
        // closure and cannot borrow the caller's frame. `CompiledPass` is pure
        // data -- no closures, no backend types -- so a clone is a few strings
        // and a couple of small vecs, once per frame.
        let pass_list: Arc<Vec<chassis::graph::CompiledPass>> = Arc::new(passes.to_vec());
        let extent = self.extent;
        // `Arc`d, so the closure a worker runs outlives this frame's borrow of
        // them. A `SlotTable` clone rather than a borrow for the same reason.
        let slots = Arc::new(self.slots().clone());
        // One encoder for the frame, shared by every pass.
        //
        // It owns its `ash::Device` rather than borrowing, because the passes run
        // on worker threads and a closure handed to `grind::parallel_for` is
        // `'static`. An `ash::Device` clone is two words and one driver refcount
        // bump, once per frame.
        let encoder = Arc::new(BarrierEncoder::new(tier, &device));
        let shared = SharedRegistries::new(&scope_resources);
        let recorder: Arc<dyn PassRecorder> = Arc::clone(&record);
        let barriers_total = Arc::new(AtomicU32::new(0));
        // One slot per pass. A failure is recorded *in its own slot*, not in the
        // first to arrive, so two passes failing at once cannot make which error
        // comes back depend on which thread won the race.
        let failures = Arc::new(Mutex::new(vec![None::<BackendError>; pass_count]));
        // Each worker takes its own recording out and sends it back, rather than
        // sharing a `Vec` it would have to mutate. `parallel_for` takes an `Fn`,
        // and `Fn` cannot mutate what it captured.
        let (sender, finished) = std::sync::mpsc::channel();
        // `Option` so a worker can *take* its own recording out by index. A
        // worker and the submit path must not both be able to reach the same
        // command buffer, and taking is what makes that structurally impossible
        // rather than a matter of discipline.
        let arena = Arc::new(Mutex::new(
            recordings.into_iter().map(Some).collect::<Vec<_>>(),
        ));
        let shared_recordings = Arc::clone(&arena);

        let parallel = {
            let pass_list = Arc::clone(&pass_list);
            let slots = Arc::clone(&slots);
            let encoder = Arc::clone(&encoder);
            let shared = shared.shared();
            let recorder = Arc::clone(&recorder);
            let barriers_total = Arc::clone(&barriers_total);
            let failures = Arc::clone(&failures);

            move |position: u32| {
                let position = position as usize;
                let sender = sender.clone();
                let Some(pass) = pass_list.get(position) else {
                    return;
                };
                let Some(mut recording) = shared_recordings.lock().unwrap()[position].take() else {
                    return;
                };

                // Read guards for the whole pass. Held together, so all four are
                // released at once and no writer can slip between two of them.
                let pipelines = shared.pipelines.read().expect("registry lock");
                let textures = shared.textures.read().expect("registry lock");
                let buffers = shared.buffers.read().expect("registry lock");
                let descriptors = shared.descriptors.read().expect("registry lock");

                let mut list = VulkanCommandList::new(
                    &mut recording,
                    &pipelines,
                    &textures,
                    &buffers,
                    &descriptors,
                );

                let recorded = record_pass(
                    pass,
                    position as u32,
                    pass_count as u32,
                    queries,
                    &encoder,
                    &slots,
                    &textures,
                    &buffers,
                    &extent,
                    &mut list,
                    recorder.as_ref(),
                );
                barriers_total.fetch_add(recorded.barriers, Ordering::Relaxed);

                if let Err(error) = recorded.outcome {
                    let mut guard = failures.lock().unwrap();
                    if guard[position].is_none() {
                        guard[position] = Some(error);
                    }
                }

                // The receiver outlives `parallel_for` and is drained straight
                // after it, so this cannot fail -- and a dropped send would be a
                // command buffer never ended, which is the one leak this type can
                // make.
                sender
                    .send((position, recording))
                    .expect("the recording channel outlives parallel_for");
            }
        };

        // One job per pass, batch size one: recording a pass is not a cheap body
        // that wants slicing, and batching would serialise passes that are
        // independent.
        jobs.parallel_for(pass_count as u32, parallel, 1);

        let barrier_count = barriers_total.load(Ordering::Relaxed);

        // Which thread finished first is not an order, so re-sort by pass index.
        let mut returned: Vec<(usize, Recording)> = finished.try_iter().collect();
        returned.sort_by_key(|(position, _)| *position);
        recordings = returned
            .into_iter()
            .map(|(_, recording)| recording)
            .collect();

        // The arena is taken under the lock rather than unwrapped.
        //
        // Unwrapping the `Arc` is the obvious move and it is wrong: `JobCounter`
        // settles *before* the job closure is destroyed, so a worker can still
        // hold a clone for a few instructions after `wait` returns, and
        // `try_unwrap` would fail -- with `Recording` not being `Clone`, there
        // being no fallback. Taking works whether or not anyone else still holds
        // a reference, and leaves behind an empty mutex.
        {
            let mut guard = arena.lock().unwrap();
            for slot in guard.iter_mut() {
                *slot = None;
            }
        }

        let failure = failures.lock().unwrap().iter().flatten().next().cloned();

        if let Some(error) = failure {
            // The recordings were begun, so they must be ended or the pool reset
            // is illegal on a buffer in the recording state. Returning them
            // unended is the one leak this type can make.
            // End, then hand back. Both halves matter: an ended buffer that is
            // not returned shrinks the arena by one per failed frame, and a
            // buffer that is returned without ending is still in the recording
            // state when the pool is reset.
            let mut ended = Vec::with_capacity(recordings.len());
            for recording in recordings.drain(..) {
                match recording.end() {
                    Ok(ended_recording) => ended.push(ended_recording),
                    Err(error) => {
                        // Already failed; report the original, which is the one
                        // that explains why.
                        let _ = error;
                    }
                }
            }
            self.recorder.recycle(ended);
            return Err(error);
        }

        let finished: Vec<crate::vulkan_command_buffer::Recording> = recordings
            .into_iter()
            .map(|recording| recording.end())
            .collect::<Result<Vec<_>>>()?;
        // One submit, not one per pass. The compiled graph emits a barrier for
        // every cross-pass dependency, and those barriers order the passes across
        // a single submission, so the per-link semaphore chain bought nothing but
        // one `vkQueueSubmit2` per pass.
        let token = self.recorder.submit_batched(finished)?;
        // `value_base` still advances: `transition_to_present` rides the chain,
        // and it must signal a value past the timeline's current one. The batched
        // path signals nothing, so skipping values here is not a duplicate.
        self.value_base += passes.len().max(1) as u64;

        self.stats.passes = passes.len() as u32;
        self.stats.barriers = barrier_count;
        self.stats.frames += 1;
        self.last_frame = passes.len();
        // No wait here. The token names the fence this frame will signal, and a
        // readback waits on that fence when it reads the slot this frame wrote.
        Ok(token)
    }

    /// Move one image into `PRESENT_SRC_KHR` and submit that.
    ///
    /// ## Why this is here and not in the render graph
    ///
    /// The graph derives barriers from declared reads and writes, and a
    /// swapchain image is never declared by anyone -- it is handed out by
    /// `vkAcquireNextImageKHR`. So the final transition to a presentable layout
    /// has no pass that could own it, and without it `vkQueuePresentKHR` fails
    /// with `VUID-VkPresentInfoKHR-pImageIndices-01430`: "must be in
    /// `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`, but is in `UNDEFINED`".
    ///
    /// ## Why `UNDEFINED` on the way in
    ///
    /// Discarding is always legal, so this does not care what was there before --
    /// including a previous frame's contents. That is the one thing a present
    /// transition must not assume.
    ///
    /// ## One list, submitted on its own
    ///
    /// Not appended to the frame's lists: the frame is already submitted by the
    /// time a swapchain is presentable. It returns its fence token so the present
    /// path can wait on exactly this transition, not the whole device.
    pub fn transition_to_present(
        &mut self,
        image: ash::vk::Image,
        format: chassis::TextureFormat,
    ) -> Result<crate::vulkan_submission_ring::SubmissionToken> {
        let tier = self.recorder.tier();
        let mut recordings = self.recorder.take(1)?;
        let mut recording = match recordings.pop() {
            Some(recording) => recording,
            None => {
                return Err(crate::vulkan_error::BackendError::ListArenaExhausted {
                    requested: 1,
                    available: 0,
                })
            }
        };

        // One barrier description, lowered for the tier. Building it twice and
        // letting each lowering read its own copy is how the two spellings drift.
        let barrier = chassis::graph::types::ImageBarrier {
            // Not a real texture: this barrier names an image directly, and
            // nothing reads `texture` when the target is given.
            texture: 0,
            src_access: chassis::graph::types::Access::None,
            dst_access: chassis::graph::types::Access::None,
            src_stage: chassis::graph::types::Stage::TopOfPipe,
            dst_stage: chassis::graph::types::Stage::BottomOfPipe,
            src_state: chassis::graph::types::ImageState::Undefined,
            dst_state: chassis::graph::types::ImageState::Present,
        };
        let target = crate::vulkan_barrier_encoder::Target::color(image, 1, 1);

        if tier.uses_sync2() {
            let barrier2 = crate::vulkan_barrier_encoder::image_barrier2(target, barrier, format);
            recording.barrier_image(&[barrier2]);
        } else {
            recording.barrier_image_legacy(&[crate::vulkan_barrier_encoder::image_barrier_legacy(
                target, barrier, format,
            )]);
        }

        recordings.push(recording.end()?);
        let token = self.recorder.submit(recordings, self.value_base);
        // The timeline value has to advance for *every* submit, including this
        // one. Without it the present transition signals the value the frame's
        // last pass already signalled, and that is
        // `VUID-VkSubmitInfo2-semaphore-03882`: "signal value (2) must be greater
        // than current value (2)".
        self.value_base += 1;
        token
    }

    pub fn last_frame_passes(&self) -> usize {
        self.last_frame
    }
}

/// The graph, and the executor that runs it, as one value.
///
/// The pairing is the point: the executor holds the recorder, so a frame cannot
/// be submitted from anywhere but here.
pub struct RenderGraphWithExecutor {
    pub graph: Graph,
    pub executor: VulkanRenderGraphExecutor,
}
