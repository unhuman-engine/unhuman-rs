//! The wgpu render graph executor.
//!
//! ## Why this lives in `RenderGraph/`
//!
//! Because the Vulkan backend puts its executor there, and a reader who knows
//! where to look in one backend should not have to go looking in the other. Both
//! backends keep their executor in a `RenderGraph/` directory.
//!
//! ## What is the same and what is not
//!
//! The *graph* is shared: `chassis::graph` compiles a frame, and both backends
//! consume the same `CompiledGraph`. Ordering, culling and versioning are
//! identical, and they have to be -- two backends that ordered a frame
//! differently would not be interchangeable, they would be two engines.
//!
//! The *execution* is not shared, and the difference is not a workaround:
//!
//!  * Vulkan records barriers. `chassis::graph` emits them from declared reads
//!    and writes; `vulkan_render_graph_barriers` lowers them.
//!  * wgpu emits nothing. It inserts transitions itself, from the access a
//!    render pass declares. A barrier recorded here would be a lie about a
//!    pipeline that does not take them.
//!
//! So the two executors share a shape and not a body. Each has its own
//! `ExecutorResources`, its own command buffer type, and its own idea of what a
//! frame costs.

//! Executes a compiled frame on wgpu.
//!
//! ## The one thing this does not do
//!
//! **It records no barriers.** The graph emits backend-neutral barriers and the
//! Vulkan executor lowers them through `VulkanBarrierEncoder`, choosing
//! `vkCmdPipelineBarrier2` or `vkCmdPipelineBarrier` per the device's `SyncTier`.
//! wgpu has no per-barrier control at all -- it inserts its own transitions -- so
//! this executor ignores the graph's barrier output entirely.
//!
//! That is not a gap, and it is worth being precise about why it is safe:
//! barriers are *one backend's expression* of the graph's ordering. Ordering is
//! what makes a frame correct and what lets passes record in parallel;
//! expressions are per-backend. The graph would have been unusable from a second
//! backend if its compiler emitted `VkImageMemoryBarrier`, which is exactly why
//! it emits neutral ones.
//!
//! ## What is shared with the Vulkan executor, verbatim
//!
//! `SlotTable`, `PassScope`, `PassRecorder` and `NoRecording` all live in
//! `chassis::graph::pass`. A renderer implements `PassRecorder` once and both
//! backends call it with the same scope. That is what "the front end is the same
//! and only the renderer has options" actually means.
//!
//! ## What wgpu makes simpler
//!
//! No command pools. The Vulkan executor takes one list per pass from a `Recorder`
//! that owns a pool per slot, and that pool is the external-synchronisation
//! object for recording -- sharing one across worker threads is invalid usage. wgpu
//! has no equivalent: a `Recording` owns its own encoder and there is nothing to
//! share. No timeline semaphores either; `queue.submit` is the whole synchronisation
//! story and wgpu orders submissions to one queue itself.
//!
//! ## The swapchain image, which is the one slot the frontend cannot fill
//!
//! A graph names a texture by slot, and the slot *is* the `TextureId` the frame's
//! `import_texture`/`create_texture` returned -- import order, not a number anybody
//! chooses. The frontend fills `SlotTable` for every texture it created; the one it
//! cannot is the swapchain's, because the image belongs to the surface and arrives
//! with the acquire, so there is no RHI `Texture` handle to put there until a
//! backend mints one for the image it acquired. That is what
//! [`resolve_backbuffer`] does, from the frame loop, between the acquire and the
//! recording -- so `PassScope::texture` answers for every pass in the frame,
//! including the one that composites into the swapchain. [`backbuffer_slot`] is
//! how the slot is found: from the compiled frame, which is all a backend is ever
//! handed.

use chassis::graph::pass::{PassRecorder, PassScope, SlotTable};
use chassis::graph::{CompiledGraph, ImageState, TextureId};
use chassis::CommandList;

use crate::wgpu_command_buffer::{Recording, WgpuCommandList};
use crate::wgpu_device::Queue;
use crate::wgpu_resources::WgpuResources;

/// What the executor did, for a frame dump and for tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutorStats {
    /// Passes submitted.
    pub passes: u32,
    /// Command buffers finished.
    pub buffers: u32,
    /// Frames run.
    pub frames: u64,
}

/// Runs compiled frames.
///
/// Owns nothing that has to be torn down -- wgpu owns its objects -- so this is a
/// thin type by design. It exists to make the ordering guarantee legible in one
/// place rather than spread across a call site.
pub struct WgpuRenderGraphExecutor {
    queue: Queue,
    resources: WgpuResources,
    slots: SlotTable,
    extent: chassis::Extent2d,
    stats: ExecutorStats,
}

impl WgpuRenderGraphExecutor {
    pub fn new(queue: Queue, resources: WgpuResources, extent: chassis::Extent2d) -> Self {
        Self {
            queue,
            resources,
            slots: SlotTable::default(),
            extent,
            stats: ExecutorStats::default(),
        }
    }

    pub fn stats(&self) -> ExecutorStats {
        self.stats
    }

    pub fn slots(&self) -> &SlotTable {
        &self.slots
    }

    pub fn slots_mut(&mut self) -> &mut SlotTable {
        &mut self.slots
    }

    pub fn resources(&self) -> &WgpuResources {
        &self.resources
    }

    /// Mutable, because the executor has to own the only registry.
    ///
    /// It has to, and that was a bug: `WgpuGpu` once built a `WgpuResources` for
    /// itself *and* handed a second one to the executor, so a texture created
    /// through `dyn Resources` had a handle the executor could not resolve. Every
    /// such use failed with `InvalidHandle` at record time -- far from the cause.
    ///
    /// One registry, owned here, reachable from both directions.
    pub fn resources_mut(&mut self) -> &mut WgpuResources {
        &mut self.resources
    }

    /// The frame's size, so a pass gets a viewport without asking the swapchain.
    pub fn set_extent(&mut self, extent: chassis::Extent2d) {
        self.extent = extent;
    }

    /// [`resolve_backbuffer`] against the table this executor records from.
    ///
    /// Once per frame, after the acquire and before [`Self::execute`] -- it is the
    /// first half of that call and the reason it can record at all.
    pub fn resolve_backbuffer(
        &mut self,
        graph: &CompiledGraph,
        acquired: Option<chassis::Texture>,
    ) -> Result<(), chassis::Error> {
        resolve_backbuffer(graph, &mut self.slots, acquired)
    }

    /// Record every pass and submit, in the graph's compiled order.
    ///
    /// `recordings` must be in compiled order and the submit keeps it, because
    /// wgpu executes submissions to one queue in the order they were made. That is
    /// what replaces the Vulkan executor's whole chain of timeline semaphore
    /// waits: same guarantee, one fewer piece of machinery to get wrong.
    ///
    /// [`Self::resolve_backbuffer`] comes first: a frame that draws the swapchain
    /// image records against a slot nothing has bound until it has run.
    pub fn execute(
        &mut self,
        graph: &CompiledGraph,
        record: &dyn PassRecorder,
    ) -> Result<(), chassis::Error> {
        // The swapchain slot is the backend's to bind, so a frame that draws one
        // and never got it bound is refused here rather than at `begin_rendering`
        // -- where the message would name a pass and not the image that was never
        // bound to it.
        let unbound = backbuffer_slot(graph).filter(|slot| self.slots.texture(*slot).is_none());
        if let Some(slot) = unbound {
            return Err(chassis::Error::Unsupported(format!(
                "the frame draws the swapchain image (slot {slot}) but nothing is bound to \
                 it: `resolve_backbuffer` did not run for this frame"
            )));
        }

        let mut buffers = Vec::with_capacity(graph.passes.len());

        for pass in &graph.passes {
            let mut recording = Recording::new(self.resources.device().inner(), &pass.name);
            {
                let mut list = WgpuCommandList::new(&mut recording, &self.resources);
                let pass_extent =
                    attachment_extent(pass, &self.slots, &self.resources, self.extent);
                let scope = PassScope {
                    index: pass.index,
                    name: &pass.name,
                    extent: pass_extent,
                    slots: &self.slots,
                };
                let opened =
                    open_render_scope(pass, &self.slots, &self.resources, pass_extent, &mut list)?;
                record.record(&scope, &mut list)?;
                if opened {
                    list.end_rendering();
                }
            }
            if recording.is_inside_render_pass() {
                // A pass that began a render scope and never ended it. Submitted
                // as-is, wgpu ends the scope at `finish` and the tail of the pass
                // is quietly lost -- so it is refused here, where the message can
                // name the pass.
                return Err(chassis::Error::Unsupported(format!(
                    "pass '{}' ended while still inside a render pass",
                    pass.name
                )));
            }
            buffers.push(recording.finish());
        }

        let submitted = buffers.len() as u32;
        // Moved in, not copied: a `CommandBuffer` is not `Copy`, and the
        // executor has no use for them afterwards.
        self.queue.inner().submit(buffers);

        self.stats.passes = graph.passes.len() as u32;
        self.stats.buffers = submitted;
        self.stats.frames += 1;
        Ok(())
    }

    /// Block until the queue has drained.
    ///
    /// Test and shutdown only -- the frame loop never calls this, for the same
    /// reason the Vulkan backend's `Gpu::wait_idle` is documented as test-only.
    pub fn wait_idle(&self) -> Result<(), chassis::Error> {
        self.resources
            .device()
            .inner()
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| chassis::Error::Unsupported(format!("poll failed: {error:?}")))
    }
}

/// Begin dynamic rendering for a pass, if it has attachments.
///
/// The executor opens the scope, not the pass: the compiled pass already names
/// its attachments, so a pass that opened its own could open it on the wrong
/// ones, and the failure would surface as a draw with no active render pass.
/// Returns whether a scope was opened, so the caller closes exactly what it
/// opened. A pass with no attachments -- a compute or transfer pass -- gets no
/// scope.
/// The size a pass records at: its first attachment's.
///
/// Not the frame's. A shadow pass renders into a square map of its own while the
/// frame is a wide rectangle, so recording it at the frame's extent draws into a
/// corner of the map. The Vulkan executor asks the same question of its own
/// textures, so the two record a pass at the same size.
fn attachment_extent(
    pass: &chassis::graph::CompiledPass,
    slots: &SlotTable,
    resources: &WgpuResources,
    frame: chassis::Extent2d,
) -> chassis::Extent2d {
    let size = |handle: chassis::Texture| {
        resources
            .get_texture(handle)
            .map(|texture| chassis::Extent2d {
                width: texture.extent().width,
                height: texture.extent().height,
            })
    };

    pass.colors
        .first()
        .and_then(|color| slots.texture(color.texture))
        .and_then(size)
        .or_else(|| {
            pass.depth
                .as_ref()
                .and_then(|depth| slots.texture(depth.texture))
                .and_then(size)
        })
        // The backbuffer has no `chassis::Texture` here either: the surface owns
        // it, so a pass writing to it falls back to the frame's own extent.
        .unwrap_or(frame)
}

fn open_render_scope(
    pass: &chassis::graph::CompiledPass,
    slots: &SlotTable,
    resources: &WgpuResources,
    extent: chassis::Extent2d,
    list: &mut WgpuCommandList<'_>,
) -> Result<bool, chassis::Error> {
    if pass.colors.is_empty() && pass.depth.is_none() {
        return Ok(false);
    }

    let unbound = |kind: &str| {
        chassis::Error::Unsupported(format!(
            "pass '{}' has a {kind} attachment on a slot with no texture bound",
            pass.name
        ))
    };

    let mut desc = chassis::RenderPassDesc {
        label: pass.name.clone(),
        ..Default::default()
    };
    for color in &pass.colors {
        let handle = slots
            .texture(color.texture)
            .ok_or_else(|| unbound("colour"))?;
        let format = resources
            .get_texture(handle)
            .map(|texture| texture.format())
            .ok_or_else(|| unbound("colour"))?;
        desc.colors.push(chassis::ColorAttachmentDesc {
            texture: handle,
            format,
            load: load_op(color.load),
            clear: color.clear,
            store: true,
            resolve: chassis::Texture::null(),
        });
    }
    if let Some(depth) = &pass.depth {
        let handle = slots
            .texture(depth.texture)
            .ok_or_else(|| unbound("depth"))?;
        let format = resources
            .get_texture(handle)
            .map(|texture| texture.format())
            .ok_or_else(|| unbound("depth"))?;
        desc.depth = Some(chassis::DepthAttachmentDesc {
            texture: handle,
            format,
            load: load_op(depth.load),
            clear_depth: depth.clear_depth,
            store: true,
        });
    }

    list.begin_rendering(&desc);
    list.set_viewport(extent);
    list.set_scissor(0, 0, extent.width, extent.height);
    Ok(true)
}

/// The graph's load op as the RHI's: same three variants, one crate apart.
fn load_op(load: chassis::graph::LoadOp) -> chassis::LoadOp {
    match load {
        chassis::graph::LoadOp::Load => chassis::LoadOp::Load,
        chassis::graph::LoadOp::Clear => chassis::LoadOp::Clear,
        chassis::graph::LoadOp::DontCare => chassis::LoadOp::DontCare,
    }
}

/// The slot a frame draws the swapchain image into, if it draws one at all.
///
/// The swapchain image is the one resource a frame claims is *external* and has
/// to leave presentable: the renderer imports it with `import_texture`, declared
/// `Undefined` in and `Present` out, because the image arrives with the acquire
/// and the frame ends with a present. The compiler carries that declaration into
/// the pass list as the epilogue transition to `Present` on the last pass that
/// touches the resource (`synthesize_barriers`' own comment: "A swapchain image,
/// for instance, has to reach `Present` after the last pass that writes it"), so
/// the compiled frame names the slot on its own -- which matters, because the
/// compiled frame is all a backend is handed.
///
/// Nothing else in a frame reaches `Present`: every other texture ends at
/// whatever state the pass that last touched it left it in.
pub fn backbuffer_slot(graph: &CompiledGraph) -> Option<TextureId> {
    graph
        .passes
        .iter()
        .flat_map(|pass| pass.pre_barriers.iter().chain(&pass.post_barriers))
        .find(|barrier| barrier.dst_state == ImageState::Present)
        .map(|barrier| barrier.texture)
}

/// Bind the frame's acquired swapchain image to the slot it was imported at.
///
/// ## Who calls this, and when
///
/// The frame loop: once per frame, after the acquire and before any pass records
/// (`WgpuGpu::submit` is the caller here). Not the frontend, and not a pass -- a
/// swapchain image has no RHI `Texture` handle to bind until a backend mints one
/// for the image it acquired, which is why this exists at all.
///
/// ## `None` is an error, not a no-op
///
/// A frame that draws the swapchain image and has no image to draw into cannot be
/// recorded: the pass would attach nothing, and the failure would surface at
/// `begin_rendering` with no word about the slot or the frame. `None` covers two
/// situations and both are the caller's to fix -- a headless device, where the
/// frame asked for something that is not there, and an acquire that timed out or
/// came back occluded, where the frame should not have been submitted.
///
/// The only no-op is a frame that never draws the swapchain image: an offscreen or
/// transfer-only frame needs nothing from the surface, and is fine headless.
///
/// The slot is cleared in every case where it is not bound, because the handle
/// from the previous frame names an image the surface has already released -- a
/// resize, a present, or a skipped acquire. Leaving it in the table would resolve,
/// at record time, to a texture no longer in the registry: a missing attachment
/// rather than this error.
pub fn resolve_backbuffer(
    graph: &CompiledGraph,
    slots: &mut SlotTable,
    acquired: Option<chassis::Texture>,
) -> Result<(), chassis::Error> {
    let Some(slot) = backbuffer_slot(graph) else {
        return Ok(());
    };
    match acquired.filter(|texture| !texture.is_null()) {
        Some(texture) => {
            slots.set_texture(slot, texture);
            Ok(())
        }
        None => {
            slots.set_texture(slot, chassis::Texture::null());
            Err(chassis::Error::Unsupported(format!(
                "the frame draws the swapchain image (slot {slot}) but no image is acquired: \
                 this device has no surface, or the acquire was skipped"
            )))
        }
    }
}
