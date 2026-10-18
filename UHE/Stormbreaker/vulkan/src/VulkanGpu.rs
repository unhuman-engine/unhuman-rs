//! `chassis::Gpu` for Vulkan.
//!
//! ## The other half of the seam
//!
//! [`crate::vulkan_resources`] has been a working `chassis::Resources` for a
//! while. This is the `Gpu` that sits above it, and it is deliberately shaped
//! like [`crate::wgpu_gpu::WgpuGpu`] method for method: same constructors, same
//! accessors, same trait impls. Two backends that differ in their accessor names
//! are not interchangeable however well the trait fits.
//!
//! Where they genuinely differ, the difference is in the return type and nothing
//! else:
//!
//! | | wgpu | Vulkan |
//! |---|---|---|
//! | headless | ordinary | ordinary |
//! | swapchain | none yet | `Option<VulkanSwapchain>` |
//! | queue | one, unnumbered | a family index out of several |
//!
//! ## What this backend is for
//!
//! Bare-metal control: per-barrier transitions chosen from the device's
//! `SyncTier`, explicit memory, and the only path that reaches a 1.0 device.
//! It is opt-in (`--backend vulkan`) because wgpu is the default and this is the
//! one more likely to be the thing that is misbehaving.
//!
//! ## The executor is the interesting part
//!
//! Unlike the wgpu one, this executor records barriers: the graph's neutral
//! barriers lowered by [`crate::vulkan_barrier_encoder`], per the device's tier.
//! And unlike wgpu, it needs its resources to be the *same* registry the `Gpu`
//! exposes: a handle made through `dyn Resources` has to resolve in the command
//! list, or every pass that uses one fails with `InvalidHandle`.

use std::sync::Arc;

use ash::vk;
use ash::vk::Handle as _;

use chassis::graph::pass::{NoRecording, PassRecorder, SlotTable};
use chassis::{
    BackendChoice, GpuInfo, PassId, RenderGraph as RenderGraphTrait, Resources as ResourcesTrait,
    Result as RhiResult, Swapchain,
};

use crate::vulkan_context::Context;
use crate::vulkan_device::DeviceOptions;
use crate::vulkan_render_graph_executor::{ExecutorResources, VulkanRenderGraphExecutor};
use crate::vulkan_resources::VulkanResources;
use crate::vulkan_swapchain::Swapchain as VulkanSwapchain;

/// Which backend this implementation is.
pub const BACKEND: BackendChoice = BackendChoice::Vulkan;

/// The Vulkan backend, as the RHI sees it.
pub struct VulkanGpu {
    /// **Declaration order is drop order. Do not reorder these.**
    ///
    /// `context` owns the `vk::Device`, and everything below owns objects made
    /// from it. Rust drops fields in declaration order, so `context` must come
    /// *last* -- it is first here only to be corrected, and it was not: with
    /// `context` first the device was destroyed before the recorder freed its
    /// command pools, and every one of the 64 lists leaked.
    ///
    /// It is visible under validation and completely invisible otherwise, which
    /// is the worst combination: a leak that only shows up on a machine that is
    /// set up to catch it. So it is stated here rather than left to be re-broken
    /// by a tidy-up.
    executor: VulkanRenderGraphExecutor,
    resources: VulkanResources,
    swapchain: Option<VulkanSwapchain>,
    graph: GraphAdapter,
    recorder: Option<Arc<dyn PassRecorder>>,
    info: GpuInfo,
    extent: chassis::Extent2d,
    /// The RHI handle for each swapchain image view, minted once and reused.
    ///
    /// `Vec<Option<_>>` rather than `Vec<_>`: a lazily-created view has no
    /// handle until it exists, and a null handle would look like a real view of
    /// image zero to anything that only compares handles.
    view_handles: Vec<Option<chassis::ImageView>>,
    /// Acquire fence, made with the swapchain and reset before every acquire.
    acquire_fence: vk::Fence,
    /// The pass-recording thread pool.
    ///
    /// Held here rather than borrowed from the caller because `Gpu::submit` takes
    /// no jobs argument, and a frame should not have to thread one through every
    /// layer to record itself in parallel.
    jobs: std::sync::OnceLock<Arc<grind::Jobs>>,
    /// Last, because it destroys the device everything above was made from.
    context: Context,
}

impl Drop for VulkanGpu {
    fn drop(&mut self) {
        // Null-guarded because a headless `VulkanGpu` never made one, and
        // `vkDestroyFence` on a null handle is a crash rather than an error worth
        // reporting.
        unsafe {
            if !self.acquire_fence.is_null() {
                let _ = self.context.device.device_wait_idle();
                self.context.device.destroy_fence(self.acquire_fence, None);
            }
        }
    }
}

impl VulkanGpu {
    /// Headless: a device, no surface, no swapchain.
    ///
    /// Takes an existing [`Context`] rather than creating one, so a caller that
    /// already has a device -- a test, a second view of the same GPU -- does not
    /// create a second one.
    pub fn headless(context: Context, extent: chassis::Extent2d) -> RhiResult<Self> {
        let capabilities = &context.capabilities;
        let family = capabilities.graphics_queue_family.ok_or_else(|| {
            chassis::Error::Unsupported("no graphics queue family on this device".to_owned())
        })?;

        let max_lists = 64;
        let recorder = crate::vulkan_frame_context::Recorder::new(
            &context.device,
            context.graphics_queue,
            family,
            capabilities.sync_tier,
            max_lists,
        )
        .map_err(|error| chassis::Error::Unsupported(error.to_string()))?;

        let memory = context.memory_properties();
        // With the upload ring, for the same reason the wgpu backend passes its
        // queue: `bifrost` uploads its 1x1 textures while its plugin is built, and
        // a `Gpu` whose resources cannot upload cannot build the renderer.
        let resources = VulkanResources::new(&context.device, memory)
            .with_uniform_buffer_alignment(capabilities.min_uniform_buffer_offset_alignment)
            .with_upload(
                &context.device,
                &memory,
                context.graphics_queue,
                family,
                1024 * 1024,
            )?;
        let executor = VulkanRenderGraphExecutor::new(recorder, extent);

        let info = GpuInfo {
            name: capabilities.device_name.clone(),
            sync_tier: capabilities.sync_tier,
            max_texture_size: capabilities.max_texture_dimension_2d,
            // One. More needs the recorder to hold two sets of pools and submit to
            // a swapchain image that is not the current one; claiming more here
            // would be a lie that shows up as a validation error about a fence.
            frames_in_flight: 1,
        };

        Ok(Self {
            executor,
            resources,
            swapchain: None,
            graph: GraphAdapter::new(),
            recorder: None,
            info,
            extent,
            view_handles: Vec::new(),
            acquire_fence: vk::Fence::null(),
            jobs: std::sync::OnceLock::new(),
            context,
        })
    }

    /// Headless, creating the device. The constructor a launcher uses.
    pub fn create_headless(
        engine: &'static std::ffi::CStr,
        extent: chassis::Extent2d,
    ) -> RhiResult<Self> {
        let context = crate::vulkan_device::create_device(DeviceOptions::from_env(engine))
            .map_err(|error| chassis::Error::Unsupported(error.to_string()))?;
        Self::headless(context, extent)
    }

    /// With a swapchain, which is what a window gives.
    pub fn with_swapchain(mut self, swapchain: VulkanSwapchain) -> RhiResult<Self> {
        let (width, height) = swapchain.extent();
        self.extent = chassis::Extent2d { width, height };
        self.executor.set_extent(self.extent);
        self.view_handles = vec![None; swapchain.image_count() as usize];

        // Unsigned: a fence starts unsignalled, which is exactly what
        // `acquire` wants before it has acquired anything.
        let fence_info = vk::FenceCreateInfo::default();
        unsafe {
            self.acquire_fence = self
                .context
                .device
                .create_fence(&fence_info, None)
                .map_err(|error| chassis::Error::Unsupported(format!("vkCreateFence: {error}")))?;
        }
        self.swapchain = Some(swapchain);
        Ok(self)
    }

    /// Install the code that records each pass.
    pub fn set_pass_recorder(&mut self, recorder: Arc<dyn PassRecorder>) {
        self.recorder = Some(recorder);
    }

    /// The pool passes record on. Built on first use.
    ///
    /// `OnceLock` rather than a plain field, so *constructing* a `Gpu` -- in a
    /// test, in a tool that only wants a device -- costs no threads. A pool
    /// spawns `available_parallelism() - 1` of them, and a program that makes
    /// four `Gpu`s to compare backends would otherwise have thirty-one threads
    /// for nobody.
    pub fn jobs(&self) -> Arc<grind::Jobs> {
        Arc::clone(self.jobs.get_or_init(|| Arc::new(grind::Jobs::new())))
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn resources(&mut self) -> &mut VulkanResources {
        &mut self.resources
    }

    pub fn executor(&self) -> &VulkanRenderGraphExecutor {
        &self.executor
    }

    pub fn executor_mut(&mut self) -> &mut VulkanRenderGraphExecutor {
        &mut self.executor
    }

    pub fn slots(&self) -> &SlotTable {
        self.executor.slots()
    }

    pub fn slots_mut(&mut self) -> &mut SlotTable {
        self.executor.slots_mut()
    }

    pub fn set_extent(&mut self, extent: chassis::Extent2d) {
        self.extent = extent;
        self.executor.set_extent(extent);
    }
}

/// The registries a command list borrows, for the duration of one frame.
///
/// Takes `&VulkanResources` rather than `&self`. As a method it would borrow all
/// of `self`, which includes `self.executor`, and the executor needs that
/// mutably one line later -- so the two disjoint fields would be fused into one
/// borrow and refused. Taking the registry alone is what lets the borrow checker
/// see that they really are disjoint.
fn executor_resources(resources: &VulkanResources) -> ExecutorResources {
    ExecutorResources {
        pipelines: resources.shared_pipelines(),
        textures: resources.shared_textures(),
        buffers: resources.shared_buffers(),
        descriptors: resources.shared_descriptors(),
    }
}

impl Swapchain for VulkanGpu {
    fn extent(&self) -> chassis::Extent2d {
        self.swapchain.as_ref().map_or(self.extent, |swapchain| {
            let (width, height) = swapchain.extent();
            chassis::Extent2d { width, height }
        })
    }

    fn format(&self) -> chassis::TextureFormat {
        self.swapchain
            .as_ref()
            .map_or(chassis::TextureFormat::Rgba8Unorm, |swapchain| {
                crate::vulkan_types::from_vk_format(swapchain.format())
                    .unwrap_or(chassis::TextureFormat::Rgba8Unorm)
            })
    }

    fn current_image_view(&self) -> chassis::ImageView {
        // `Swapchain` is a trait method, so it cannot create a view -- and it
        // should not: this is a query about the frame in flight, not a place to
        // allocate one. The views are built by `acquire_view` below, which the
        // frame loop calls once it has acquired an image.
        self.acquired_handle()
            .unwrap_or_else(chassis::ImageView::null)
    }

    fn frame_index(&self) -> u32 {
        self.acquired_index().unwrap_or(0)
    }

    fn image_count(&self) -> u32 {
        self.swapchain
            .as_ref()
            .map_or(0, |swapchain| swapchain.image_count())
    }

    fn is_out_of_date(&self) -> bool {
        self.swapchain
            .as_ref()
            .is_some_and(VulkanSwapchain::is_out_of_date)
    }
}

/// Implementation of the three accessors above, which need to read the swapchain.
impl VulkanGpu {
    /// Build and register the view of the acquired image.
    ///
    /// Split out of [`Swapchain::current_image_view`] for the reason in its
    /// comment: creating a `vk::ImageView` is a device call, and a trait getter
    /// that quietly performs one turns every debug print of the current view into
    /// a potential allocation and a potential failure.
    ///
    /// `None` when headless or when nothing has been acquired.
    pub fn acquired_view(&mut self) -> RhiResult<Option<chassis::ImageView>> {
        let Some(index) = self.acquired_index() else {
            return Ok(None);
        };
        let slot = index as usize;
        if self.view_handles.len() <= slot {
            return Err(chassis::Error::InvalidDescriptor(format!(
                "swapchain returned image {index}, but the swapchain has {} images",
                self.view_handles.len()
            )));
        }
        if let Some(handle) = self.view_handles[slot] {
            return Ok(Some(handle));
        }
        let view = self
            .swapchain
            .as_mut()
            .expect("acquired index with no swapchain")
            .acquired_image_view()
            .map_err(|error| chassis::Error::Unsupported(error.to_string()))?
            .ok_or_else(|| {
                chassis::Error::InvalidDescriptor("acquired image has no view".to_owned())
            })?;
        let handle = self.resources.register_image_view(view);
        self.view_handles[slot] = Some(handle);
        Ok(Some(handle))
    }

    pub fn acquired_index(&self) -> Option<u32> {
        self.swapchain
            .as_ref()
            .and_then(VulkanSwapchain::acquired_image_index)
    }

    pub fn acquired_handle(&self) -> Option<chassis::ImageView> {
        self.acquired_index()
            .and_then(|index| self.view_handles.get(index as usize).copied().flatten())
    }

    /// Take the next swapchain image.
    ///
    /// ## Why this is not on `chassis::Swapchain`
    ///
    /// Acquiring is a *frame loop* step that can fail and can block, and it needs
    /// a semaphore and a fence. The RHI's `Swapchain` is a set of queries about a
    /// frame in flight; putting a blocking acquire on it would make `extent()` and
    /// `format()` look as though they might block too.
    ///
    /// ## The fence and semaphore
    ///
    /// One of each, created with the swapchain and destroyed with it. The fence
    /// is created *unsignalled*, which is what `acquire` wants before it has
    /// acquired anything, and waited at `present`.
    pub fn acquire(&mut self, timeout: std::time::Duration) -> RhiResult<()> {
        let Some(swapchain) = self.swapchain.as_mut() else {
            return Err(chassis::Error::InvalidDescriptor(
                "acquire on a gpu with no swapchain".to_owned(),
            ));
        };
        // The fence must be unsignalled before it is submitted, and
        // `present` waits it -- which signals it. So every acquire after the
        // first is submitted in a signalled state, which is
        // `VUID-vkAcquireNextImageKHR-fence-01287`.
        unsafe {
            self.context
                .device
                .reset_fences(&[self.acquire_fence])
                .map_err(|error| chassis::Error::Unsupported(format!("vkResetFences: {error}")))?;
        }

        // No semaphore. `vkAcquireNextImageKHR` allows a null one, and a real
        // one would have to be consumed by a submit before it can be reused --
        // which is the render-finish semaphore work that is not wired yet.
        // Using the fence alone is correct and, with `present` waiting anyway,
        // no slower. Passing a semaphore nothing ever waits on would leave it
        // signalled, and the *next* acquire would be
        // `VUID-vkAcquireNextImageKHR-semaphore-01286`.
        match swapchain.acquire(timeout, self.acquire_fence, vk::Semaphore::null()) {
            Ok(_image) => Ok(()),
            Err(error) => {
                // Out of date is not a failure. The frame is skipped and the
                // swapchain rebuilt, which is the documented path; reporting it as
                // an error would make every window resize look like a crash.
                if swapchain.is_out_of_date() {
                    return Ok(());
                }
                Err(chassis::Error::Unsupported(error.to_string()))
            }
        }
    }

    /// Present the acquired image and release it.
    ///
    /// ## The empty semaphore list, and why it is not a lie
    ///
    /// `vkQueuePresentKHR` takes semaphores to wait on. Render-finish semaphores
    /// are not wired through the pass recorder yet, so this passes none, and
    /// present instead waits the acquire fence below.
    ///
    /// That is correct but slow -- correct because the fence is waited, not
    /// because there is nothing to wait for. An empty list *believed* to be a
    /// finished frame is how this becomes a race later, so the comment names what
    /// is missing rather than describing what is there.
    pub fn present(&mut self) -> RhiResult<()> {
        let Some(swapchain) = self.swapchain.as_mut() else {
            return Err(chassis::Error::InvalidDescriptor(
                "present on a gpu with no swapchain".to_owned(),
            ));
        };
        let queue = self.context.graphics_queue;
        let (format, image) = match (
            crate::vulkan_swapchain::chassis_format(swapchain.format())
                .or(Some(chassis::TextureFormat::Rgba8Unorm)),
            swapchain.acquired_image(),
        ) {
            (Some(format), Some(image)) => (format, image),
            // Nothing acquired: the out-of-date path. Presenting anyway would be
            // presenting an image index nobody holds.
            _ => return Ok(()),
        };

        unsafe {
            let _ = self
                .context
                .device
                .wait_for_fences(&[self.acquire_fence], true, u64::MAX);
        }

        let _ = swapchain;

        // The acquired image has to be in PRESENT_SRC before present, and the
        // render graph cannot do it: a swapchain image is handed out by the
        // driver and no pass ever declares it.
        self.executor
            .transition_to_present(image, format)
            .map_err(|error| chassis::Error::Unsupported(error.to_string()))?;

        let result = self
            .swapchain
            .as_mut()
            .expect("checked above")
            .present(queue, &[])
            .map_err(|error| chassis::Error::Unsupported(error.to_string()));
        // The image is no longer ours whatever present said. `suboptimal` is not
        // an error and leaves the swapchain usable; only the index is cleared.
        self.release_acquired();
        result
    }

    /// Forget the acquired image without presenting it -- the out-of-date path.
    fn release_acquired(&mut self) {
        if let Some(swapchain) = self.swapchain.as_mut() {
            swapchain.release_acquired();
        }
    }
}

impl RenderGraphTrait for GraphAdapter {
    fn reset(&mut self) {
        self.graph.reset();
        self.refusal = None;
    }

    fn add_pass(&mut self, desc: chassis::RenderPassDesc) -> PassId {
        // Refused, and named, with stormglass's own words so the two backends
        // refuse identically. Recorded rather than returned because this
        // signature returns an id, so an error has nowhere to go and `compile` is
        // the next point a caller can be told.
        let name = if desc.label.is_empty() {
            "unnamed"
        } else {
            desc.label.as_str()
        };
        self.refusal = Some(format!(
            "no pass '{name}': the graph is declared through \
             `chassis::graph::RenderGraph`, where a pass names its resources by id \
             and by state, not through `RenderGraph::add_pass`, whose \
             `RenderPassDesc` carries no resource states"
        ));
        NO_PASS
    }

    fn add_dependency(&mut self, _producer: PassId, _consumer: PassId) {
        // Refused, and named. Accepting an edge the compiler ignores would be worse
        // than an error, and a panic is worse than an error the caller can read.
        let why = "`add_dependency` has nowhere to go: ordering comes from declared \
                   reads and writes";
        self.refusal = Some(why.to_owned());
    }

    fn compile(&mut self) -> RhiResult<Vec<PassId>> {
        self.refused()?;
        Ok(self
            .graph
            .passes()
            .iter()
            .enumerate()
            .map(|(index, _)| index as PassId)
            .collect())
    }

    fn execute(&mut self, _jobs: &grind::Jobs) -> RhiResult<()> {
        // Recording is the executor's job and it owns its command lists, which is
        // why `RenderGraph::execute` takes none. This validates the frame.
        self.refused()?;
        self.graph
            .compile()
            .map_err(|errors| chassis::Error::Unsupported(describe(&errors)))?;
        Ok(())
    }
}

/// `chassis::Gpu` for Vulkan.
impl chassis::Gpu for VulkanGpu {
    fn info(&self) -> &GpuInfo {
        &self.info
    }

    fn swapchain(&self) -> Option<&dyn Swapchain> {
        // `self`, not the inner `VulkanSwapchain`. The backend swapchain hands
        // out `vk::Format` and `(u32, u32)` and owns its own acquire; the RHI
        // swapchain hands out `TextureFormat` and `Extent2d` and is a *view* of
        // one. Two types with one job and two names is how a caller ends up
        // asking the wrong one for the extent.
        //
        // Returning `self` works because `&VulkanGpu` coerces to
        // `&dyn Swapchain` -- the same object, both interfaces.
        self.swapchain.as_ref().map(|_| self as &dyn Swapchain)
    }

    fn extent(&self) -> chassis::Extent2d {
        // The `Swapchain` impl below already answers this -- the swapchain's
        // extent, or the stored headless one -- so this only lifts it onto `Gpu`
        // where a frontend holding `&mut dyn Gpu` can reach it.
        Swapchain::extent(self)
    }

    fn resize(&mut self, extent: chassis::Extent2d) -> RhiResult<()> {
        // With a swapchain the surface owns the size, and this backend has no
        // path that rebuilds one: `is_out_of_date` is reported and nothing
        // recreates it. Refused by name rather than ignored, because a resize
        // that silently does nothing is a window that never redraws.
        if self.swapchain.is_some() {
            return Err(chassis::Error::Unsupported(
                "resize: the swapchain owns the size and this backend cannot rebuild it".to_owned(),
            ));
        }

        self.set_extent(extent);
        Ok(())
    }

    fn graph(&mut self) -> &mut dyn RenderGraphTrait {
        &mut self.graph
    }

    fn resources(&mut self) -> &mut dyn ResourcesTrait {
        &mut self.resources
    }

    fn record_frame(
        &mut self,
        frame: &chassis::graph::CompiledGraph,
        slots: &SlotTable,
        recorder: Arc<dyn PassRecorder>,
    ) -> RhiResult<()> {
        // The swapchain's extent when there is one: it is the clamped, configured
        // size, and the frontend sized this frame's targets from the same one.
        self.executor.set_extent(Swapchain::extent(self));

        // The frontend's table is the frame's table. It minted the textures the
        // graph names by slot, so it is the only side that knows which handle is
        // which; replaced rather than merged, because slot indices move when the
        // frontend's frame changes and a stale entry would name a texture that
        // has been destroyed.
        *self.executor.slots_mut() = slots.clone();

        // A swapchain image has no `chassis::Texture` on this backend: the
        // swapchain owns it and nothing registers it in `Textures`, which is what
        // both `resolve_frame` and every barrier resolve a slot through. So the
        // one slot no frontend can fill is also one this backend cannot fill yet,
        // and it is refused here by name -- `resolve_frame` would otherwise fail
        // with `InvalidHandle` against whichever pass attached it.
        if let Some(slot) = backbuffer_slot(frame) {
            if self.executor.slots().texture(slot).is_none() {
                return Err(chassis::Error::Unsupported(format!(
                    "the frame draws the swapchain image (slot {slot}) but nothing is bound to \
                     it: a Vulkan swapchain image has no chassis::Texture to bind"
                )));
            }
        }

        // The recorder is moved into the executor, which owns the command lists
        // and the submit order. Cloned out of the registries first so the mutable
        // borrow of the executor is not live across the immutable one, as
        // `submit` does for the same reason.
        let resources = executor_resources(&self.resources);
        let jobs = self.jobs();
        self.executor
            .execute(frame, resources, recorder, &jobs)
            .map_err(|error| chassis::Error::Unsupported(error.to_string()))?;

        // No present here. The frame loop is `frame_begin` -> this -> `submit`,
        // and `submit` owns the present: one per frame, at the point every caller
        // reaches through `Box<dyn Gpu>`.
        Ok(())
    }

    fn frame_begin(&mut self) -> RhiResult<()> {
        self.graph.reset();
        Ok(())
    }

    fn submit(&mut self) -> RhiResult<()> {
        // A refused declaration submits nothing on purpose: the frame the caller
        // described was never declared, so running it would record a frame from
        // the declaration path this backend does not have and call it done.
        self.graph.refused()?;

        let compiled = self
            .graph
            .graph
            .compile()
            .map_err(|errors| chassis::Error::Unsupported(describe(&errors)))?;

        // The recorder is cloned out first so the mutable borrow of the executor
        // is not live across the immutable one, which the borrow checker is right
        // to refuse.
        let recorder = self.recorder.clone();
        let resources = executor_resources(&self.resources);
        // Cloned so the executor can hand it to `grind::parallel_for`, whose
        // closure is `'static` and therefore cannot borrow this one.
        let jobs = self.jobs();
        let result = match recorder {
            Some(recorder) => self.executor.execute(&compiled, resources, recorder, &jobs),
            None => self
                .executor
                .execute(&compiled, resources, Arc::new(NoRecording), &jobs),
        };
        result.map_err(|error| chassis::Error::Unsupported(error.to_string()))?;

        // The trait's own sentence for `submit` is "submit every list produced this
        // frame, then present", and this is the only place a caller gets the second
        // half: `App::frame` runs `frame_begin` -> the plugins' `on_render` (where
        // `record_frame` records and submits the frontend's frame) -> here, so the
        // work this presents is this frame's. `present` is inherent on this backend
        // and the engine holds a `Box<dyn Gpu>`.
        //
        // A `submit` with no frontend frame attached -- this backend's own graph,
        // which nothing can declare into -- presents whatever the acquire handed
        // out, and nothing wrote it. That is the caller's frame to skip, not a
        // reason to stop presenting: a present with nothing acquired is a no-op
        // (`present` returns before the queue call), and headless there is no
        // swapchain to present at all.
        if self.swapchain.is_some() {
            self.present()?;
        }
        Ok(())
    }

    fn wait_idle(&mut self) {
        unsafe {
            let _ = self.context.device.device_wait_idle();
        }
    }
}

/// Adapts the fluent graph to the `RenderGraph` trait.
///
/// `add_pass` and `add_dependency` are the two methods that refuse -- see the
/// comments on them for why, and `refused` for where the refusal becomes an
/// error.
pub struct GraphAdapter {
    graph: chassis::graph::RenderGraph,
    /// Why a pass or an edge was refused, if one was.
    ///
    /// Kept because the two methods that can refuse have no return value to put an
    /// error in; `refused` is where it becomes one.
    refusal: Option<String>,
}

impl GraphAdapter {
    pub fn new() -> Self {
        Self {
            graph: chassis::graph::RenderGraph::new(),
            refusal: None,
        }
    }

    pub fn graph(&mut self) -> &mut chassis::graph::RenderGraph {
        &mut self.graph
    }

    /// Reports a refusal recorded by `add_pass` or `add_dependency`.
    ///
    /// Called by `compile`, `execute` and `submit`, which are the three points a
    /// caller can be told. `reset` clears it, so a refused frame cannot poison the
    /// next one.
    fn refused(&self) -> RhiResult<()> {
        match &self.refusal {
            Some(why) => Err(chassis::Error::Unsupported(why.clone())),
            None => Ok(()),
        }
    }
}

impl Default for GraphAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// The id `add_pass` returns when it refuses.
///
/// Not a declaration index: nothing was declared, and `compile` reports the same
/// refusal, so there is no pass to look up by it. `u32::MAX` is the graph's own
/// "no resource" sentinel (`chassis::graph::types::NO_RESOURCE`).
const NO_PASS: PassId = u32::MAX;

/// The slot a frame draws the swapchain image into, if it draws one at all.
///
/// The swapchain image is the one resource a frame claims is external and has to
/// leave presentable: the renderer imports it with `Undefined` in and `Present`
/// out, and the compiler carries that declaration into the pass list as the
/// epilogue transition to `Present` on the last pass that touches it. So the
/// compiled frame names the slot on its own -- which matters, because a compiled
/// frame is all a backend is handed.
///
/// Nothing else in a frame reaches `Present`: every other texture ends at
/// whatever state the pass that last touched it left it in. The wgpu backend has
/// the same rule, in `stormglass::wgpu_render_graph_executor::backbuffer_slot`.
fn backbuffer_slot(frame: &chassis::graph::CompiledGraph) -> Option<chassis::graph::TextureId> {
    frame
        .passes
        .iter()
        .flat_map(|pass| pass.pre_barriers.iter().chain(&pass.post_barriers))
        .find(|barrier| barrier.dst_state == chassis::graph::ImageState::Present)
        .map(|barrier| barrier.texture)
}

/// Formats the graph's validation errors into one message.
///
/// The graph returns a `Vec` because it collects every error rather than stopping
/// at the first -- which is right, so they must be joined rather than the first
/// one dropped.
fn describe(errors: &[chassis::graph::types::ValidationError]) -> String {
    errors
        .iter()
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}
