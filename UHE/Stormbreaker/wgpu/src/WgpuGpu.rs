//! `chassis::Gpu` for wgpu, the first implementation of it in this project.
//!
//! ## Why this file matters more than its size
//!
//! Until now `chassis::Gpu` had **zero** implementations. The trait existed to
//! describe a seam, and a seam described and never implemented is a hope rather
//! than an architecture. This is the first thing on the other side of it.
//!
//! Everything above it (the renderer, the editor, whatever comes after) reaches
//! the GPU through `dyn Gpu` and cannot tell this from the Vulkan one. When
//! `VulkanGpu` exists, the two are chosen at startup and nothing above changes.
//! That is the whole claim of a two-backend engine, and until this file compiled
//! it was untested.
//!
//! ## The render graph adapter, and its two refusals
//!
//! `chassis::RenderGraph` wants `add_pass`, `add_dependency`, `compile` and
//! `execute`. The graph in `chassis::graph` wants a fluent builder and derives
//! ordering from declared reads and writes. Those are two different authoring
//! styles, and the adapter below bridges only the half that carries the same
//! information: neither `add_pass` nor `add_dependency` can be honoured.
//!
//! Both refusals become a `chassis::Error::Unsupported`: neither method has room
//! for one in its own signature (`add_pass` returns an id, `add_dependency`
//! returns nothing), so a refusal is recorded and named by `compile` -- and by
//! `submit`, so a refused frame is not submitted as an empty one and presented.
//!
//! `add_dependency` first: the shared graph has no explicit edge list. Ordering
//! comes from reads and writes, which is what makes a read-after-write hazard
//! unrepresentable rather than merely discouraged. Accepting an edge the compiler
//! would ignore would be worse than refusing it, a caller would believe something
//! the graph does not do.
//!
//! `add_pass` cannot be honoured faithfully either. A pass in the shared graph
//! names its resources by id and by *state* (`ImageState`), which is what the
//! compiler orders a frame from; a `RenderPassDesc` carries attachments as handles
//! with a format and a load op and no state at all, its label may be empty (which
//! the compiler refuses as an unnamed pass), and a resolve target has nowhere to
//! go. A pass declared from one would be ordered against states the caller never
//! gave: a wrong frame, which is worse than a missing one.

use std::sync::Arc;

use chassis::graph::pass::{NoRecording, PassRecorder, SlotTable};
use chassis::{
    BackendChoice, GpuInfo, PassId, RenderGraph as RenderGraphTrait, Resources as ResourcesTrait,
    Result as RhiResult, Swapchain, SyncTier,
};

use crate::wgpu_device::{Adapter, Device, Queue};
use crate::wgpu_error::WgpuError;
use crate::wgpu_render_graph_executor::WgpuRenderGraphExecutor as WgpuExecutor;
use crate::wgpu_resources::WgpuResources;
use crate::wgpu_surface::WgpuSurface;

/// The wgpu backend, as the RHI sees it.
pub struct WgpuGpu {
    device: Device,
    queue: Queue,
    adapter_index: u32,
    executor: WgpuExecutor,
    graph: GraphAdapter,
    info: GpuInfo,
    extent: chassis::Extent2d,
    /// The windowed path. `Some` once [`WgpuGpu::with_surface`] has run, and the
    /// same field [`chassis::Gpu::swapchain`] reports -- a stored swapchain that
    /// `swapchain()` did not report was a real bug in the Vulkan backend
    /// (`docs/WORK.md`), and one field is what stops it happening twice.
    surface: Option<WgpuSurface>,
    /// The pass recorder a frame is driven by. `None` until one is installed,
    /// which is not an error: a frame with no frontend attached still submits.
    recorder: Option<Arc<dyn PassRecorder>>,
}

impl WgpuGpu {
    /// A headless device.
    ///
    /// The default construction: no window, no surface, no swapchain. A `Gpu`
    /// without a window is an ordinary thing here, and it is what the test suite
    /// and any server-side frame want.
    pub fn headless(extent: chassis::Extent2d) -> RhiResult<Self> {
        Self::with_adapter(Adapter::request(0), extent)
    }

    /// A device from a chosen adapter, or the first that works.
    pub fn with_adapter(adapter: Option<Adapter>, extent: chassis::Extent2d) -> RhiResult<Self> {
        let Some(adapter) = adapter.or_else(|| Adapter::request(0)) else {
            return Err(chassis::Error::Unsupported(
                "no wgpu adapter on this machine".to_owned(),
            ));
        };
        let adapter_index = adapter.index();
        let Some((device, queue)) = adapter.create_device("stormglass") else {
            return Err(chassis::Error::Unsupported(format!(
                "wgpu could not create a device on adapter {adapter_index}"
            )));
        };
        Ok(Self::from_device(device, queue, adapter_index, extent))
    }

    /// A device with a window, which is what a windowed run wants.
    ///
    /// ## Why the surface is made by the caller
    ///
    /// It comes from a window, and the crate that owns windows (`eyes`) re-exports
    /// `winit::Window` so that `Instance::create_surface(window)` can happen outside
    /// this crate. Nothing here names a windowing library, and the same code serves
    /// every platform wgpu does.
    ///
    /// ## Why the instance is a parameter
    ///
    /// The adapter is requested *with* the surface attached, because an adapter that
    /// cannot present to the window is not one this backend can use -- and it has to
    /// be asked on the instance that created the surface, since a surface is only
    /// meaningful to its own instance. One value, three uses.
    pub fn with_surface(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        extent: chassis::Extent2d,
    ) -> Result<Self, WgpuError> {
        let adapter =
            Adapter::request_for_surface(0, instance, &surface).ok_or(WgpuError::NoAdapter)?;
        let adapter_index = adapter.index();
        let Some((device, queue)) = adapter.create_device("stormglass") else {
            return Err(WgpuError::NoDevice);
        };
        // The surface is built before the gpu so its chosen extent -- clamped, and
        // chosen from what the surface actually offers -- is the one the frame is
        // sized to, rather than the one the window asked for.
        let swapchain = WgpuSurface::new(surface, adapter, device.clone(), queue.clone(), extent)?;
        let mut gpu = Self::from_device(device, queue, adapter_index, swapchain.extent());
        gpu.surface = Some(swapchain);
        Ok(gpu)
    }

    /// Everything a `Gpu` is once a device exists.
    ///
    /// Split out because both constructors need it and neither should have a second
    /// copy of it: the registry the executor owns, the info a caller reads, and the
    /// extent the frame is recorded at have to be the same on both paths.
    fn from_device(
        device: Device,
        queue: Queue,
        adapter_index: u32,
        extent: chassis::Extent2d,
    ) -> Self {
        let name = device.adapter_name().to_owned();
        let max_texture_size = device.max_texture_size();

        // One registry, owned by the executor. Two is a bug: a handle created
        // through `dyn Resources` would not resolve inside it.
        //
        // With the queue, because `upload_texture` is `queue.write_texture` and
        // `bifrost` uploads its 1x1 textures while its plugin is built. Built
        // without one, the engine's own `Gpu` cannot build the renderer at all.
        let executor = WgpuExecutor::new(
            queue.clone(),
            WgpuResources::with_queue(&device, Some(queue.inner().clone())),
            extent,
        );

        let info = GpuInfo {
            name,
            sync_tier: SyncTier::Sync2Timeline,
            max_texture_size,
            // One. wgpu orders submissions to a queue itself and gives back a
            // promise per submit, so more than one frame in flight is a real
            // option here, and it is not wired. Claiming 2 here would be the
            // same "one less than the truth" that the Vulkan backend's
            // `frames_in_flight: 1` deliberately is not.
            frames_in_flight: 1,
        };

        Self {
            device,
            queue,
            adapter_index,
            executor,
            graph: GraphAdapter::new(),
            info,
            extent,
            surface: None,
            recorder: None,
        }
    }

    /// The attached surface, or `None` headless.
    ///
    /// The RHI reaches it through `Gpu::swapchain`; this is the concrete one, for a
    /// caller that wants the frame's `Texture` handle or the present mode.
    pub fn surface(&self) -> Option<&WgpuSurface> {
        self.surface.as_ref()
    }

    /// Rebuild the swapchain for a new size.
    ///
    /// Headless this only records the extent: there is no swapchain to rebuild, and
    /// the executor needs the size for its viewports.
    pub fn resize(&mut self, extent: chassis::Extent2d) -> Result<(), WgpuError> {
        let configured = match self.surface.as_mut() {
            Some(surface) => {
                surface.reconfigure(extent)?;
                Some(surface.extent())
            }
            None => None,
        };
        self.set_extent(configured.unwrap_or(extent));
        Ok(())
    }

    /// Present the frame's image.
    ///
    /// A no-op when nothing was acquired -- the acquire timed out, or the window is
    /// not visible -- because presenting an image nobody holds is presenting an
    /// index that means nothing.
    pub fn present(&mut self) -> Result<(), WgpuError> {
        let Some(surface) = self.surface.as_mut() else {
            return Err(WgpuError::NoSwapchain);
        };
        surface.present(self.executor.resources_mut())
    }

    /// Install the code that records each pass.
    ///
    /// `Arc`, because a backend may call it from worker threads and a closure
    /// capturing the renderer cannot be cloned otherwise.
    pub fn set_pass_recorder(&mut self, recorder: Arc<dyn PassRecorder>) {
        self.recorder = Some(recorder);
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter_index(&self) -> u32 {
        self.adapter_index
    }

    /// The one registry, through the executor that owns it.
    pub fn resources(&mut self) -> &mut WgpuResources {
        self.executor.resources_mut()
    }

    pub fn executor(&self) -> &WgpuExecutor {
        &self.executor
    }

    /// The graph slot table the executor records from.
    ///
    /// A read of it alongside a write needs both, which `&mut self` cannot give,
    /// so the read takes a temporary: `gpu.slots()` is for assertions and
    /// debugging, not for the frame loop.
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

impl Swapchain for WgpuGpu {
    fn extent(&self) -> chassis::Extent2d {
        // The swapchain's extent when there is one: it is the clamped, configured
        // size, and the window's own is what it was chosen from.
        match self.surface.as_ref() {
            Some(surface) => surface.extent(),
            None => self.extent,
        }
    }

    fn format(&self) -> chassis::TextureFormat {
        // `Rgba8Unorm` with no surface, which is what the Vulkan backend reports
        // without one too: nothing chose a format, and `Gpu::swapchain` is `None`,
        // so the answer is only ever read by a caller that ignored it.
        match self.surface.as_ref() {
            Some(surface) => surface.format(),
            None => chassis::TextureFormat::Rgba8Unorm,
        }
    }

    fn current_image_view(&self) -> chassis::ImageView {
        match self.surface.as_ref() {
            Some(surface) => surface.image_view(),
            None => chassis::ImageView::null(),
        }
    }

    fn frame_index(&self) -> u32 {
        // See `WgpuSurface`: wgpu hands out a texture, not an index.
        0
    }

    fn image_count(&self) -> u32 {
        // Zero headless. `Gpu::swapchain` is `None` then, so nothing can ask --
        // and zero rather than one because one would be a claim about an image
        // count this backend does not have.
        match self.surface.as_ref() {
            Some(surface) => surface.image_count(),
            None => 0,
        }
    }

    fn is_out_of_date(&self) -> bool {
        match self.surface.as_ref() {
            Some(surface) => surface.is_out_of_date(),
            None => false,
        }
    }
}

impl RenderGraphTrait for GraphAdapter {
    fn reset(&mut self) {
        self.graph.reset();
        self.refusal = None;
    }

    fn add_pass(&mut self, desc: chassis::RenderPassDesc) -> PassId {
        // Refused, and named -- see the file header for why a `RenderPassDesc`
        // cannot be declared into the shared graph. Recorded rather than returned:
        // this signature returns an id, so an error has nowhere to go, and
        // `compile` is the next point a caller can be told.
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

    fn execute(&mut self, jobs: &grind::Jobs) -> RhiResult<()> {
        // A no-op, deliberately: recording is the backend executor's job and it
        // owns its command lists, which is why `execute` no longer takes any.
        // `RenderGraph::execute` exists so a caller can order *and* run a frame
        // through one trait object; the two halves live in `submit` and here, and
        // this validates that the frame is sound before either.
        self.refused()?;
        let compiled = self
            .graph
            .compile()
            .map_err(|errors| chassis::Error::Unsupported(describe(&errors)))?;
        let _ = (jobs, compiled);
        Ok(())
    }
}

/// `chassis::Gpu` for wgpu.
impl chassis::Gpu for WgpuGpu {
    fn info(&self) -> &GpuInfo {
        &self.info
    }

    fn swapchain(&self) -> Option<&dyn Swapchain> {
        // `Some` once a surface is attached, `None` headless -- the honest answer
        // the option exists for, and the one the Vulkan backend had to learn:
        // storing a swapchain and reporting `None` meant attaching a window changed
        // nothing observable (`docs/WORK.md`).
        self.surface
            .as_ref()
            .map(|surface| surface as &dyn Swapchain)
    }

    fn extent(&self) -> chassis::Extent2d {
        // The `Swapchain` impl above already answers this -- the surface's
        // extent, or the stored headless one -- so this only lifts it onto `Gpu`
        // where a frontend holding `&mut dyn Gpu` can reach it.
        chassis::Swapchain::extent(self)
    }

    fn resize(&mut self, extent: chassis::Extent2d) -> RhiResult<()> {
        // Reconfigures the surface when there is one, and records the extent the
        // executor sets its viewports from when there is not.
        WgpuGpu::resize(self, extent).map_err(chassis::Error::from)
    }

    fn graph(&mut self) -> &mut dyn RenderGraphTrait {
        &mut self.graph
    }

    fn resources(&mut self) -> &mut dyn ResourcesTrait {
        self.executor.resources_mut()
    }

    fn frame_begin(&mut self) -> RhiResult<()> {
        self.graph.reset();
        if let Some(surface) = self.surface.as_mut() {
            let status = surface.acquire(self.executor.resources_mut())?;
            // Acquired, skipped, or out of date: all three are ordinary, and an
            // out-of-date one has already rebuilt the configuration once inside
            // `acquire`. What is left -- a lost surface, a validation error -- is
            // not something recording another frame fixes.
            if !status.acquired() && !status.skips_frame() && !status.needs_reconfigure() {
                return Err(chassis::Error::Unsupported(status.to_string()));
            }
        }
        Ok(())
    }

    fn submit(&mut self) -> RhiResult<()> {
        // A refused declaration submits nothing on purpose: the frame the caller
        // described was never declared, so running it would present a cleared image
        // and call it a frame.
        self.graph.refused()?;

        let compiled = self
            .graph
            .graph
            .compile()
            .map_err(|errors| chassis::Error::Unsupported(describe(&errors)))?;

        // The image this frame draws into, if it has one: `None` headless, and
        // `None` when the acquire was skipped. The frame's swapchain slot is bound
        // from it here, before a single pass records.
        let acquired = self.surface.as_ref().and_then(WgpuSurface::texture);
        self.executor.resolve_backbuffer(&compiled, acquired)?;

        // No frontend attached is not an error: a frame that only does transfers
        // still submits, and `NoRecording` is the honest recorder for it.
        let recorder: Arc<dyn PassRecorder> = match self.recorder.clone() {
            Some(recorder) => recorder,
            None => Arc::new(NoRecording),
        };
        self.executor.execute(&compiled, recorder.as_ref())?;

        // The trait's own sentence for `submit` is "submit every list produced this
        // frame, then present", and this is the only place a caller gets the second
        // half: the engine's frame holds `Box<dyn Gpu>`, and `present` is inherent
        // on this backend. Headless there is nothing to present, which is not a
        // failure.
        if self.surface.is_some() {
            self.present()?;
        }
        Ok(())
    }

    /// Execute a frame the frontend compiled.
    ///
    /// [`WgpuGpu::submit`] runs the *backend's* graph; this runs the frame a
    /// frontend declared and compiled itself, which is the only shape a renderer
    /// that sees `&mut dyn Gpu` can produce. The slot table comes with it because
    /// the frontend minted the textures the graph names by slot, so it is the only
    /// side that knows which handle is which -- and binding the one slot it cannot
    /// fill, the swapchain image, is this backend's job.
    fn record_frame(
        &mut self,
        frame: &chassis::graph::CompiledGraph,
        slots: &SlotTable,
        recorder: Arc<dyn PassRecorder>,
    ) -> RhiResult<()> {
        // The swapchain's extent when there is one: it is the clamped, configured
        // size, and the frontend sized this frame's targets from the same one.
        let extent = chassis::Swapchain::extent(self);
        self.executor.set_extent(extent);

        // The frontend's table is the frame's table. Replaced rather than merged:
        // slot indices move when the frame changes, so a stale entry would name a
        // texture that has been destroyed.
        *self.executor.slots_mut() = slots.clone();

        // The one slot no frontend can fill. A swapchain image has no `Texture`
        // handle until this backend mints one for the image it acquired, so a frame
        // that draws it with no image is refused here, by slot -- rather than at
        // `begin_rendering`, where the message would name a pass and not the image
        // that was never bound.
        let acquired = self.surface.as_ref().and_then(WgpuSurface::texture);
        self.executor.resolve_backbuffer(frame, acquired)?;

        self.executor.execute(frame, recorder.as_ref())?;

        // No present here, deliberately: `submit` is the frame loop's last call and
        // the single present site, and this frame is already recorded onto the
        // queue, so `submit` presents it. Presenting here too would present twice
        // per frame -- once for this frame and once for the backend's own graph.
        Ok(())
    }

    fn wait_idle(&mut self) {
        let _ = self.executor.wait_idle();
    }
}

/// Which backend this implementation is. So a launcher can print what it got
/// without hard-coding a string next to the match that chose it.
pub const BACKEND: BackendChoice = BackendChoice::Wgpu;

/// Formats the graph's validation errors into one message.
///
/// The graph returns a `Vec` because it collects every error rather than stopping
/// at the first, which is right -- so they must be joined rather than the first
/// one dropped.
fn describe(errors: &[chassis::graph::types::ValidationError]) -> String {
    errors
        .iter()
        .map(|error| error.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

/// The id `add_pass` returns when it refuses.
///
/// Not a declaration index: nothing was declared, and `compile` reports the same
/// refusal, so there is no pass to look up by it. `u32::MAX` is the graph's own
/// "no resource" sentinel (`chassis::graph::types::NO_RESOURCE`).
const NO_PASS: PassId = u32::MAX;

/// Adapts the fluent graph to the `RenderGraph` trait.
///
/// See the file header for `add_pass` and `add_dependency`, the two methods that
/// refuse.
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
