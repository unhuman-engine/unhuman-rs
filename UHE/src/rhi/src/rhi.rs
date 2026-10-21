use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::desc::{
    BufferDesc, ComputePipelineDesc, DescriptorSetLayoutDesc, PipelineDesc, RenderPassDesc,
    SamplerDesc, TextureDesc,
};
use crate::format::TextureFormat;
use crate::handle::{BindGroup, Buffer, ImageView, Pipeline, Sampler, Shader, Texture};

/// Error type for every fallible RHI operation.
///
/// One type at the boundary rather than one per operation, because callers at
/// this layer treat a failed pipeline creation and a failed submit the same way:
/// log it with context and tear the frame down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The backend could not satisfy the request on this device.
    Unsupported(String),
    /// A handle did not resolve: wrong generation, or never created.
    InvalidHandle,
    /// A descriptor that fails backend validation.
    InvalidDescriptor(String),
    /// Device lost, or a surface/out-of-date condition.
    DeviceLost,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::InvalidHandle => write!(f, "invalid handle"),
            Error::InvalidDescriptor(why) => write!(f, "invalid descriptor: {why}"),
            Error::DeviceLost => write!(f, "device lost"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Extent2d {
    pub width: u32,
    pub height: u32,
}

/// Which submit/barrier path the device supports.
///
/// Every barrier and submit goes through one lowering point that picks the
/// modern or legacy encoding from the tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SyncTier {
    /// Plain `vkCmdPipelineBarrier` + `vkQueueSubmit`.
    #[default]
    Legacy,
    /// `vkCmdPipelineBarrier2` + `vkQueueSubmit2` (Vulkan 1.3).
    Sync2,
    /// As above, plus timeline semaphores.
    Sync2Timeline,
}

impl SyncTier {
    /// True for the two tiers that spell barriers `vkCmdPipelineBarrier2`.
    ///
    /// The question callers actually ask. `Sync2` and `Sync2Timeline` differ in
    /// semaphore type, not in barrier spelling, so anything that lowers a
    /// barrier should ask this rather than match on a tier name -- matching the
    /// name is how the next tier gets added and one branch silently stops
    /// compiling.
    pub fn uses_sync2(self) -> bool {
        matches!(self, SyncTier::Sync2 | SyncTier::Sync2Timeline)
    }

    pub fn supports_sync2(self) -> bool {
        !matches!(self, SyncTier::Legacy)
    }

    pub fn supports_timeline_semaphores(self) -> bool {
        matches!(self, SyncTier::Sync2Timeline)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub sync_tier: SyncTier,
    pub max_texture_size: u32,
    /// Number of frames the backend allows in flight.
    pub frames_in_flight: u32,
}

/// One pass's GPU time from the last frame a backend collected.
///
/// The name is owned because the graph rebuilds its pass names every frame;
/// a borrowed name would either leak or dangle before the panel reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuPassTime {
    pub name: String,
    pub ms: f32,
}

/// The swapchain, readable without mutable access so passes can query extent.
pub trait Swapchain {
    fn extent(&self) -> Extent2d;
    fn format(&self) -> TextureFormat;
    /// Acquired for the current frame. Valid between `Gpu::frame_begin` and
    /// `Gpu::submit`.
    fn current_image_view(&self) -> ImageView;
    /// Index of the acquired image.
    fn frame_index(&self) -> u32;
    /// How many images the swapchain has.
    ///
    /// A frame loop needs this and it is not derivable: one image means the CPU
    /// waits for the GPU every frame, two means it never does. A backend with no
    /// swapchain returns `None` from `Gpu::swapchain` instead of reporting `0`,
    /// so a `0` here would mean a real swapchain with no images, which cannot
    /// happen.
    fn image_count(&self) -> u32;
    fn is_out_of_date(&self) -> bool;
}

/// A secondary command list.
///
/// **No barrier methods, by design.** Barriers come from the render graph,
/// derived from the declared read/write sets. The moment a pass can hand-write
/// a barrier, the graph can no longer prove the frame is correct.
pub trait CommandList {
    fn begin_rendering(&mut self, desc: &RenderPassDesc);
    fn end_rendering(&mut self);

    /// Opens a compute scope.
    ///
    /// Separate from [`Self::begin_rendering`] because a compute pass has no
    /// attachments: wgpu needs its own scope before a compute pipeline or bind
    /// group can be set, while Vulkan has no scope at all. The executor opens it
    /// for a pass whose kind is compute, so a recorder does not have to know
    /// which backend it is recording into.
    fn begin_compute(&mut self);
    fn end_compute(&mut self);

    fn set_viewport(&mut self, extent: Extent2d);
    fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32);

    fn bind_pipeline(&mut self, pipeline: &Pipeline);
    /// Binds a set at `set`, against the *bound pipeline's* layout.
    ///
    /// A pipeline must be bound first. Vulkan takes the pipeline layout as an
    /// argument to `vkCmdBindDescriptorSets` and has none before a pipeline is
    /// bound; wgpu ignores it, so a pass that binds a set first works there and
    /// fails here.
    fn bind_group(&mut self, set: u32, group: &BindGroup);

    /// Sets push constants, or reports that this backend cannot.
    ///
    /// wgpu has no push constants. Vulkan does, but this RHI has no declaration
    /// for a pipeline's push-constant range, so a `vkCmdPushConstants` against
    /// one of its layouts is invalid. Both backends therefore return
    /// [`Error::Unsupported`]; use a small uniform buffer in a bind group
    /// instead. The `Result` is the point: the previous signature returned
    /// nothing, so wgpu could only panic.
    fn set_push_constants(&mut self, data: &[u8]) -> Result<()>;

    fn bind_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: u64);
    fn bind_index_buffer(&mut self, buffer: &Buffer, offset: u64);

    fn draw(&mut self, vertices: u32, instances: u32);
    fn draw_indexed(&mut self, indices: u32, instances: u32);
    fn dispatch(&mut self, groups_x: u32, groups_y: u32, groups_z: u32);

    fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer);

    /// Copy a texture's whole mip 0 into a buffer, for reading it back.
    ///
    /// Part of the RHI rather than a backend extra because a debug view, a
    /// screenshot and a GPU test all need it, and none of them should have to
    /// reach past the trait to get it. The graph orders it: declaring a pass
    /// that reads the texture and writes the buffer emits the image-to-buffer
    /// transition, which is the thing that has to be right.
    fn copy_texture_to_buffer(&mut self, src: &Texture, dst: &Buffer);
}

/// Resource creation and destruction. Separate from `Gpu` so a backend can make
/// creation thread-safe while frame recording stays single-owner.
pub trait Resources {
    fn create_buffer(&mut self, desc: BufferDesc, initial: Option<&[u8]>) -> Result<Buffer>;

    /// Uploads `data` into `buffer` at `offset`, replacing what was there.
    ///
    /// The handle stays valid, and so does every bind group that names it, which
    /// is the point: a per-frame arena is refilled once instead of re-created
    /// along with every set that binds it. The buffer must have been created with
    /// [`crate::format::BufferUsage::TRANSFER_DST`] (wgpu's `COPY_DST`), and the
    /// write must stay inside it.
    fn write_buffer(&mut self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()>;

    fn create_texture(&mut self, desc: TextureDesc) -> Result<Texture>;
    fn create_image_view(&mut self, texture: &Texture) -> Result<ImageView>;
    /// `spirv` is the output of `forge`; the module is cached by hash so a
    /// rebuild with identical source costs nothing.
    fn create_shader(&mut self, label: &str, spirv: &[u8]) -> Result<Shader>;
    fn create_pipeline(&mut self, desc: PipelineDesc) -> Result<Pipeline>;
    /// A compute pipeline, from the one description both backends take.
    ///
    /// The shader is a single module: a compute pipeline has one stage, unlike a
    /// graphics pipeline's vertex/fragment pair.
    fn create_compute_pipeline(&mut self, desc: ComputePipelineDesc) -> Result<Pipeline>;
    /// A bind group: a set of resources a pipeline can read, written once and
    /// bound many times.
    ///
    /// Takes the [`DescriptorSetLayoutDesc`] the pipeline declares, so both are
    /// built from one description and cannot disagree. Entries are keyed by
    /// binding number, since a layout may leave gaps for another resource kind;
    /// images and samplers are separate lists because Vulkan wants two bindings
    /// where wgpu wants one, and `DescriptorKind::CombinedImageSampler` collapses
    /// them.
    fn create_bind_group(
        &mut self,
        layout: &DescriptorSetLayoutDesc,
        buffers: &[(u32, Buffer, u64)],
        images: &[(u32, ImageView)],
        samplers: &[(u32, Sampler)],
    ) -> Result<BindGroup>;

    /// A sampler.
    ///
    /// Its own object rather than parameters to `create_bind_group`, because a
    /// sampler outlives the group it is written into and is shared by every
    /// group that samples the same way.
    fn create_sampler(&mut self, desc: &SamplerDesc) -> Result<Sampler>;

    fn destroy_buffer(&mut self, buffer: Buffer);
    fn destroy_texture(&mut self, texture: Texture);

    /// Puts `data` into a texture.
    ///
    /// Row order is top-to-bottom, the opposite of Vulkan's and wgpu's: one rule
    /// for both backends rather than an option every caller must carry.
    ///
    /// Returns when the copy is *queued*, not finished. Destroying the texture
    /// before the queue drains is `VUID-vkDestroyImage-image-01000`, so a caller
    /// must wait before touching anything the upload wrote.
    fn upload_texture(
        &mut self,
        texture: &Texture,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()>;

    /// The whole buffer's contents, on the CPU.
    ///
    /// The buffer must be readable: created with [`crate::BufferUsage::MAP_READ`]
    /// on wgpu, and in host-visible memory on Vulkan. That is what makes a
    /// rendered image reachable: a pass copies a texture into a readable buffer
    /// with [`CommandList::copy_texture_to_buffer`], and this hands the pixels
    /// back.
    ///
    /// Blocks until the memory can be mapped, so a caller must have waited for
    /// the work that wrote it -- a copy is not visible to the CPU until the
    /// queue has retired it. `readback` here is not a hot path.
    fn read_buffer(&mut self, buffer: &Buffer) -> Result<Vec<u8>>;

    /// The whole buffer's contents, after `submission` has completed.
    ///
    /// This is what makes a readback ring overlap: a frame reads the buffer it
    /// wrote a few submissions ago, waiting only for *that* submission rather
    /// than every submission the queue has since accepted. The default waits for
    /// all, which is always correct but gives the ring nothing.
    fn read_buffer_after(&mut self, buffer: &Buffer, submission: Submission) -> Result<Vec<u8>> {
        let _ = submission;
        self.read_buffer(buffer)
    }

    fn destroy_pipeline(&mut self, pipeline: Pipeline);
    fn destroy_shader(&mut self, shader: Shader);
    fn destroy_bind_group(&mut self, group: BindGroup);
    fn destroy_sampler(&mut self, sampler: Sampler);
}

pub type PassId = u32;

/// One queue submission, opaque here because the token is the backend's.
///
/// A frontend must be able to say "wait for the frame that wrote this buffer"
/// without naming a wgpu `SubmissionIndex` or a Vulkan timeline value. The
/// backend that made the token is the one that reads it back.
#[derive(Clone)]
pub struct Submission(Arc<dyn Any + Send + Sync>);

impl Submission {
    pub fn new<T: Any + Send + Sync>(token: T) -> Self {
        Self(Arc::new(token))
    }

    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}

impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Submission")
    }
}

/// The one owner of synchronization.
pub trait RenderGraph {
    /// Clears the graph for the next frame. The graph is rebuilt and recompiled
    /// every frame; there is deliberately no compile cache.
    fn reset(&mut self);

    fn add_pass(&mut self, desc: RenderPassDesc) -> PassId;
    /// `producer` must complete before `consumer` records.
    fn add_dependency(&mut self, producer: PassId, consumer: PassId);

    /// Topological sort, dead-pass cull, lifetimes, barrier synthesis.
    fn compile(&mut self) -> Result<Vec<PassId>>;

    /// Records every pass into its own command list and submits, in compiled
    /// order.
    ///
    /// The backend owns command-list lifetime: Vulkan needs one pool per list and
    /// wgpu's encoder holds a borrow the caller cannot express, so passing lists
    /// in would impose a requirement neither API has.
    ///
    /// `jobs` is the parallelism a backend *may* use for recording, not a
    /// promise; one backend recording on N threads and another on one is a
    /// legitimate difference.
    fn execute(&mut self, jobs: &grind::Jobs) -> Result<()>;
}

/// The device.
///
/// `gpu` is `None` in headless mode; that is the whole of rule 6 in the
/// architecture doc, and it is why every field the GPU touches is an
/// `Option<Box<dyn Gpu>>` rather than a concrete backend type.
pub trait Gpu {
    fn info(&self) -> &GpuInfo;
    /// The swapchain, or `None` when there is not one.
    ///
    /// A headless run has a real `Gpu` and no swapchain; a bare reference would
    /// force every caller to carry a dummy.
    fn swapchain(&self) -> Option<&dyn Swapchain>;

    /// The extent the executor records a frame at.
    ///
    /// The swapchain's when there is one, the configured headless size
    /// otherwise. A frontend must size its render targets from this: the passes
    /// it declares are recorded at this extent, so a mismatch draws into a
    /// corner of a larger target.
    fn extent(&self) -> Extent2d;

    /// Changes the extent a frame is recorded at.
    ///
    /// Headless this is the size asked for: the editor's viewport panel, which
    /// then renders at its own resolution instead of a fixed one scaled to fit.
    /// With a swapchain the surface owns the size -- the backend reconfigures or
    /// marks the swapchain for recreation -- so what [`Gpu::extent`] reports
    /// afterwards is the size that resulted, not necessarily the one passed.
    ///
    /// A frontend resizes its render targets when `extent()` changes; this call
    /// only changes the device's answer.
    fn resize(&mut self, extent: Extent2d) -> Result<()>;

    fn graph(&mut self) -> &mut dyn RenderGraph;
    fn resources(&mut self) -> &mut dyn Resources;

    /// Starts collecting per-pass GPU timestamps.
    ///
    /// `false` means this backend or queue cannot write timestamps, so the rows
    /// are unavailable rather than zero. Defaults to `false`: a backend that has
    /// not implemented timing must not look like a frame of free passes.
    fn enable_gpu_timing(&mut self) -> bool {
        false
    }

    /// Reads the timestamps of the last submitted frame.
    ///
    /// The caller waits for that submission first. A backend without timing
    /// does nothing here rather than reporting a failure.
    fn collect_gpu_timings(&mut self) -> Result<()> {
        Ok(())
    }

    /// Per-pass times from the last collection, empty when timing is off.
    fn gpu_timings(&self) -> &[GpuPassTime] {
        &[]
    }

    /// Whole-frame GPU time from the last collection, `None` when unavailable.
    fn gpu_frame_ms(&self) -> Option<f32> {
        None
    }

    /// Record and submit one compiled frame from a renderer frontend, in the
    /// graph's compiled order.
    ///
    /// **Not `present`.** The frame loop is `frame_begin` -> `record_frame` ->
    /// [`Gpu::submit`], and `submit` is where the present goes, once per frame.
    ///
    /// `SlotTable` is the frontend's half of the frame: the graph names resources
    /// by slot, the RHI by generational handle, and only the frontend knows which
    /// handle it put at which slot. The backend binds the unbound swapchain slot.
    ///
    /// `Arc`, not `&`, because a backend may record passes on worker threads and
    /// a job pool's closure is `'static`.
    ///
    /// The token is the frame's submission, for a later
    /// [`Resources::read_buffer_after`]: a renderer that copies the frame into a
    /// ring buffer passes the token back when it reads that buffer.
    ///
    /// The default refuses by name rather than recording nothing: a silent no-op
    /// frame is the failure this method exists to remove.
    fn record_frame(
        &mut self,
        _frame: &crate::graph::CompiledGraph,
        _slots: &crate::graph::SlotTable,
        _recorder: std::sync::Arc<dyn crate::graph::PassRecorder>,
    ) -> Result<Submission> {
        Err(Error::Unsupported(
            "this gpu does not execute frontend frames".to_owned(),
        ))
    }

    /// Blocks until the GPU has finished `submission`.
    ///
    /// The token is the one [`Gpu::record_frame`] returned. A frontend uses it
    /// to reuse a ring slot only after the frame that last wrote it retired; a
    /// default no-op would leave that reuse racing the GPU, so there is none.
    fn wait_for_submission(&mut self, submission: Submission) -> Result<()>;

    /// Fence wait, swapchain acquire, deferred-deletion flush.
    fn frame_begin(&mut self) -> Result<()>;
    /// Submit whatever the backend recorded for this frame, then present.
    ///
    /// This is the backend's own graph, not a frontend's: a renderer's frame goes
    /// through [`Gpu::record_frame`], which also presents when a swapchain is
    /// attached. Vulkan can only run an empty `submit`, because nothing can
    /// declare into its graph adapter, so a frontend frame loop calls
    /// `record_frame` and not `submit`.
    fn submit(&mut self) -> Result<()>;
    /// Blocks until the device is idle. Test and shutdown only.
    fn wait_idle(&mut self);
}
