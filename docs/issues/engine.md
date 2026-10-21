# Engine

Renderer, backends, editor, physics, assets, tooling. Part of the queue in
`docs/issues/README.md`.

`docs/WORK.md` is the plan and the phase order; this is the list of what is
actually wrong, with the evidence that says so and the thing that would say it is
fixed.

Priority is about what blocks what, not about difficulty:

- **P0** the frame is wrong, missing or cannot be seen at all
- **P1** correctness, or a claim the repo makes that is not true yet
- **P2** cost, or a capability a phase needs
- **P3** polish, or a decision left open

## Renderer (bifrost)

### P0 - the readback is the frame

At 1998x1135 the engine's frame is 11.8 ms on wgpu, of which the CPU's recording
is **0.4 ms** and the GPU plus readback is **11.2 ms**. The readback copies the
composed image to the CPU, then *waits for the GPU to finish* (`bifrost::render`
-> `record_frame` -> `capture_readback`), so the frame is GPU + CPU with no
overlap instead of the larger of the two. The same engine at 640x360 is 1.4 ms
(699 fps), which is what the GPU's own time looks like.

**Fixed when:** the frame time at 2.3 MP is the GPU's own time -- a readback ring
of N buffers with a fence each, reading the frame from N-1 submissions ago, and
no per-frame wait in the backends.

### P1 - the de-pad copy

`strip_copy_padding` copies the whole frame again on the CPU, every frame, because
the render width is arbitrary and the copy's rows are padded to 256 bytes.

**Fixed when:** the render extent's width is a multiple of 64 texels
(`chassis::copy_row_pitch`), the pitch equals the packed row, and the copy is gone.

### Fixed - the entry point is the module's own

`chassis::spirv::entry_points` reads every `OpEntryPoint` and its `ExecutionModel`,
so the module's own name is authoritative. `create_shader` stores the parsed
entries, and both backends name a pipeline stage from the entry for that stage
instead of a fixed `main`. `forge::wgsl::to_spirv` keeps the names the source
declared and `to_spirv_entry` is gone; the Slang path still emits `main`, which
is now just the name in that module and not a convention the WGSL path copies.

### P2 - a steady-state frame creates GPU objects

`BindSet::rebuild` destroys and recreates a uniform buffer and a bind group per
set, per frame (`passes/uniforms.rs`). `docs/WORK.md` Phase 2 is the fix: a
frame-numbered lifetime with a deferred destruction queue.

**Fixed when:** two consecutive frames with the same scene create nothing.

### P2 - the gizmo is invisible against the grid

Both draw the world's X and Z axes as one-pixel lines, so a selected entity at the
origin has its X and Z arrows lying exactly on the grid's axis lines.

**Fixed when:** the gizmo reads as a handle at the origin -- wider lines, a
different grid colour, or a depth treatment.

### P2 - shadows and SSAO are unverified

Both passes run (the frame shows the grid), but no test renders geometry and
checks a pixel, so "the shadow map has no acne at the cascade seams" and the SSAO
output are claims with nothing behind them.

**Fixed when:** a scene with a mesh is rendered and read back in a test.

## Backends (stormbreaker, stormglass)

### P0 - nothing renders a bifrost frame on a real GPU

`UHE/src/renderer/tests/renderer.rs` runs on `FakeGpu`. The six Vulkan bugs found
by hand (see the `stormbreaker` commit) were invisible to the suite: no test
recorded a real multi-pass frame through a real backend.

**Fixed when:** one frame is recorded through both backends and the readback
images are compared -- `docs/WORK.md` Phase 1's "same frame, both backends, same
image".

### P1 - Vulkan's CPU recording is 20x wgpu's

Measured at 1998x1135: **8.8 ms** of recording on Vulkan against **0.4 ms** on
wgpu, for the same graph. The per-frame work is in `record_frame` and the barrier
encoder.

**Fixed when:** profiled (RenderDoc or Tracy) and in the same order as wgpu's.

### Fixed - Vulkan's GPU work is 3x wgpu's

The 33 ms was wall time, not GPU time. `VkQueryPool` timestamps around every
bifrost pass (`stormbreaker/tests/bench_gpu.rs`, release, 1998x1135, 100 frames)
give Vulkan **7.1 ms** of GPU work:

| pass | GPU ms |
|---|---:|
| shadow | 0.005 |
| opaque | 0.043 |
| ssao | 6.132 |
| editor | 0.186 |
| composite | 0.197 |
| readback | 0.364 |
| **frame** | **7.069** |

The same frame on wgpu is **7.2 ms** whole-frame wall; wgpu has no per-pass
timing hook in this tree, so its column cannot be split without editing
`stormglass`. Since the Vulkan total is already below the wgpu total, no Vulkan
pass is 3x its wgpu counterpart.

The other 36 ms of the old wall was the CPU readback: a `MAP_READ` buffer was
allocated `HOST_VISIBLE | HOST_COHERENT` without `HOST_CACHED`, which RADV maps
uncached on this RENOIR. Reading the 9.1 MB frame ran at **250 MB/s**, 36 ms per
frame. `VulkanBuffer` now prefers `HOST_CACHED` for `MAP_READ` (falling back when
a device has none), which is **4.6 GB/s**, 2.0 ms. Frame wall went from
**43.0 ms to 7.2 ms**, and the GPU frame time is unchanged: the gap was never
GPU.

`ssao` was 6.1 ms, 87% of the GPU frame: 64 depth samples per pixel over 2.3 MP.
It is the same shader and the same RADV driver on both backends, so it was the
pass's budget, not Vulkan state. The profile rules out the listed backend causes:
no MSAA or resolve, `Depth32Float` in both, dynamic rendering throughout, and
shadow, opaque, editor, composite and readback together are under 0.8 ms, so
barriers and layout transitions are not the cost. `opaque` was 0.185 ms before
the readback change and 0.042 ms after; that is clock noise, not a state change.

### Fixed - SSAO runs at half resolution

Occlusion now renders at half the frame extent with 32 rotated kernel samples,
then a separate full-rate pass upsamples it depth-aware:

| extent | before ssao | after ssao | after upsample | after pair | before frame | after frame |
|---|---:|---:|---:|---:|---:|---:|
| 1920x1080 | 5.468 | 0.838 | 0.492 | 1.330 | 6.362 | 2.414 |
| 1998x1135 | 6.132 | 1.035 | 0.552 | 1.587 | 7.091 | 2.808 |

The technique:

- The AO target is `render_extent / 2` in each axis. The 32-sample kernel is
  rotated per pixel by interleaved gradient noise, so the reduced count reads as
  noise rather than as one fixed band pattern.
- `origin_clip` is projected once per pixel; each sample adds a `w = 0` offset
  instead of re-projecting `origin + offset` 32 times.
- The upsample reads the four half-rate texels under a full-rate pixel and keeps
  the one whose reconstructed view-space depth matches the pixel, breaking a tie
  by proximity. That avoids both silhouette halos and the loss of contact
  darkening a bilinear average causes.
- There was no full-rate blur to move; none was added.

`Depth32Float`, radius 0.5, bias 0.025 and power 2.0 are unchanged. The
`shadow_ssao` scene still darkens the contact region: the far wall stays equal
with and without the caster, and the largest drop is 98 -> 87 on Vulkan and
79 -> 73 on wgpu. The old fixed kernel's uniform 1.2% self-occlusion on a flat
wall (251 rather than 255 sRGB) is gone; that was a kernel artifact, and the
contact effect is still visible. `render_parity` is mean 0.138, max 81, against
the target 0.2 / 85.

### P1 - the wgpu swapchain

`WgpuSwapchain` is the parity gap: configure, acquire, present, resize, with a
real `chassis::ImageView` as the current image, and `WgpuGpu::swapchain()`
returning `Some`.

**Fixed when:** a windowed `sandbox` run presents on wgpu and resizes.

### P1 - the Vulkan windowed path cannot resize

`Gpu::resize` refuses with a swapchain ("the surface owns the size and this
backend cannot rebuild it"), and `VulkanSwapChain` reports `is_out_of_date` that
nothing consumes -- there is no recreation path.

**Fixed when:** a windowed Vulkan run resizes and keeps drawing.

### P2 - wgpu's panics are not errors

`WgpuError` is a stub: `create_bind_group` and friends panic inside wgpu instead
of returning `chassis::Error` at the `Resources` boundary.

**Fixed when:** every infallible-looking wgpu call has a pre-check or an error
scope, and the two known panics are gone.

### P2 - compute pipelines, both backends

Compute pipelines and dispatch now work on both backends: one
`chassis::ComputePipelineDesc`, `begin_compute`/`dispatch`/`end_compute` through
`CommandList`, and a real GPU test per backend that reads the written word back.
`DescriptorKind::StorageBufferReadWrite` is separate from the read-only kind
because wgpu validates a layout against the shader's access. Push constants are
an explicit `Error::Unsupported` on both: the RHI has no range declaration, so
Vulkan's call would be invalid, and wgpu has none.

**Still open:** one dispatch list per pass and no indirect dispatch or timestamp
queries; nothing in the renderer declares a compute pass yet, and
`PassType::Transfer` still has no producer. The particle system and the light
culling in `docs/WORK.md` Phases 6 and 8 therefore have a target but no user.

### P2 - wgpu parallel recording

One encoder per pass. Easy on wgpu because there is no command pool to be the
external-synchronisation object, and it is the parity of what Vulkan already does.

### P3 - open decisions

- `VulkanRenderPass`: written, never wired into device or pipeline selection.
- `VulkanBinaryFence` / `VulkanBinarySemaphore`: implement or delete.
- `WgpuResources` / `WgpuPipeline` / `WgpuDevice`: still one file each.

## Editor (UHE-EDITOR)

### P1 - the orbit camera and the keyboard are unverified

Both are written: `CameraInput` from window events, W/E/R, Ctrl+Z, Delete. Neither
has been exercised, because no key event reaches any application in this session
(ydotool's virtual keyboard is registered in Hyprland and neither the editor nor a
terminal received a key). Alt is the gate on every camera control, so the orbit is
unverified too.

**Fixed when:** someone orbits with Alt+drag and presses W/E/R on a real session.

### P1 - no layout persistence, no docking, no tabs

`Layout` is four fixed panels and three splitters. Nothing survives a restart.

**Fixed when:** a rearranged layout comes back after a relaunch.

### P2 - no selection outline

The viewport draws the readback image and the gizmo. A selected entity is visible
only by its gizmo.

### P2 - the content browser cannot drag into the viewport

`drop_files` handles drops from outside the window; dragging a model from the
content browser to the viewport is not wired.

### P2 - undo/redo covers every mutation, or it lies

`mirror`'s `Command` history and the key bindings exist. Every panel mutation has
to go through it -- the inspector and the hierarchy are the ones to audit.

**Fixed when:** a test mutates a property, a name, a reparent and a create, then
undoes all four.

### P3 - the viewport has no accessibility role

gpui logs `focused element ... has an id but no role` for the focusable viewport.

## Physics (femur)

### P1 - three fixes from the port are unverified

- Sleeping bodies never sync back (`!IsActive` in the C++): a body that falls
  asleep must stop writing its transform, not keep the last one.
- `AllowedDOFs` is serialized and never applied.
- `BoxCollider`'s `RuntimeShape` is never written.

**Fixed when:** each has a test that fails without the fix.

## Assets (vault)

### P1 - every load is synchronous

The C++ tree loads everything synchronously and `grind` exists with no loader
using it. A model load blocks the frame that asks for it.

**Fixed when:** a mesh load happens on a worker and the frame that asked for it
does not wait.

### P2 - no mesh optimisation

`optimize_vertex_cache`, `optimize_vertex_fetch` and `optimize_overdraw` are not
there. `meshopt` is a 16-file C++ build, so the port is hand-rolled by design.

### P3 - YAML

`yaml_serde`, not `serde_yaml`: deprecated, and the only YAML option carrying a C
dependency.

## Tooling

### P2 - a skip is not a pass, and the rest of the suite is one grep away

Fixed for the shader suites: nine tests in `stormbreaker` printed
`skipping: the vendored slangc is not available` and returned, so a machine with
no compiler ran them green having compiled nothing. They panic now, naming
`SLANGC`, and `scripts/gate.sh` counts `SKIP:` lines and prints the number.

What is left is the same audit everywhere else: the renderer's own tests run on
`FakeGpu`, so no test would notice a backend that records nothing (see the P0
above), and 44 device guards still skip silently on a machine with no driver.

**Fixed when:** every suite's skip count is zero on a machine that has what the
suite needs, and the gate fails when it is not.

### P3 - RenderDoc has no in-app trigger

Installed (v1.45, Vulkan). A capture needs its GUI, the layer's hotkey, or ~60
lines around `RENDERDOC_GetAPI` behind an env var. The last is the only one that
works without a keyboard or a display.

### P3 - Tracy is not installed

`tracy-client` is the Rust side; the profiler server that receives the stream has
to be built from source, and nothing on this machine can do it today.
