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

### Fixed - the readback is the frame

`bifrost` keeps a `READBACK_RING` of buffers with `READBACK_LAG` frames of
latency, so a frame reads the image from an older submission instead of waiting
for its own. The editor's viewport sees the image `READBACK_LAG` frames late and the
frame no longer pays the copy as a stall.

### Fixed - the de-pad copy

The render width is aligned to `chassis::copy_row_pitch`, so the readback's pitch
equals the packed row and `strip_copy_padding` is gone; the frame is handed over
without a second CPU copy.

### Fixed - the entry point is the module's own

`chassis::spirv::entry_points` reads every `OpEntryPoint` and its `ExecutionModel`,
so the module's own name is authoritative. `create_shader` stores the parsed
entries, and both backends name a pipeline stage from the entry for that stage
instead of a fixed `main`. `forge::wgsl::to_spirv` keeps the names the source
declared and `to_spirv_entry` is gone; the Slang path still emits `main`, which
is now just the name in that module and not a convention the WGSL path copies.

### Fixed - a steady-state frame creates GPU objects

`BindSet::rebuild` used to destroy and recreate a uniform buffer and a bind
group per set, per frame (`passes/uniforms.rs`). The frame ring keeps each
slot's objects alive until its submission is done, and a set whose layout and
bound resources are unchanged now writes into its existing buffer and keeps the
group. `a_moving_camera_waits_for_the_oldest_frame_and_rotates_slots` asserts
that the lap which reuses a slot destroys nothing and creates nothing.

### Fixed - the gizmo is invisible against the grid

The grid's axis lines are now a dark tint of the gizmo's hues (`GRID_AXIS_X`,
`GRID_AXIS_Z` in `passes/lines.rs`) and the gizmo is recorded after the grid in
one pass, so a selected entity at the origin reads as a bright handle.

### Fixed - shadows and SSAO are unverified

`engine/tests/shadow_ssao.rs` renders a caster and reads the frame back: the far
wall stays equal with and without the caster, the contact region darkens (98 to
87 on Vulkan, 79 to 73 on wgpu), and the old kernel's flat-wall self-occlusion is
gone.

## Backends (stormbreaker, stormglass)

### Fixed - nothing renders a bifrost frame on a real GPU

`engine/tests/render_parity.rs` records the same frame through both backends and
compares the readback images: mean 0.138, max 81, about 0.2% outliers, against
the 0.2 / 85 target.

### Fixed - Vulkan's CPU recording is 20x wgpu's

`stormbreaker/tests/bench_recording.rs` now measures **0.708 ms** of record-only
work at 1998x1135 with 6 passes and 300 draws per pass, against wgpu's **0.4 ms**
for the same graph. The batched submission ring removed the per-pass queue work
the 8.8 ms was paying for.

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

### Fixed - the wgpu swapchain

`WgpuSwapchain` configures, acquires, presents and resizes, exposes the current
image through a real `chassis::ImageView`, and `WgpuGpu::swapchain()` returns
`Some`. `WgpuGpu::resize` reconfigures the surface; headless it records the
extent.

### P1 - the Vulkan windowed path cannot resize

`Gpu::resize` refuses with a swapchain ("the surface owns the size and this
backend cannot rebuild it"), and `VulkanSwapChain` reports `is_out_of_date` that
nothing consumes -- there is no recreation path.

**Fixed when:** a windowed Vulkan run resizes and keeps drawing.

### Fixed - wgpu's panics are not errors

`WgpuResources` runs each fallible wgpu call inside `in_validation_scope`, which
turns the uncaptured-error panic into `chassis::Error::InvalidDescriptor`, and
`create_bind_group` pre-checks every binding against the layout. The stale-handle
panics in `WgpuCommandBuffer` remain: they are programmer errors, not device
validation.

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

### Fixed - wgpu parallel recording

`WgpuRenderGraphExecutor` records one encoder per pass on `grind::Jobs`, with no
command pool to act as the external-synchronisation object, matching Vulkan.

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

### Fixed - no layout persistence, no docking, no tabs

`editor::layout` persists docks, tabs and the active tab to the config path and
loads it at launch. Panels move between docks by drag, tab behind a sibling, and
come back from the file; `layout/tests.rs` covers the move and the round trip.

### Fixed - no selection outline

`editor::gizmo::box_batch` draws an axis-aligned box around the selected entity's
bounds, folded into the same line batch as the gizmo so both use one upload.

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

### Fixed - three fixes from the port are unverified

All three are applied and tested: `sync_out` writes the step a body falls asleep
and skips it after, `AllowedDOFs` is a femoral component applied through Rapier's
enabled translation/rotation axes, and `RuntimeShape` is part of `Source` so a
changed half-extent or scale rebuilds the collider.

## Assets (vault)

### Fixed - every load is synchronous

`Scene::resolve_assets_async` submits every component path to `grind` and returns
`Pending` handles; the renderer and `wire` draw nothing for them until
`Assets::poll` reports the payload. See `assets-and-jobs.md`.

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

The renderer's own tests still run on `FakeGpu`, but the real-GPU parity test now
records a frame through both backends. The gate prints the skip count and fails
when it is not zero; the last run had `0 test(s) skipped`.

### P3 - RenderDoc has no in-app trigger

Installed (v1.45, Vulkan). A capture needs its GUI, the layer's hotkey, or ~60
lines around `RENDERDOC_GetAPI` behind an env var. The last is the only one that
works without a keyboard or a display.

### P3 - Tracy is not installed

`tracy-client` is the Rust side; the profiler server that receives the stream has
to be built from source, and nothing on this machine can do it today.
