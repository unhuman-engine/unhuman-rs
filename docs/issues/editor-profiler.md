# Editor profiler

A live panel in the editor, not a capture file: frame time, memory, CPU and GPU
load, synchronisation, and how long each draw took -- all of it real time, on
screen, and none of it written to disk. The storage constraint is part of the
requirement: a profiler that saves traces eats a disk, and this one has to be
free to leave running.

What exists today: the status bar names the loop period, the engine frame, the
readback and the UI paint. The panel adds the counters and, on Vulkan, the
per-pass GPU times; wgpu reports them unavailable.

## Fixed - the status bar measures the wrong thing

`mirror::Stats::update` smooths `dt`, the interval between ticks, so `frame_ms` is
the loop's period and not the engine's frame. Measured this session: at a 719x439
viewport the loop was 7.0 ms and the engine's frame was 4.0 ms of it; at
1998x1135 the loop was 25.9 ms and the engine 15.3 ms. The status bar now shows
the loop period, the engine frame, the readback (or `-` when it was not
collected) and the UI paint separately, and `timings_status` has a test with
four distinct values so a collapse back into one number fails.

## P1 - what the panel has to show

| number | where it would come from |
|---|---|
| frame time, and the loop's period | `mirror::Stats`, plus a timer around `advance_frame` |
| engine frame, render vs readback | a timer inside `bifrost::render`, which is where the 11.2 ms of 11.8 ms lives |
| UI paint | the loop's period minus the engine's frame |
| draw calls, passes, instances | `bifrost`'s existing counters, which nothing displays |
| GPU time per pass | `chassis::Gpu::gpu_timings`, filled by Vulkan's timestamp queries; wgpu has none, so the panel reads unavailable |
| CPU load | `/proc/self/stat` on Linux; the platform crate is where that belongs |
| memory | `/proc/self/statm` for resident, plus the renderer's own allocations if it grows an allocator hook |
| GPU load | AMD's `fdinfo` or `sysfs` where it exists; not portable, so it is a best-effort line |
| sync stalls | the readback's wait, which is the one the frame actually pays |

**Fixed when:** every row above has a source, the panel updates every frame, and
nothing is written to disk.

### Fixed - GPU time per pass

`chassis::Gpu` grew `enable_gpu_timing`, `collect_gpu_timings`, `gpu_timings`
and `gpu_frame_ms`. Vulkan answers them from its existing `VkQueryPool` and
carries the graph's own pass names through, owned, so the panel reads a name
that outlives the frame. wgpu answers `false` and `None`, and the panel says the
row is not collected rather than showing a zero. The editor enables timing only
when the profiler panel is first shown, so a hidden panel pays nothing, and
`bifrost` collects the split for the frame it just submitted.

## P2 - a ring of history, in memory

A number tells you where a frame is; a sparkline tells you when it went wrong.
Bounded, in memory, never flushed.

## P2 - the panel has to survive the frame being slow

Reading `/proc` and the GPU's counters every frame is itself work, and a profiler
that costs 2 ms of a 10 ms frame has changed the thing it measures. Sample the
expensive rows on a timer, not per frame.

## P3 - a toggle

Off by default, one key to show it, and no cost when hidden.
