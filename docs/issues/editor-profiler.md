# Editor profiler

A live panel in the editor, not a capture file: frame time, memory, CPU and GPU
load, synchronisation, and how long each draw took -- all of it real time, on
screen, and none of it written to disk. The storage constraint is part of the
requirement: a profiler that saves traces eats a disk, and this one has to be
free to leave running.

What exists today: the status bar shows `fps` and `frame_ms` from
`mirror::Stats`, which is a smoothed frame interval -- the whole loop, not the
engine's frame. There is a per-pass counter in `bifrost` (`FrameStats`) and a
draw-call counter, and neither is displayed.

## P1 - the status bar measures the wrong thing

`mirror::Stats::update` smooths `dt`, the interval between ticks, so `frame_ms` is
the loop's period and not the engine's frame. Measured this session: at a 719x439
viewport the loop was 7.0 ms and the engine's frame was 4.0 ms of it; at
1998x1135 the loop was 25.9 ms and the engine 15.3 ms. One number cannot be read
as the other, and the panel has to show both.

**Fixed when:** the panel separates the loop's period, the engine's frame, the
readback, and the UI's own paint.

## P1 - what the panel has to show

| number | where it would come from |
|---|---|
| frame time, and the loop's period | `mirror::Stats`, plus a timer around `advance_frame` |
| engine frame, render vs readback | a timer inside `bifrost::render`, which is where the 11.2 ms of 11.8 ms lives |
| UI paint | the loop's period minus the engine's frame |
| draw calls, passes, instances | `bifrost`'s existing counters, which nothing displays |
| GPU time per pass | timestamp queries on both backends -- `docs/WORK.md` Phase 10 |
| CPU load | `/proc/self/stat` on Linux; the platform crate is where that belongs |
| memory | `/proc/self/statm` for resident, plus the renderer's own allocations if it grows an allocator hook |
| GPU load | AMD's `fdinfo` or `sysfs` where it exists; not portable, so it is a best-effort line |
| sync stalls | the readback's wait, which is the one the frame actually pays |

**Fixed when:** every row above has a source, the panel updates every frame, and
nothing is written to disk.

## P2 - a ring of history, in memory

A number tells you where a frame is; a sparkline tells you when it went wrong.
Bounded, in memory, never flushed.

## P2 - the panel has to survive the frame being slow

Reading `/proc` and the GPU's counters every frame is itself work, and a profiler
that costs 2 ms of a 10 ms frame has changed the thing it measures. Sample the
expensive rows on a timer, not per frame.

## P3 - a toggle

Off by default, one key to show it, and no cost when hidden.
