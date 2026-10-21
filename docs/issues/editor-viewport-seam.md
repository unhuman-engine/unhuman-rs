# The editor's viewport seam

The editor draws the engine's frame by copying it to the CPU every frame and
uploading it as a gpui image (`editor-ui.md`, P0). The copy is the engine's whole
frame (`engine.md`, "the readback is the frame"), and it lands in the editor as a
synchronous `frame_image()` call that blocks until the GPU is done.

This is the interface the editor asks `bifrost` for, so the editor can stop
paying for the readback on its own critical path and start showing a frame that
is one submission old.

## What the editor needs

A frame source the editor polls, rather than one it waits on:

```rust
/// The most recent completed frame, or `None` when none is ready yet.
pub fn poll_frame(&mut self) -> Option<FrameImage>;

/// The submission the next poll will observe, for timing and staleness.
pub fn frame_index(&self) -> u64;
```

- `poll_frame` never blocks. When the GPU has not finished the newest submission,
  it returns the newest *completed* one, or `None` on the first frames. The
  editor already tolerates `None`: it keeps the last image and shows the reason
  (`viewport::frame_status`).
- The returned `FrameImage` owns its pixels. The editor's `FrameSource` trait
  (`frame_source.rs`) keys on `frame_index` and turns a changed image into a
  `gpui::RenderImage`; it must not be handed a buffer the next frame overwrites.
- `frame_index` is monotonic per submission, so the editor can tell a repeated
  poll (same frame, skip the upload) from a stale one (the ring is behind).

## What exists today

`BifrostPlugin::frame_image() -> FrameImage` returns the frame just captured, and
`frame_index()` counts submissions. `frame_image` is the blocking call: by the
time it returns, the readback has waited for the GPU. `frame_source::BifrostSource`
wraps both behind `FrameSource`, so the editor's call sites already go through the
seam; only the wait inside `frame_image` has to go.

## The fix on the bifrost side

A ring of N readback buffers, one fence each, read N-1 submissions back
(`engine.md`, P0). `poll_frame` then maps the oldest buffer whose fence has
signalled with no wait, copies it into an owned image, and returns it; the frame
being recorded reuses the freed slot. The engine keeps its own no-overlap
guarantee -- no buffer is written until its fence is re-signalled -- so the editor
never sees a torn frame.

**Fixed when:** `editor` advances and draws frames while `frame_image`-equivalent
polling blocks for no measurable time, and the editor's loop period at 2.3 MP is
the engine's GPU time plus the UI paint, with no readback wait in between.
