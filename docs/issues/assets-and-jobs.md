# Assets and jobs

Both need redesigning, and they are one subject because the second is what the
first was waiting for: `vault` now has a worker path and scene resolution uses
it, while `grind`'s policy is still being decided.

What exists: `vault::Assets` is a mount registry with virtual paths and extension
dispatch; a content cache keyed on a hash; a file watcher and a reload path; glTF
loading for models, materials, textures and lights. `grind` is a job system with a
task graph, used by the renderer to record passes in parallel.

## Fixed - every load blocks the frame that asks for it

`Assets::load_async` submits the read and decode to `grind`, and scene
resolution has an asynchronous form: `Scene::resolve_assets_async` sets
`Pending` handles and returns. The editor's drops, scene opening, the sandbox
and aimlab's build all use it. Consumers already answer a pending handle with
"nothing yet" and retry: `Bifrost::ensure_mesh` and `ensure_texture` return
`Ok(None)` and are called again every frame, skinning attach retries until the
model resolves, and `wire` refuses a voice until the clip's bytes exist.

`an_async_resolve_hands_out_a_pending_handle` asserts the handle is `Pending`
with no payload on return and becomes `Ready` after `poll`. Still synchronous:
`meats::load` parses the scene file itself, which is a separate blocking read.

## P1 - what the asset system is for

The questions a redesign has to answer, none of which the current shape does:

- **Identity.** A virtual path, a content hash, and a UUID are three names for one
  thing in this tree. Which one is the key, and which are lookups.
- **Lifetime.** An asset is referenced by a scene, a material, a mesh renderer and
  a cache. Reference counting, a frame-numbered release, or a session-long cache
  are three different answers.
- **Residency.** What happens when the working set is bigger than VRAM. The
  renderer's `docs/WORK.md` Phase 2 asks the same question from the other side.
- **Failure.** A missing texture today is a substituted 1x1. Which failures are
  substituted, which are errors, and which are reported per consumer.
- **Hot reload.** The watcher exists. What a reload does to the handles a frame is
  already holding is the hard half.
- **Threading.** Who may load, who may upload, and where the GPU handoff happens.
  Uploads are not thread-safe in either backend today.

## P1 - what the job system is for

- **A dependency graph, or a pool.** `grind` has both a task graph and a
  `parallel_for`. The renderer uses the second and the graph is exercised by tests.
- **Priorities.** A frame's recording and a texture decode are not the same
  urgency, and today they share a pool.
- **Cancellation.** A load for a scene that was closed has no way to be stopped.
- **Determinism.** Parallel recording is not ordered; anything that feeds a
  simulation step has to be.
- **The main thread's work.** Physics, the frame's submission and the GPU upload
  are main-thread by construction. The system should say so rather than leave it
  to whoever calls it.

## P2 - the cache's policy

`cache.rs` and `cache_hash.rs` exist. Eviction, size limits and what happens on a
hash collision (a hash is not a key until it is compared) are unstated.

## P2 - a load report

See `model-loading.md`: what a load read, skipped and substituted, per asset.

## To add

The redesign's own issues belong here as they are decided.
