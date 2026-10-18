# Issues

One file per subject, so a chain of work has somewhere to accumulate instead of
growing one document. Add to the file that fits; start a new one when a subject
does not have a file yet.

## The files

| file | subject |
|---|---|
| `engine.md` | renderer, backends (Vulkan and wgpu), the RHI, the editor's engine side, physics, assets, tooling |
| `editor-ui.md` | the editor's UI and features. The editor needs renewing, not patching |
| `editor-profiler.md` | a live profiler in the editor: frame time, memory, CPU, GPU, sync, per-draw |
| `logging.md` | the engine's log and the editor's console are two different things |
| `scene-format.md` | the save format and its parser |
| `model-loading.md` | glTF loading, and every extension it has to handle |
| `physics.md` | femur needs a rewrite, not a repair |
| `assets-and-jobs.md` | the asset system and the job system, both redesigned together |
| `client-modes.md` | independent client, dependent client, the protocol between them, and the backend |
| `aimlab.md` | the demo layer |
| `scripting.md` | Lua, later |

## How an entry reads

Each entry says what is wrong, the evidence for it, and the thing that would say
it is fixed. Evidence is a measurement, a file, or a test that does not exist --
not a feeling, because a list of feelings cannot be worked through.

Priorities:

- **P0** the frame is wrong, missing or cannot be seen at all
- **P1** correctness, or a claim the repo makes that is not true yet
- **P2** cost, or a capability a phase needs
- **P3** polish, or a decision left open

## Working through it

- The gate is `scripts/gate.sh`. It needs `SLANGC`: the vendored Slang it looks
  for lives in a C++ checkout that is not in this tree, so a local run is
  `SLANGC=/opt/shader-slang-bin/bin/slangc ./scripts/gate.sh`. Without it the
  shader tests panic by design, and the gate prints how many tests skipped.
- `UHE_VALIDATION=1` is what makes the Vulkan suites mean anything.
- Measurements in `engine.md` were taken on this machine: RADV on a RENOIR iGPU,
  release build, 1998x1135 unless a size is given.
