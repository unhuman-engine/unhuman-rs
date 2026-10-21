# Physics

femur needs a rewrite, not a repair. It is rapier3d 0.36 with a glam-first public
API and the only nalgebra access in the tree, in `convert`.

What exists: rigid bodies and colliders built from the scene's description, joints
from the scene's `Joint` component, collision filtering as interaction groups, a
fixed timestep with an accumulator and a substep clamp, and tests for CCD, joints,
filtering and the timestep.

## P1 - the three fixes the port carried and nobody applied

From the C++ tree's own defects. All three are done; each has a test that fails
without the fix.

- **Sleeping bodies never sync back.** Done. `sync_out` skips a body only once it
  is already asleep, and writes on the step it falls asleep (`body::writes_pose`,
  unit-tested). The write-once is what a stale saved scene needed; the skip after
  is the frame-budget win. The C++ guard skips the transition too, which is the
  bug.
- **`AllowedDOFs` is serialized and never applied.** Done, as a femoral
  component (`components::AllowedDOFs`): translation and rotation per axis, from
  the file's `[tx, ty, tz, rx, ry, rz]` order, applied through Rapier's
  `enabled_translations`/`enabled_rotations`. The test locks one translation axis
  and asserts the body stays on its plane; the control falls.
- **`BoxCollider`'s `RuntimeShape` is never written.** Done. The shape is part of
  `Source`, so a changed half-extent or transform scale rebuilds the collider; the
  test moves the file's half-extent and the scale and asserts the resting height
  follows.

**Fixed when:** each has a test that fails without the fix.

## P1 - a rewrite has to decide, not inherit

Decided; the ones with code are implemented in `src/`.

- **Units and scale.** The metre is canonical. A scene is metres and Rapier is SI
  metres, so there is no scene-to-engine conversion and no global physics scale:
  the second scale would be a second thing to keep consistent. A model's own
  scale converts once, through `Transform::scale`, in `shape::scaled_shape`.
- **Where the fixed step lives.** In `adam::Time`. It owns the accumulator and
  the fixed rate; `Physics3D::step` consumes the `dt` it is handed and integrates
  once. Femur has no accumulator. There is one clock.
- **Determinism.** A stepped frame is reproducible for a fixed build: Rapier is
  bit-deterministic, the backend holds no generator, and the step is
  single-threaded. It breaks on a different build or SIMD width, on a different
  `dt`, and on stepping from more than one thread. `two_identical_runs_agree_bit_for_bit`
  is the claim; async stepping would break it, so the step stays on one thread
  (P2).
- **The math boundary.** `convert` is the only file that names a Rapier math
  type, and it has tests in both directions, including a single-axis rotation
  that an Euler-order mix-up would transpose.
- **Collision events into `pings`.** `contacts` diffs touching pairs and queues
  `Enter`, `Contact` and `Exit` messages; the engine drains them and forwards to
  `pings`. The backend owns no bus, so a dropped queue cannot change a stepped
  frame.
- **Sleeping, islands and the frame budget.** A sleeping body is not written, and
  the sleep-transition is. Stepping itself stays on the main thread; the barrier
  is the determinism claim (P2).
- **Serialization.** Scene data: `Transform`, `RigidBody`, `Collider`. Authored
  physics data: `Ccd`, `CollisionFilter`, `AllowedDOFs`, `Joint`. Runtime state,
  never saved: `BodyHandle`, the touching set, the collision queue, the Rapier
  sets.

## P2 - stepping stays on the main thread

Decided against, with the measurement. `bench_step` (500 dynamic boxes on a
floor at 60 Hz) costs 0.453 ms/step in the dev profile. The aimlab scene is a
handful of bodies, far below that. Overlapping the step with the frame would
save at most half a millisecond at a body count no scene here reaches, while
adding a fixed step (16.7 ms) between authored input and the contact it causes,
and breaking the single-thread determinism claim unless it were a second,
opt-in backend.

Smoothness is solved without moving the step: `PhysicsBackend::interpolate`
blends the poses before and after the last fixed step by `Time::alpha`, and
records the blend as `written`, so `sync_in` does not mistake it for an authored
teleport. `tests/interpolation.rs` covers the blend, the no-feedback property,
and an authored change still winning.

**Fixed when:** the measurement exists and the queue says why the answer is no.

## Fixed - the debug view

`debug::PhysicsDebug` is the data: collider wireframes, contact crosses and
broad-phase boxes as world-space `DebugLine` segments, built by
`Physics3D::debug_view`. The sandbox's `--physics-debug` converts them into
`bifrost`'s line format and draws them in the editor pass; the physics scene
shows 196 segments from the second frame.

## To add

The rewrite's own issues belong here as they are decided.
