# Physics

femur needs a rewrite, not a repair. It is rapier3d 0.36 with a glam-first public
API and the only nalgebra access in the tree, in `convert`.

What exists: rigid bodies and colliders built from the scene's description, joints
from the scene's `Joint` component, collision filtering as interaction groups, a
fixed timestep with an accumulator and a substep clamp, and tests for CCD, joints,
filtering and the timestep.

## P1 - the three fixes the port carried and nobody applied

From the C++ tree's own defects, all three still open:

- **Sleeping bodies never sync back.** A body that falls asleep stops being
  written back to its transform, so a scene that was saved after a body settled
  holds the transform from the frame before it slept. The C++ guards on
  `!IsActive`; the Rust side has to either keep syncing a sleeping body once, or
  write the transform when it sleeps.
- **`AllowedDOFs` is serialized and never applied.** A file can say a body is
  constrained to a plane and the simulation ignores it.
- **`BoxCollider`'s `RuntimeShape` is never written**, so a box collider's shape
  in the file and the shape in the simulation can disagree.

**Fixed when:** each has a test that fails without the fix.

## P1 - a rewrite has to decide, not inherit

The questions the current shape does not answer, and a rewrite has to:

- **Units and scale.** A scene is metres; a physics engine has defaults; a model
  has its own. One of them is canonical and the other two convert.
- **Where the fixed step lives.** `adam`'s `Time` has a fixed step and an
  accumulator, femur has one too, and the scene has a scale for the whole
  simulation. Two accumulators is one too many.
- **Determinism.** rapier is deterministic per build; the job system is not
  ordered. Whether a stepped frame is reproducible is a property to state, not to
  discover.
- **nalgebra at the boundary.** Keeping glam canonical is right; every conversion
  is a place a matrix layout can be transposed by accident, so the boundary should
  be one file with tests either side.
- **Collision events into `pings`.** Contacts, enter and exit, as deferred
  messages rather than a second event system.
- **Sleeping, islands and the frame budget.** A large scene that is mostly asleep
  is the case that decides whether stepping can be parallel.
- **Serialization.** Which of the physics components are scene data and which are
  runtime state, decided once, because it decides what a save contains.

## P2 - no async stepping

`grind` exists and physics steps on the main thread. Whether rapier's step can
overlap the renderer's frame -- and what the one-frame latency does to a contact
-- is the design question.

## P2 - the debug view

Contacts, shapes and broad-phase cells drawn through `bifrost`'s line pass, which
exists and already draws the grid and the gizmo.

## To add

The rewrite's own issues belong here as they are decided.
