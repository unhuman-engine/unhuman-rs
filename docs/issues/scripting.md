# Scripting

Lua, later, as stated. Nothing exists today: no scripting crate, no binding, no
host. This file is here so the decisions have somewhere to land before the work
starts, because two of them are already being made by other parts of the tree.

## P2 - the decisions, before any binding

- **Which Lua.** `mlua` (bindings to the reference implementation, C), `rlua`
  (same family), `piccolo`/`lua-rs` (pure Rust, incomplete). The tree is Rust-first
  with one documented C exception (ALSA), so this is the second exception to argue
  for or against -- and `forge` already has the same argument to make about
  `libslang.so`.
- **What a script may touch.** An ECS world, a component, a transform, an asset.
  A script API that mirrors the Rust one leaks lifetimes into a language that has
  none; one that mirrors the protocol's messages (`client-modes.md`) is smaller and
  is the same data on a client and on a server.
- **Where scripts run.** On the simulation thread with a step budget, or as
  coroutines the step drains. A script that blocks a frame is a script that stalls
  a training run.
- **Determinism.** The training case needs a run reproducible from a seed and an
  input log (`client-modes.md`), so a script's view of time and randomness cannot
  come from the wall clock.
- **Hot reload.** The same question `assets-and-jobs.md` asks about assets, and
  the answer has to be the same one.

## P3 - what a script replaces

`aimlab` is a plugin with rules. A scripting host's first proof is the same demo
written as a script, because a host that cannot express the one game in the tree
is a host nobody will use.

## To add

The scripting issues belong here as the design is decided.
