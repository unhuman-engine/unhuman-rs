# Aimlab

The demo layer: `UHE-AIMLAB/aimlab`, an `adam` plugin with rules, a scene setup
and tests, used as the game-side proof that the engine's pieces compose.

It needs fixing. What is wrong with it is not visible from the outside -- the suite
is green and the plugin builds -- so this file is a placeholder for the specific
issues, one entry each.

## What is there

- `rules.rs`: the scoring and hit rules of the aim trainer.
- `scene.rs`: the scene the demo spawns.
- `lib.rs`: the plugin, which is the composition root's second example after the
  editor.

## P2 - what a demo layer is for

Two things, and they are the reason it is worth keeping working:

- It is the only consumer that exercises the engine the way a game would: a plugin
  with its own rules, spawning entities, reading input, driving physics.
- It is the parity target for the C++ tree's own demo, so a behaviour that differs
  between the two is a porting bug with a reference to compare against.

**Fixed when:** the demo runs on both backends, its rules have tests that do not
depend on a device, and the difference from the C++ demo is written down.

## To add

The aimlab issues you have in mind belong here, one entry each.
