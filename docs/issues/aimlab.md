# Aimlab

The demo layer: `UHE-AIMLAB/aimlab`, an `adam` plugin with rules, a scene setup
and tests, used as the game-side proof that the engine's pieces compose.

It needs fixing. What is wrong with it is not visible from the outside -- the suite
is green and the plugin builds -- so this file is a placeholder for the specific
issues, one entry each.

## What is there

- `rules.rs`: the scoring, hit and magazine rules of the aim trainer, all pure
  functions of their arguments.
- `scene.rs`: the scene the demo spawns.
- `lib.rs`: the plugin, the input edges, and the display seam.
- `tests/aimlab.rs`: the rules against a real `meats` scene, no GPU.

## The display seam

`AimLabPlugin::report` returns a `Report`: hits, misses, shots, points,
accuracy, rounds, `max_ammo`, the reloading flag and its progress, plus
`Report::CONTROLS`. This is the HUD contract. The demo never draws; a HUD is
the sandbox's or the engine's to build from this. Nothing here reaches into
another crate's renderer.

## Fixed - the gun seam

`bifrost` skins with `JOINTS_0`/`WEIGHTS_0` and the `Animator` glue, so the demo
attaches `assets/models/pistol_animations_blender.glb` as a camera child with
the C++'s offset (`GUN_OFFSET`), facing fix (`GUN_ROTATION_DEGREES`) and scale
(`GUN_SCALE`, 0.05). `Idle` is the resting clip, `Fire` runs on a shot for the
clip's own duration, and `Reload_Complete` runs while the magazine reloads; the
animator stays when the asset is absent, so ammo and reload tests need no file.
Still open: a HUD renderer, or an ImGui-equivalent panel, to read `Report`.

## P2 - what a demo layer is for

Two things, and they are the reason it is worth keeping working:

- It is the only consumer that exercises the engine the way a game would: a plugin
  with its own rules, spawning entities, reading input, driving physics.
- It is the parity target for the C++ tree's own demo, so a behaviour that differs
  between the two is a porting bug with a reference to compare against.

**Fixed when:** the demo runs on both backends, its rules have tests that do not
depend on a device, and the difference from the C++ demo is written down.

## Difference from the C++ demo

The camera framing, target wall, hit rule, spawn cadence, seven-round magazine,
`R` reload, auto-reload at zero, 3.5 s reload timer and accuracy maths match.

- **HUD.** The C++ draws an ImGui window, crosshair and gun tweak sliders. Rust
  exposes `Report`; the sandbox draws a crosshair, nothing else.
- **Audio.** The shot is a one-shot through `wire` from the camera's sound
  component; a counting backend test observes the voice.
- **Animation.** `Fire`'s 0.5 s and `Reload_Complete`'s duration, and the recoil
  kick, come from the pistol's clips in the C++; neither is present here, so the
  reload uses the C++'s documented 3.5 s default and there is no recoil.
- **Picking.** The C++ reads an entity-ID framebuffer pixel from the previous
  frame. Rust casts a ray at the same instant.
- **Controls.** The C++ lists `ESC` cursor release/lock, `ALT`+mouse orbit and
  scroll zoom from its `EditorCamera`; this demo has its own yaw/pitch and does
  not port those, so they are not in `Report::CONTROLS`.
- **Reload sound.** The C++ plays `reload.wav`; this tree has only `gunshot.wav`,
  so the reload is silent.

## To add

The aimlab issues you have in mind belong here, one entry each.
