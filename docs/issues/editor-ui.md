# Editor UI and features

The editor needs renewing, not patching. What is here is a first pass at a
layout: four fixed panels, a toolbar and a status bar, drawn by gpui, over a
readback image of the engine's frame.

What it does today: the hierarchy lists entities, the inspector edits their
properties, the content browser lists the mounted assets, the viewport orbits,
picks and drags a gizmo, and the toolbar plays, pauses, steps and stops. Nothing
else. No docking, no tabs, no floating panels, no layout persistence, no
selection outline, no drag from the browser into the viewport, no undo of
anything the panels do not record, and no menu bar.

## P0 - the viewport is a picture of the engine, not the engine

The engine renders headless and `bifrost` copies the composed frame back to the
CPU every frame; gpui uploads it as an image. The copy is ~95% of the engine's
frame (`engine.md`, the readback entry), and it is why the editor is slower than
the engine it is showing.

**Fixed when:** the viewport shows the engine's texture without a round trip
through the CPU, or the readback is at least asynchronous and off the frame's
critical path. The interface the editor asks `bifrost` for is in
`editor-viewport-seam.md`.

## P1 - no layout persistence

`Layout` in `UHE-EDITOR/editor/src/app.rs` is three widths and a height, in
memory, for the life of the process. There is no equivalent of an `imgui.ini`.

**Fixed when:** a rearranged layout comes back after a relaunch, and a panel can
be closed and reopened.

## P1 - docking and tabs

Every panel is fixed in place. Godot and Unity both let a panel be dragged into
another, tabbed behind a sibling, or pulled out into its own window.

**Fixed when:** a panel can be moved to a different edge, tabbed, and restored.

## P1 - the keyboard is unreachable from the viewport's perspective

W/E/R for the gizmo mode, Ctrl+Z and Ctrl+Shift+Z for undo and redo, Delete to
destroy: all bound to the viewport element's key handler, none of them exercised,
because no key event reaches an application in this session (see `engine.md`,
"the orbit camera and the keyboard are unverified").

**Fixed when:** they are pressed on a real session, and the shortcut list is
somewhere a user can find it.

## P2 - no selection outline

A selected entity is visible by its gizmo and by nothing else. A model under the
gizmo, or an entity off screen, has no indication.

**Fixed when:** the selection is drawn in the viewport, and the hierarchy's
selection matches it.

## P2 - the content browser cannot drag into the viewport

`drop_files` handles a drop from outside the window. A drag from the content
browser to the viewport -- the way a model gets added -- is not wired, and neither
is a drag onto the hierarchy to reparent.

## P2 - the panels do not record what they change

`mirror` has a `Command` history and the keys are bound. Every mutation a panel
performs has to go through it or undo silently loses it.

**Fixed when:** a test mutates a property, a rename, a reparent and a create, then
undoes all four.

## P2 - a menu bar, and a command registry the menus read

`mirror::commands` exists and the toolbar hardcodes its buttons. A menu that lists
commands from the registry is how a shortcut and a menu stop being two places to
change one thing.

## P3 - the theme is two systems

`mirror`'s semantic tokens and the editor's `EditorTheme` both exist; the engine
palette wins for a non-editor build.

## P3 - the viewport has no accessibility role

gpui logs `focused element ... has an id but no role` for the focusable viewport.

## To add

The list above is what I could see. The chain of UI and feature issues is yours to
extend -- one entry per file in this directory, or a new file if it is a subject
of its own.
