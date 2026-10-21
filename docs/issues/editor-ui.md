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

## Fixed - no layout persistence

`editor::layout` persists docks, tabs, the active tab and panel visibility to the
config path and loads it at launch; a rearranged layout comes back after a
relaunch and a closed panel can be reopened from the View menu.

## Fixed - docking and tabs

Panels move between docks by drag and tab behind a sibling; `layout/tests.rs`
covers moving a panel into another dock and the round trip through the saved
file. Tearing a panel out into its own window is not implemented.

## P1 - the keyboard is unreachable from the viewport's perspective

W/E/R for the gizmo mode, Ctrl+Z and Ctrl+Shift+Z for undo and redo, Delete to
destroy: all bound to the viewport element's key handler, none of them exercised,
because no key event reaches an application in this session (see `engine.md`,
"the orbit camera and the keyboard are unverified").

**Fixed when:** they are pressed on a real session, and the shortcut list is
somewhere a user can find it.

## Fixed - no selection outline

`editor::gizmo::box_batch` draws the selected entity's bounds as an axis-aligned
box in the same line batch as the gizmo, driven by the hierarchy's selection.

## P2 - the content browser cannot drag into the viewport

`drop_files` handles a drop from outside the window. A drag from the content
browser to the viewport -- the way a model gets added -- is not wired, and neither
is a drag onto the hierarchy to reparent.

## P2 - the panels do not record what they change

`mirror` has a `Command` history and the keys are bound. Every mutation a panel
performs has to go through it or undo silently loses it.

**Fixed when:** a test mutates a property, a rename, a reparent and a create, then
undoes all four.

## Fixed - a menu bar, and a command registry the menus read

`editor::commands` is the one registry the menu bar, the toolbar, the shortcut
table and the console's command line read; a command with no key is reachable
from the menu alone.

## P3 - the theme is two systems

`mirror`'s semantic tokens and the editor's `EditorTheme` both exist; the engine
palette wins for a non-editor build.

## P3 - the viewport has no accessibility role

gpui logs `focused element ... has an id but no role` for the focusable viewport.

## To add

The list above is what I could see. The chain of UI and feature issues is yours to
extend -- one entry per file in this directory, or a new file if it is a subject
of its own.
