# Logging and the console

Two logs exist and they cannot see each other. That is the whole issue, and it is
why leaving the editor to read the engine's output is the normal thing to do
right now.

- The engine logs through `adam::logging` to stderr, the way a headless process
  should. A sink belongs to the logger and nothing can read it back.
- The editor has `mirror::console`, a bounded buffer of *editor* messages, whose
  own doc says exactly this: "The process log goes to stderr through
  `adam::logging`; that sink belongs to the logger and cannot be read back. This
  is the editor's own log, written by the editor."

So a `log::warn!` from `bifrost` about a frame that failed to record goes to the
terminal the editor was launched from, and the console panel shows nothing about
it. A user who launched the editor from a desktop menu has no terminal at all.

## Fixed - one log, two sinks

`adam::logging::Sink` is a trait the logger holds as `Arc<dyn Sink>`, the editor
attaches `mirror::Console`, and the console shows engine and editor records in
one list with level and target. The file sink stays unfiltered.

Notes that matter for the design:

- The console is bounded, and the engine can log more per frame than a bounded
  buffer holds. A record dropped because the panel is full has to say so, or the
  panel lies about what happened.
- A frame that logs per frame (a warning repeated 60 times a second) has to be
  counted rather than listed -- `bifrost` already does this by hand, with a
  `warned` flag per message, which is the same idea done locally.
- Records arrive on whichever thread logged them, including the worker threads
  that record passes, so the sink has to be thread-safe and the panel has to be
  allowed to miss a frame of them.

## Fixed - the editor needs to start without a terminal

The editor attaches a file sink next to its config, prints the path as the
console's first line, and mirrors the same records into the panel.

## Fixed - the console is where a command would go

The console has a one-line input with history; it resolves names and prefixes
against `editor::commands`, runs them (including `undo`, `new scene`, `play`,
`profiler`, `shortcuts`), and reports unknown or ambiguous names as errors.

## Fixed - a level filter

`Sink::enabled` is asked before a record is formatted, and `mirror::Console`
owns the minimum level and a case-sensitive target prefix. The filter is the
logger's, not a second pass over the panel's buffer.
