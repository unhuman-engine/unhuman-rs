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

## P1 - one log, two sinks

**Fixed when:** the logger has a sink that can be attached at runtime, the editor
attaches one, and the console panel shows engine records and editor records in
one list with their level and their target.

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

## P1 - the editor needs to start without a terminal

Launching from a desktop entry, the process has no stderr anyone will read, so
everything logged before the window opens is lost. That is also where the
interesting failures are: no GPU, no driver, a shader that will not compile.

**Fixed when:** the editor writes its log to a file *and* to the console, and the
path is printed in the console's first line.

## P2 - the console is where a command would go

`mirror::console` is a buffer. The same panel is where a command line would live
once there are commands to type -- the command registry in `mirror::commands`
already has the vocabulary.

## P3 - a level filter

A debug build logs a lot. The panel needs to filter by level and by target, and
the filter has to be the logger's, not a second copy of it.
