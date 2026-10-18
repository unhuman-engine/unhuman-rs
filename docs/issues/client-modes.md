# Client modes and the protocol

The plan, as stated: three parts -- an independent client, a dependent client, and
the protocol between a client and the backend -- so the engine can run anywhere,
including on a phone, and the editor (with an AI prompt) can drive training for an
NPC, a game, a video, a robot or a drone against that backend.

None of this exists yet. Today the editor and the engine are one process: the
editor owns an `adam::App`, runs the engine headless inside its own frame loop and
reads the frame back. There is no client, no protocol and no server.

## The three parts, as I understand them

- **Independent client.** Engine, assets and simulation all local. Works offline,
  on a phone, with no backend. This is the case that decides the design, because
  everything else is this plus a connection.
- **Dependent client.** The simulation, or the assets, or both, live on a backend.
  The client renders what it is told and sends input; a phone is a thin client to
  a workstation's GPU.
- **The protocol.** What a client and a backend agree on. It has to carry input
  and state at minimum, and -- for the training case -- a way for the backend to
  drive the simulation faster than real time.

## P1 - what has to be decided before any of it is written

- **What crosses the boundary.** Input, transforms, the render command stream, or
  pixels? Each is a different product: the first is a game, the last is a
  streaming editor.
- **Authority.** Who owns the simulation when a client is dependent -- and what
  happens when the connection drops mid-step.
- **Determinism.** The training case needs a run to be reproducible from a seed
  and an input log, which means the simulation cannot depend on the frame rate.
  `femur`'s fixed step and `grind`'s unordered pool are the two things in the way.
- **The frame.** Whether the protocol carries frames or the client renders its
  own. The editor's current readback is the shape of the problem: a frame is
  2.3 MP and 9 MB, and moving it every frame is what makes the editor slow
  (`engine.md`).
- **Headless and windowed as one binary.** The engine already runs with `gpu:
  None` (`adam`'s rule 6), which is the seed of an independent client that does no
  rendering at all.

## P1 - the backend is a build, not a mode

`docs/WORK.md` Phase 10 asks for the backend to be a compile-time choice, so a
build ships one driver instead of two. A client that runs on a phone and a backend
that runs on a workstation are the same question asked twice.

## P2 - the editor as the driver

The editor's AI prompt is the client of the backend for training. What it needs
from the protocol is: start a run, step it faster than real time, observe the
result, and stop. That is a smaller protocol than a game's and it should not be
designed as one.

## P2 - what the protocol is carried over

WebSocket, QUIC, a plain TCP length-prefixed stream, or the loopback case of the
same code. The loopback case has to be the same code, or the local path is the one
that never gets tested.

## P3 - scripting, later

`scripting.md`. The protocol's state messages and a script's view of the world
should be the same data, or there are two APIs.

## To add

The design decisions above are the ones I can see. Yours go here as you make them.
