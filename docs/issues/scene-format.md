# Scene format and its parser

The format exists: `.unhuman`, **YAML**, round-tripped by `meats`
(`UHE/src/scene/tests/serializer.rs` is "the `.unhuman` YAML round trip, and the
shipped scenes it has to read"). It is parsed with `serde_yaml`.

## P1 - the YAML crate is the wrong one

`UHE/src/scene/Cargo.toml` depends on `serde_yaml`, which is deprecated and is the
only YAML option in this tree that carries a C dependency (`unsafe-libyaml`). The
rest of the workspace is Rust-first, and this is the one place that is not.

**Fixed when:** the dependency is a maintained pure-Rust YAML crate and the round
trip tests still pass against the shipped scenes.

## P1 - one format or two

Two things describe a scene in this tree, and they need to be reconciled before
either is called the save format:

- `meats`: `.unhuman`, YAML, the ECS scene document.
- `vault`: a binary document with a 24-byte header, a content-addressed schema
  hash and a `postcard` payload, with a TOML export for editing.

**Fixed when:** it is written down which one a save is, which one an asset is, and
whether the other is an import path or a mistake.

## P1 - what a save has to survive

The parts that break a format are the ones nobody writes down. Each of these is a
test that does not exist yet:

- An entity id that outlives the file: `.unhuman` keys hierarchy by UUID, so a
  reload has to keep it.
- A component that was added since the file was written: unknown components are
  currently logged and skipped (`UHE/src/scene/src/serialize.rs`), which is right,
  but the *file* is then silently lossy on the next save.
- A component that was *removed* since the file was written.
- A scene with a cycle, a reparented subtree, or a parent that no longer exists.
- Numbers: floats round-tripped through text, and negative zero.
- A path that no longer resolves, which `resolve_assets` reports per entity.

**Fixed when:** each has a test, and the file says what it dropped.

## P2 - schema versioning

The header carries a schema hash, which catches a mismatch and cannot migrate one.
A version number and a migration path is the difference between "this file cannot
be opened" and "this file was upgraded".

## To add

The parser issues you have in mind belong here, one entry each.
