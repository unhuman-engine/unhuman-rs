# Scene format and its parser

The format exists: `.unhuman`, **YAML**, round-tripped by `meats`
(`UHE/src/scene/tests/serializer.rs` is "the `.unhuman` YAML round trip, and the
shipped scenes it has to read"). It is parsed with `yaml_serde`.

## P1 - the YAML crate is the wrong one

Done. The dependency is `yaml_serde` (0.10), the maintained fork of `serde_yaml`
by the YAML org, pure Rust through `libyaml-rs` 0.3 -- a Rust transpilation of
libyaml, no C and no `unsafe-libyaml`. Its API is `serde_yaml`'s, so the swap is
the dependency line in `UHE/src/scene/Cargo.toml` and the workspace root; the
round trip tests and the shipped scenes pass unchanged.

## P1 - one format or two

Decided. **A save is `.unhuman`, and an asset is a payload in `vault`.** They are
two roles of the same text, not two formats:

- `meats` owns `.unhuman`: YAML, diffable, the only form a save writes. `vault`
  already classifies it as an asset (`AssetKind::Text`).
- `vault` owns content addressing, not a second scene syntax. The binary document
  in `docs/TRANSFER.md` (24-byte header, schema hash, `postcard` payload, TOML
  export) is a **derived import**, unimplemented: when it lands it is a cook of
  the YAML, and its loader has to produce the same `meats::Scene` by calling
  `meats::from_yaml`, never a parallel reader. The seam is `meats::from_yaml` /
  `meats::to_yaml`; anything that needs a `Scene` goes through them.

## P1 - what a save has to survive

Done. Each is a test in `UHE/src/scene/tests/survival.rs`:

- An entity id that outlives the file: hierarchy is keyed by UUID and survives.
- A component added since the file was written: logged and skipped on load, and
  named under `DroppedComponents` for that entity on the next save, so the file
  says what it lost instead of dropping it silently.
- A component *removed* since the file was written: the shipped `box2d.unhuman`
  `RigidBody2DComponent` is caught this way.
- A cycle, a reparented subtree, a dangling parent: a cycle loads as a forest and
  saves once; a subtree saves under its new parent; a dangling parent is left a
  root with `Parent: 0`.
- Numbers: floats round-trip through text including negative zero, `f32::MAX` and
  `f32::MIN_POSITIVE`.
- A path that no longer resolves: reported per entity by `resolve_assets`.

## P2 - schema versioning

Done. The document carries `Format: unhuman` and `FormatVersion`; unversioned
files load as version 0 and are migrated to the current version on save, and a
file from a newer build is refused naming its version. The migration loop in
`serialize.rs` is the place a real upgrade goes.

## To add

The parser issues you have in mind belong here, one entry each.
