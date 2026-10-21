# Model loading, and every extension

glTF is the model format (`gltf` 1.4.1) and the loaders are split by what they
load: `load_model`, `load_model_geometry`, `load_model_animation`, `load_material`,
`load_texture`, `load_lights`. Sixteen extensions are named in the code today:

```
KHR_texture_transform          KHR_materials_specular
KHR_materials_emissive_strength KHR_materials_iridescence
KHR_materials_ior              KHR_materials_sheen
KHR_materials_unlit            KHR_materials_transmission
KHR_materials_clearcoat        KHR_materials_volume
KHR_materials_anisotropy       KHR_materials_diffuse_transmission
KHR_lights_punctual            KHR_texture_basisu
KHR_draco_mesh_compression     EXT_meshopt_compression
```

## Fixed - naming an extension is not handling it

`extensions.rs` classifies every name as handled, refused (with a reason) or
ignored, and `extensions/tests` cover the list; Draco and meshopt are refused by
name because this loader cannot decompress them.

## Fixed - an unknown extension is silent

`extensions::check` runs before decoding: a required name that is not handled is
an error naming it, and `extensionsUsed` is classified into the load report.

## Fixed - the bugs to fix

- `material_extensions.rs` reads every optional block through `json`, so a
  present field of the wrong type is a reported error rather than a silent
  default.
- `uv.rs` owns the V flip and converts `KHR_texture_transform` into engine space
  with `F * T * F`; a rotation test covers the reflected sign.
- `load_texture`'s cache is keyed on `(image index, colour space, sampler)`, so
  two materials sharing an image share one decode, while the same file used as
  sRGB and linear stays two textures.
- KTX2 and decoded PNG both produce `TextureData` through the same material
  path; `KHR_texture_basisu` picks the KTX2 image first and falls back to the
  texture's source.

## P2 - what a load reports

A model load can fail per mesh, per material, per texture and per animation, and
the caller today gets the first error or a `log::warn`. A load report -- what was
read, what was skipped, what was substituted -- is what makes "the model looks
wrong" answerable.

## To add

The specific loading bugs you have in mind belong here, one entry each.
