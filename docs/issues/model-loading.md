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

## P1 - naming an extension is not handling it

A name in the code can mean three different things and the code does not say
which: parsed into engine parameters, recognised and refused, or read and then
dropped. Draco and meshopt are the ones to check first -- both are *compression*,
so a file that uses them cannot be loaded by reading the buffers, and a loader
that recognises the name without decompressing produces a model with no geometry
rather than an error.

**Fixed when:** every extension in the list is one of handled, refused with a
message, or ignored *with a log line*, and a test loads a file for each.

## P1 - an unknown extension is silent

Nothing checks `extensionsRequired`. A file that needs an extension this loader
does not implement loads as a model missing whatever that extension described,
with no warning.

**Fixed when:** `extensionsRequired` is checked on load, an unimplemented one is
an error naming it, and `extensionsUsed` is logged.

## P1 - the bugs to fix

Loading has bugs; this is where they go, one entry each, with the file that shows
it. The ones visible from the code:

- `load_material` reads extension values as raw JSON (`extension_value`), so a
  malformed one is a runtime `None` rather than a reported error.
- Texture coordinates: glTF's V is flipped on import (`material.rs` says so), so
  every texture path has one flip to get right, and `KHR_texture_transform` adds
  another.
- The image cache in `load_texture` is per asset; whether it is keyed on the file
  or on the decoded bytes decides whether two materials sharing an image share a
  texture.
- `.ktx2` is a separate path (`ktx2.rs`) from the decoded images, so a basisu
  texture and a PNG take different routes through the same material.

## P2 - what a load reports

A model load can fail per mesh, per material, per texture and per animation, and
the caller today gets the first error or a `log::warn`. A load report -- what was
read, what was skipped, what was substituted -- is what makes "the model looks
wrong" answerable.

## To add

The specific loading bugs you have in mind belong here, one entry each.
