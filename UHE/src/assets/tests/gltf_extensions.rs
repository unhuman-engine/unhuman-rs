//! Material extensions, punctual lights, and the refusal of what cannot be
//! decoded, driven through a hand-written glTF.

use std::fs;
use std::path::PathBuf;

use vault::{
    texture_slot, AssetKind, Assets, BlendApproach, Error, LightType, MATERIAL_TEXTURE_SLOT_COUNT,
};

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("vault-extensions-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("scratch directory");
        Self { root }
    }

    fn write(&self, name: &str, contents: &[u8]) {
        fs::write(self.root.join(name), contents).expect("scratch file");
    }

    fn assets(&self) -> Assets {
        Assets::new(&self.root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

// ---- a PNG built by hand ----------------------------------------------------
//
// A 1x1 opaque white RGBA PNG, built rather than carried as a fixture so the
// decode is checked against texels the test chose. The IDAT stream is a single
// *stored* deflate block, so no compressor is needed.

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for byte in bytes {
        a = (a + *byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);

    let mut checksummed = Vec::with_capacity(4 + data.len());
    checksummed.extend_from_slice(kind);
    checksummed.extend_from_slice(data);
    out.extend_from_slice(&crc32(&checksummed).to_be_bytes());
    out
}

fn white_png() -> Vec<u8> {
    let raw = [0u8, 255, 255, 255, 255];
    let length = raw.len() as u16;
    let mut zlib = vec![0x78, 0x01, 0x01];
    zlib.extend_from_slice(&length.to_le_bytes());
    zlib.extend_from_slice(&(!length).to_le_bytes());
    zlib.extend_from_slice(&raw);
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(&1u32.to_be_bytes());
    header.extend_from_slice(&1u32.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    out.extend(chunk(b"IHDR", &header));
    out.extend(chunk(b"IDAT", &zlib));
    out.extend(chunk(b"IEND", &[]));
    out
}

/// Three positions, little-endian `f32`: (0,0,0), (1,0,0), (0,1,0).
fn triangle_bin() -> Vec<u8> {
    let mut bin = Vec::with_capacity(36);
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bin.extend_from_slice(&value.to_le_bytes());
    }
    bin
}

fn close(left: f32, right: f32) -> bool {
    (left - right).abs() < 1e-6
}

/// Every supported material extension, plus a base-colour and a clearcoat map.
const EXTENDED_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": [
    "KHR_materials_pbrSpecularGlossiness", "KHR_texture_transform", "KHR_materials_unlit",
    "KHR_materials_emissive_strength", "KHR_materials_ior", "KHR_materials_clearcoat",
    "KHR_materials_specular", "KHR_materials_sheen", "KHR_materials_transmission",
    "KHR_materials_volume", "KHR_materials_iridescence", "KHR_materials_anisotropy",
    "KHR_materials_diffuse_transmission", "KHR_lights_punctual", "KHR_texture_basisu",
    "EXT_meshopt_compression"
  ],
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ { "attributes": { "POSITION": 0 }, "material": 0 } ]
  } ],
  "materials": [ {
    "name": "Ext",
    "pbrMetallicRoughness": {
      "baseColorTexture": {
        "index": 0,
        "extensions": {
          "KHR_texture_transform": { "offset": [0.25, 0.5], "scale": [2.0, 3.0], "rotation": 0.0 }
        }
      }
    },
    "extensions": {
      "KHR_materials_emissive_strength": { "emissiveStrength": 2.5 },
      "KHR_materials_ior": { "ior": 1.45 },
      "KHR_materials_unlit": {},
      "KHR_materials_clearcoat": {
        "clearcoatFactor": 0.3, "clearcoatRoughnessFactor": 0.1,
        "clearcoatTexture": { "index": 0 }
      },
      "KHR_materials_specular": {
        "specularFactor": 0.4, "specularColorFactor": [0.9, 0.8, 0.7]
      },
      "KHR_materials_sheen": {
        "sheenColorFactor": [0.1, 0.2, 0.3], "sheenRoughnessFactor": 0.6
      },
      "KHR_materials_transmission": { "transmissionFactor": 0.75 },
      "KHR_materials_volume": {
        "thicknessFactor": 0.2, "attenuationDistance": 3.0,
        "attenuationColor": [0.5, 0.6, 0.7]
      },
      "KHR_materials_iridescence": {
        "iridescenceFactor": 0.5, "iridescenceIor": 1.4,
        "iridescenceThicknessMinimum": 120.0, "iridescenceThicknessMaximum": 350.0
      },
      "KHR_materials_anisotropy": { "anisotropyStrength": 0.8, "anisotropyRotation": 0.25 },
      "KHR_materials_diffuse_transmission": {
        "diffuseTransmissionFactor": 0.35, "diffuseTransmissionColorFactor": [0.2, 0.3, 0.4]
      },
      "KHR_materials_pbrSpecularGlossiness": {
        "diffuseFactor": [0.1, 0.2, 0.3, 0.4], "specularFactor": [0.5, 0.6, 0.7],
        "glossinessFactor": 0.85
      }
    }
  } ],
  "images": [ { "uri": "white.png" } ],
  "textures": [ { "source": 0 } ],
  "buffers": [ { "uri": "triangle.bin", "byteLength": 36 } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;

#[test]
fn every_material_extension_value_reaches_the_material() {
    let scratch = Scratch::new("extended");
    scratch.write("model.gltf", EXTENDED_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());
    scratch.write("white.png", &white_png());

    let mut assets = scratch.assets();
    let id = assets.load("model.gltf").expect("the model loads");
    assert_eq!(assets.kind(id), Some(AssetKind::Model));

    // No unsupported extension, even though all supported names are declared.
    let model = assets.model(id).expect("the model has a node graph");
    assert!(
        model.unsupported_extensions.is_empty(),
        "{:?}",
        model.unsupported_extensions
    );

    let material = assets
        .mesh(id)
        .and_then(|mesh| mesh.material.as_ref())
        .expect("the primitive names material 0");
    assert_eq!(material.uv_transforms.len(), MATERIAL_TEXTURE_SLOT_COUNT);

    // The base-colour map decoded, and its transform reached the albedo slot.
    assert!(
        material.albedo.is_some(),
        "base-colour texture did not decode"
    );
    let albedo = material.uv_transforms[texture_slot::ALBEDO];
    assert!(albedo.has_transform);
    assert_eq!(albedo.scale, [2.0, 3.0]);
    // cos(0)=1, sin(0)=0: offset.x unchanged, offset.y becomes 1 - scale.y - offset.y.
    assert!(close(albedo.offset[0], 0.25));
    assert!(close(albedo.offset[1], 1.0 - 3.0 - 0.5));

    let ext = &material.extensions;
    assert!(close(ext.emissive_strength, 2.5));
    assert!(close(ext.ior, 1.45));
    assert!(ext.unlit);

    assert!(ext.has_clearcoat);
    assert!(close(ext.clearcoat_factor, 0.3));
    assert!(close(ext.clearcoat_roughness_factor, 0.1));
    assert!(
        ext.clearcoat_texture.is_some(),
        "clearcoat texture did not decode"
    );

    assert!(ext.has_specular);
    assert!(close(ext.specular_factor, 0.4));
    assert_eq!(ext.specular_color_factor, [0.9, 0.8, 0.7]);

    assert!(ext.has_sheen);
    assert_eq!(ext.sheen_color_factor, [0.1, 0.2, 0.3]);
    assert!(close(ext.sheen_roughness_factor, 0.6));

    assert!(ext.has_transmission);
    assert!(close(ext.transmission_factor, 0.75));
    assert!(close(ext.thickness_factor, 0.2));
    assert!(close(ext.attenuation_distance, 3.0));
    assert_eq!(ext.attenuation_color, [0.5, 0.6, 0.7]);
    // A non-zero transmission factor composites after opaques.
    assert_eq!(ext.transmission_blend, BlendApproach::Blend);

    assert!(ext.has_iridescence);
    assert!(close(ext.iridescence_factor, 0.5));
    assert!(close(ext.iridescence_ior, 1.4));
    assert!(close(ext.iridescence_thickness_minimum, 120.0));
    assert!(close(ext.iridescence_thickness_maximum, 350.0));

    assert!(ext.has_anisotropy);
    assert!(close(ext.anisotropy_strength, 0.8));
    assert!(close(ext.anisotropy_rotation, 0.25));

    assert!(ext.has_diffuse_transmission);
    assert!(close(ext.diffuse_transmission_factor, 0.35));
    assert_eq!(ext.diffuse_transmission_color, [0.2, 0.3, 0.4]);

    // Specular-glossiness: the factors reach shading instead of being dropped.
    assert!(ext.has_specular_glossiness);
    assert_eq!(ext.sg_diffuse_factor, [0.1, 0.2, 0.3, 0.4]);
    assert_eq!(ext.sg_specular_factor, [0.5, 0.6, 0.7]);
    assert!(close(ext.sg_glossiness_factor, 0.85));
}

/// A material with no extension at all: the defaults must be the spec's, not
/// zero. A zeroed roughness or attenuation distance changes how every surface
/// renders.
const PLAIN_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ { "attributes": { "POSITION": 0 }, "material": 0 } ]
  } ],
  "materials": [ { "name": "Plain" } ],
  "buffers": [ { "uri": "triangle.bin", "byteLength": 36 } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;

#[test]
fn a_material_without_extensions_gets_spec_defaults() {
    let scratch = Scratch::new("plain");
    scratch.write("model.gltf", PLAIN_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let id = assets.load("model.gltf").expect("the model loads");
    let material = assets
        .mesh(id)
        .and_then(|mesh| mesh.material.as_ref())
        .expect("the primitive names material 0");

    let ext = &material.extensions;
    assert!(!ext.has_clearcoat);
    assert!(!ext.unlit);
    assert!(close(ext.emissive_strength, 1.0));
    assert!(close(ext.ior, 1.5));
    assert!(ext.attenuation_distance.is_infinite());
    assert!(close(ext.iridescence_ior, 1.3));
    assert!(close(ext.iridescence_thickness_minimum, 100.0));
    assert!(close(ext.iridescence_thickness_maximum, 400.0));
    assert_eq!(
        ext.transmission_blend,
        BlendApproach::OpaqueWithTransmission
    );

    for transform in &material.uv_transforms {
        assert!(!transform.has_transform);
        assert_eq!(transform.scale, [1.0, 1.0]);
    }

    let model = assets.model(id).expect("the model has a node graph");
    assert!(model.unsupported_extensions.is_empty());
    assert!(model.punctual_lights.is_empty());
}

/// A primitive that actually applies Draco: the geometry cannot be read by
/// reading the buffer, so the load must be refused, not produce empty geometry.
const DRACO_APPLIED_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": [ "KHR_draco_mesh_compression" ],
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ {
      "attributes": { "POSITION": 0 },
      "extensions": { "KHR_draco_mesh_compression": {
        "bufferView": 1, "attributes": { "POSITION": 0 }
      } }
    } ]
  } ],
  "buffers": [ { "uri": "triangle.bin", "byteLength": 36 } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;

#[test]
fn an_applied_draco_primitive_is_refused_naming_the_extension() {
    let scratch = Scratch::new("draco-applied");
    scratch.write("model.gltf", DRACO_APPLIED_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let error = assets.load("model.gltf").expect_err("Draco is refused");
    match error {
        Error::Unsupported(message) => {
            assert!(message.contains("KHR_draco_mesh_compression"), "{message}");
            assert!(message.contains("decoder"), "{message}");
        }
        other => panic!("expected Unsupported, got {other}"),
    }
}

/// Draco declared but never applied: the geometry is plain, so the load
/// succeeds, but the declaration is logged rather than silently dropped.
const DRACO_DECLARED_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": [ "KHR_draco_mesh_compression" ],
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ { "attributes": { "POSITION": 0 } } ]
  } ],
  "buffers": [ { "uri": "triangle.bin", "byteLength": 36 } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;

#[test]
fn a_declared_but_unapplied_compression_extension_still_loads() {
    let scratch = Scratch::new("draco-declared");
    scratch.write("model.gltf", DRACO_DECLARED_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let id = assets.load("model.gltf").expect("plain geometry loads");
    let model = assets.model(id).expect("the model has a node graph");
    assert!(model.unsupported_extensions.is_empty());
}

/// An extension this loader neither implements nor needs, required: the file
/// cannot be correct without it, so the load errors naming it.
const REQUIRED_UNKNOWN_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": [ "KHR_materials_variants" ],
  "extensionsRequired": [ "KHR_materials_variants" ],
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node" } ]
}"#;

#[test]
fn a_required_unimplemented_extension_is_refused_by_name() {
    let scratch = Scratch::new("required");
    scratch.write("model.gltf", REQUIRED_UNKNOWN_GLTF.as_bytes());

    let mut assets = scratch.assets();
    let error = assets
        .load("model.gltf")
        .expect_err("a required extension is refused");
    match error {
        Error::Unsupported(message) => {
            assert!(message.contains("KHR_materials_variants"), "{message}");
            assert!(message.contains("required"), "{message}");
        }
        other => panic!("expected Unsupported, got {other}"),
    }
}

/// An unknown used (but not required) extension: ignored, but recorded so the
/// omission is not silent.
const UNKNOWN_USED_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": [ "KHR_materials_variants" ],
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node" } ]
}"#;

#[test]
fn an_ignored_extension_is_recorded_rather_than_silent() {
    let scratch = Scratch::new("unknown-used");
    scratch.write("model.gltf", UNKNOWN_USED_GLTF.as_bytes());

    let mut assets = scratch.assets();
    let id = assets
        .load("model.gltf")
        .expect("an unknown used extension still loads");
    let model = assets.model(id).expect("the model has a node graph");
    assert_eq!(model.unsupported_extensions, vec!["KHR_materials_variants"]);
}

/// Three lights and two nodes: one within range, one pointing past the list.
const LIGHTS_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": [ "KHR_lights_punctual" ],
  "extensions": { "KHR_lights_punctual": { "lights": [
    { "type": "point", "color": [1.0, 0.0, 0.0], "intensity": 2.0, "range": 5.0 },
    { "type": "directional", "intensity": 0.5 },
    { "type": "spot", "spot": { "innerConeAngle": 0.1, "outerConeAngle": 0.5 } }
  ] } },
  "scene": 0,
  "scenes": [ { "nodes": [ 0, 1 ] } ],
  "nodes": [
    { "name": "Lamp", "extensions": { "KHR_lights_punctual": { "light": 0 } } },
    { "name": "Dangling", "extensions": { "KHR_lights_punctual": { "light": 9 } } }
  ]
}"#;

#[test]
fn punctual_lights_are_parsed_and_bound_to_their_nodes() {
    let scratch = Scratch::new("lights");
    scratch.write("model.gltf", LIGHTS_GLTF.as_bytes());

    let mut assets = scratch.assets();
    let id = assets.load("model.gltf").expect("the model loads");
    let model = assets.model(id).expect("the model has a node graph");
    assert!(model.unsupported_extensions.is_empty());

    assert_eq!(model.punctual_lights.len(), 3);
    let point = &model.punctual_lights[0];
    assert_eq!(point.light_type, LightType::Point);
    assert_eq!(point.color, [1.0, 0.0, 0.0]);
    assert!(close(point.intensity, 2.0));
    assert!(close(point.range, 5.0));

    assert_eq!(model.punctual_lights[1].light_type, LightType::Directional);
    // A directional light has no range.
    assert!(model.punctual_lights[1].range.is_infinite());

    let spot = &model.punctual_lights[2];
    assert_eq!(spot.light_type, LightType::Spot);
    assert!(close(spot.inner_cone_angle, 0.1));
    assert!(close(spot.outer_cone_angle, 0.5));

    assert_eq!(model.nodes[0].light, 0);
    // The out-of-range reference is refused rather than stored raw.
    assert_eq!(model.nodes[1].light, -1);
}

/// A meshopt-compressed buffer view cannot be handed to the accessor readers.
const MESHOPT_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsRequired": [ "EXT_meshopt_compression" ],
  "extensionsUsed": [ "EXT_meshopt_compression" ],
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ { "attributes": { "POSITION": 0 } } ]
  } ],
  "buffers": [
    { "uri": "triangle.bin", "byteLength": 36 },
    { "uri": "triangle.bin", "byteLength": 36 }
  ],
  "bufferViews": [ {
    "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962,
    "extensions": { "EXT_meshopt_compression": {
      "buffer": 1, "byteOffset": 0, "byteLength": 36, "byteStride": 12,
      "count": 3, "mode": "ATTRIBUTES"
    } }
  } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;

/// An extension value of the wrong type is a reported error, not a silent
/// default: a misspelled `emissiveStrength` must not render as 1.0.
const MALFORMED_EXTENSION_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": [ "KHR_materials_emissive_strength" ],
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ { "attributes": { "POSITION": 0 }, "material": 0 } ]
  } ],
  "materials": [ { "name": "Bad", "extensions": {
    "KHR_materials_emissive_strength": { "emissiveStrength": "bright" }
  } } ],
  "buffers": [ { "uri": "triangle.bin", "byteLength": 36 } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;

#[test]
fn a_malformed_extension_value_is_a_reported_error() {
    let scratch = Scratch::new("malformed");
    scratch.write("model.gltf", MALFORMED_EXTENSION_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let error = assets
        .load("model.gltf")
        .expect_err("a malformed value is refused");
    match error {
        Error::Decode(message) => {
            assert!(message.contains("emissiveStrength"), "{message}");
        }
        other => panic!("expected Decode, got {other}"),
    }
}

/// A rotation must survive the V-flip with the reflected sign. This is the
/// integration half of `uv`'s own unit test.
#[test]
fn a_texture_transform_rotation_is_stored_with_the_flipped_sign() {
    let scratch = Scratch::new("rotation");
    scratch.write("white.png", &white_png());
    scratch.write("triangle.bin", &triangle_bin());
    let json = EXTENDED_GLTF.replace("\"rotation\": 0.0", "\"rotation\": 1.57079632679");
    scratch.write("model.gltf", json.as_bytes());

    let mut assets = scratch.assets();
    let id = assets.load("model.gltf").expect("the model loads");
    let albedo = assets
        .mesh(id)
        .and_then(|mesh| mesh.material.as_ref())
        .map(|material| material.uv_transforms[texture_slot::ALBEDO])
        .expect("material 0 has an albedo transform");
    assert!(albedo.has_transform);
    assert!(albedo.cos_rotation.abs() < 1e-4, "{albedo:?}");
    assert!((albedo.sin_rotation + 1.0).abs() < 1e-4, "{albedo:?}");
}

/// Two materials that name the same image through different textures must share
/// one decode: the cache is keyed on the image, not the texture.
#[test]
fn two_textures_of_one_image_share_a_decode() {
    let scratch = Scratch::new("shared-image");
    scratch.write("white.png", &white_png());
    scratch.write("triangle.bin", &triangle_bin());
    let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [
      { "attributes": { "POSITION": 0 }, "material": 0 },
      { "attributes": { "POSITION": 0 }, "material": 1 }
    ]
  } ],
  "materials": [
    { "name": "A", "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 } } },
    { "name": "B", "pbrMetallicRoughness": { "baseColorTexture": { "index": 1 } } }
  ],
  "images": [ { "uri": "white.png" } ],
  "textures": [ { "source": 0 }, { "source": 0 } ],
  "buffers": [ { "uri": "triangle.bin", "byteLength": 36 } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;
    scratch.write("model.gltf", json.as_bytes());

    let mut assets = scratch.assets();
    let id = assets.load("model.gltf").expect("the model loads");
    let meshes = assets.meshes(id).expect("the model has meshes");
    let first = meshes[0]
        .material
        .as_ref()
        .and_then(|material| material.albedo.clone())
        .expect("material A has an albedo");
    let second = meshes[1]
        .material
        .as_ref()
        .and_then(|material| material.albedo.clone())
        .expect("material B has an albedo");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "two textures of one image did not share a decode"
    );
}

#[test]
fn a_meshopt_compressed_view_is_refused_with_a_reason() {
    let scratch = Scratch::new("meshopt");
    scratch.write("model.gltf", MESHOPT_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let error = assets.load("model.gltf").expect_err("the model is refused");
    match error {
        Error::Unsupported(message) => assert!(
            message.contains("meshopt"),
            "the reason should name the extension: {message}"
        ),
        other => panic!("expected Unsupported, got {other}"),
    }
}
