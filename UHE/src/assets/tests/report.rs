//! The load report: what a model read, skipped and substituted.

use std::fs;
use std::path::PathBuf;

use vault::{Assets, ExtensionClass, LoadReport};

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("vault-report-{}-{name}", std::process::id()));
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

/// Three positions, little-endian `f32`: (0,0,0), (1,0,0), (0,1,0).
fn triangle_bin() -> Vec<u8> {
    let mut bin = Vec::with_capacity(36);
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bin.extend_from_slice(&value.to_le_bytes());
    }
    bin
}

const CLEAN_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
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
fn a_clean_model_reports_what_it_loaded_and_nothing_else() {
    let scratch = Scratch::new("clean");
    scratch.write("clean.gltf", CLEAN_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let id = assets.load("clean.gltf").expect("the model loads");
    let report: &LoadReport = assets.report(id).expect("a model has a report");

    assert_eq!(report.source, "clean.gltf");
    assert_eq!(report.format, "gltf");
    assert_eq!(report.loaded.primitives, 1);
    assert_eq!(report.loaded.meshes, 1);
    assert_eq!(report.loaded.materials, 0);
    assert_eq!(report.loaded.textures, 0);
    assert_eq!(report.skipped, Default::default());
    assert!(report.extensions.is_empty());
    assert!(report.substitutions.is_empty());
    assert!(report.warnings.is_empty());
    assert!(report.is_clean());

    // The same report is on the payload itself, not parsed a second time.
    assert_eq!(report, &assets.model(id).expect("model").report);
}

/// Draco is declared but not applied: the geometry is plain, so the load
/// succeeds, and the report says the extension was refused.
const REFUSED_GLTF: &str = r#"{
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
fn a_refused_extension_is_classified_and_warned_about() {
    let scratch = Scratch::new("refused");
    scratch.write("model.gltf", REFUSED_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let id = assets
        .load("model.gltf")
        .expect("plain geometry still loads");
    let report = assets.report(id).expect("a model has a report");

    assert_eq!(report.extensions.len(), 1);
    let extension = &report.extensions[0];
    assert_eq!(extension.name, "KHR_draco_mesh_compression");
    assert_eq!(extension.classification, ExtensionClass::Refused);
    assert!(extension
        .reason
        .as_deref()
        .is_some_and(|why| why.contains("Draco")));
    assert!(
        report
            .warnings
            .iter()
            .any(|warning| warning.contains("refused")),
        "{:?}",
        report.warnings
    );
    assert_eq!(report.loaded.primitives, 1);
    assert!(!report.is_clean());
}

/// A material names an image that is not on disk: the model still loads, and
/// the report says a texture was substituted.
const MISSING_TEXTURE_GLTF: &str = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [ { "nodes": [ 0 ] } ],
  "nodes": [ { "name": "Node", "mesh": 0 } ],
  "meshes": [ {
    "name": "Tri",
    "primitives": [ { "attributes": { "POSITION": 0 }, "material": 0 } ]
  } ],
  "materials": [ {
    "name": "Missing",
    "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 } }
  } ],
  "images": [ { "uri": "missing.png" } ],
  "textures": [ { "source": 0 } ],
  "buffers": [ { "uri": "triangle.bin", "byteLength": 36 } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 } ],
  "accessors": [ {
    "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
    "min": [ 0.0, 0.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
  } ]
}"#;

#[test]
fn a_missing_texture_is_a_substitution_with_a_reason() {
    let scratch = Scratch::new("missing-texture");
    scratch.write("model.gltf", MISSING_TEXTURE_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let mut assets = scratch.assets();
    let id = assets
        .load("model.gltf")
        .expect("the model loads without the map");
    let report = assets.report(id).expect("a model has a report");

    assert_eq!(report.loaded.materials, 1);
    assert_eq!(report.loaded.textures, 0);
    assert_eq!(report.skipped.textures, 1);
    assert_eq!(report.substitutions.len(), 1);
    let substitution = &report.substitutions[0];
    assert!(
        substitution.subject.starts_with("texture"),
        "{}",
        substitution.subject
    );
    assert!(
        substitution.reason.contains("could not be read"),
        "{}",
        substitution.reason
    );
    assert!(!report.is_clean());
}
