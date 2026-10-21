//! The loaders: image decoding, KTX2 refusal for what is not portable, text.

use std::fs;
use std::path::{Path, PathBuf};

use grind::Jobs;
use serde_json::Value;
use vault::{AssetKind, AssetState, Assets, Error, Handle, MeshData};

struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("vault-loaders-{}-{name}", std::process::id()));
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
// The test builds its own bytes rather than carrying a fixture: a 2x1 image
// whose two texels differ is small enough to read, and the decode is checked
// against texels the test chose.
//
// The IDAT stream is a single *stored* deflate block, so no compressor is
// needed. PNG's chunk checksums are CRC-32 over type + data, and the zlib
// stream ends in an Adler-32 of the raw scanlines.

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

/// A non-interlaced, 8-bit RGBA PNG. `texels` is `width * height * 4` bytes.
fn png(width: u32, height: u32, texels: &[u8]) -> Vec<u8> {
    assert_eq!(texels.len(), width as usize * height as usize * 4);

    let mut raw = Vec::new();
    for row in texels.chunks_exact(width as usize * 4) {
        // Filter type 0: none.
        raw.push(0);
        raw.extend_from_slice(row);
    }

    let length = raw.len() as u16;
    let mut zlib = vec![0x78, 0x01];
    zlib.push(0x01); // final block, stored
    zlib.extend_from_slice(&length.to_le_bytes());
    zlib.extend_from_slice(&(!length).to_le_bytes());
    zlib.extend_from_slice(&raw);
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, colour type 6 (RGBA)

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    out.extend(chunk(b"IHDR", &header));
    out.extend(chunk(b"IDAT", &zlib));
    out.extend(chunk(b"IEND", &[]));
    out
}

// ---- a KTX2 container with a Basis payload ----------------------------------

const KTX2_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// Identifier, nine `u32` header fields, two empty `u64` ranges, one level index.
fn ktx2_basis_payload() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&KTX2_IDENTIFIER);
    // vkFormat VK_FORMAT_UNDEFINED, 4x4, one layer, one face, one level, and
    // supercompressionScheme BasisLZ - which is how basisu writes ETC1S.
    for value in [0u32, 1, 4, 4, 0, 0, 1, 1, 1] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[0u8; 32]);
    assert_eq!(bytes.len(), 80);
    bytes.extend_from_slice(&(104u64).to_le_bytes());
    bytes.extend_from_slice(&(32u64).to_le_bytes());
    bytes.extend_from_slice(&(32u64).to_le_bytes());
    bytes.extend_from_slice(&[0u8; 32]);
    bytes
}

/// Three positions, little-endian `f32`, and the smallest glTF that uses them.
fn triangle_bin() -> Vec<u8> {
    let mut bin = Vec::with_capacity(36);
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        bin.extend_from_slice(&value.to_le_bytes());
    }
    bin
}

const MODEL_GLTF: &str = r#"{
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
fn an_async_model_load_is_pending_until_polled_then_resolves() {
    let scratch = Scratch::new("async-mesh");
    scratch.write("model.gltf", MODEL_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let jobs = Jobs::new();
    let mut assets = scratch.assets();
    let id = assets
        .load_async(&jobs, "model.gltf")
        .expect("the path resolves");

    // The frame that asked draws without the asset: a valid handle exists, but
    // the payload does not resolve yet.
    assert_eq!(assets.state(id), Some(AssetState::Pending));
    assert!(assets.mesh(id).is_none());
    let handle: Handle<MeshData> = assets.handle(id).expect("a model has a mesh handle");
    assert!(handle.get(&assets).is_none());
    assert_eq!(handle.state(&assets), Some(AssetState::Pending));

    jobs.wait_for_all();
    let finished = assets.poll();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].0, id);
    assert!(matches!(finished[0].1, Ok(AssetKind::Model)));

    // The asset appears when the worker is done.
    assert_eq!(assets.state(id), Some(AssetState::Ready));
    assert_eq!(handle.state(&assets), Some(AssetState::Ready));
    let mesh = handle
        .get(&assets)
        .expect("the mesh resolves after the load");
    assert_eq!(mesh.positions.len(), 3);
}

#[test]
fn releasing_a_pending_load_cancels_its_delivery() {
    let scratch = Scratch::new("async-cancel");
    scratch.write("model.gltf", MODEL_GLTF.as_bytes());
    scratch.write("triangle.bin", &triangle_bin());

    let jobs = Jobs::new();
    let mut assets = scratch.assets();
    let id = assets
        .load_async(&jobs, "model.gltf")
        .expect("the path resolves");
    assert_eq!(assets.state(id), Some(AssetState::Pending));

    // Closing the scene releases the id; the worker is told to drop the load.
    assert!(assets.release(id));
    jobs.wait_for_all();
    assert!(assets.poll().is_empty(), "a cancelled load was delivered");
    assert!(!assets.contains(id));
    assert_eq!(assets.state(id), None);
}

#[test]
fn a_png_decodes_to_its_extent_and_rgba_texels() {
    let scratch = Scratch::new("png");
    let texels: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    scratch.write("texel.png", &png(2, 1, &texels));

    let mut assets = scratch.assets();
    let id = assets.load("texel.png").expect("the png loads");
    assert_eq!(assets.kind(id), Some(AssetKind::Image));

    let image = assets.image(id).expect("a png has an image");
    assert_eq!((image.width, image.height), (2, 1));
    assert_eq!(image.rgba8, texels);
    assert_eq!(image.rgba8.len(), image.byte_size());
    // The encoded file is kept too, which is what `bytes` is for.
    assert_eq!(
        assets.bytes(id).map(|bytes| &bytes[..4]),
        Some(b"\x89PNG".as_slice())
    );
}

#[test]
fn a_file_that_is_not_an_image_is_an_error_not_a_silent_binary() {
    let scratch = Scratch::new("bad-png");
    scratch.write("broken.png", b"this is not a png");

    let mut assets = scratch.assets();
    let error = assets
        .load("broken.png")
        .expect_err("a broken png is refused");
    assert!(matches!(error, Error::Decode(_)), "{error}");
    assert!(assets.is_empty());
}

#[test]
fn a_ktx2_with_a_basis_payload_is_refused_and_says_why() {
    let scratch = Scratch::new("ktx2-basis");
    scratch.write("wall.ktx2", &ktx2_basis_payload());

    let mut assets = scratch.assets();
    let error = assets.load("wall.ktx2").expect_err("Basis is refused");
    assert!(matches!(error, Error::Unsupported(_)), "{error}");

    let message = error.to_string();
    assert!(message.contains("BasisLZ"), "{message}");
    assert!(message.contains("Basis transcoder"), "{message}");
    assert!(assets.is_empty());
}

/// The atlas a renderer is handed is pre-baked: an `msdf-atlas-gen` JSON and
/// the PNG it packed. `msdf-atlas-gen` is not ported
/// (`docs/architecture/01-crate-map.md`, "Not ported"), so the two things vault
/// owes a caller are the JSON as a text asset and the PNG as an image; the
/// glyph quads built from them are `bifrost::font::atlas`'s half. This reads the
/// atlas checked into the repository, so a change to its shape fails here.
#[test]
fn the_checked_in_msdf_atlas_json_loads_as_text_and_parses() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut assets = Assets::new(repo.join("assets/fonts"));
    let id = assets
        .load("test-atlas.json")
        .expect("the atlas json loads");

    assert_eq!(assets.kind(id), Some(AssetKind::Text));
    let text = assets.text(id).expect("json is text");
    assert!(text.contains("msdf"));
    assert!(assets.image(id).is_none());

    // msdf-atlas-gen's own shape, field for field: the metrics, the atlas
    // extent and the glyph quads are what a consumer reads off it.
    let parsed: Value = serde_json::from_str(text).expect("the atlas json parses");
    assert_eq!(parsed["atlas"]["type"], "msdf");
    assert_eq!(parsed["atlas"]["width"], 64);
    assert_eq!(parsed["atlas"]["distanceRange"], 2);
    assert_eq!(parsed["metrics"]["lineHeight"], 2.0);
    assert_eq!(parsed["kerning"][0]["advance"], -0.2);
    let glyphs = parsed["glyphs"]
        .as_array()
        .expect("the atlas lists its glyphs");
    let a = glyphs
        .iter()
        .find(|glyph| glyph["unicode"] == 65)
        .expect("the atlas carries an A");
    assert_eq!(a["atlasBounds"]["right"], 12);
}

#[test]
fn an_atlas_png_next_to_its_json_is_an_image_asset() {
    let scratch = Scratch::new("atlas-png");
    scratch.write("inter-msdf.png", &png(2, 1, &[0, 1, 2, 255, 3, 4, 5, 255]));

    let mut assets = scratch.assets();
    let png_id = assets.load("inter-msdf.png").expect("the atlas png loads");
    let image = assets.image(png_id).expect("the atlas png decodes");
    assert_eq!((image.width, image.height), (2, 1));
    assert_eq!(image.rgba8, vec![0, 1, 2, 255, 3, 4, 5, 255]);
    // A standalone image is colour data, so it is sRGB by default.
    assert!(image.srgb);
}

#[test]
fn an_unknown_text_file_is_text_only_when_its_extension_says_so() {
    let scratch = Scratch::new("text");
    scratch.write("notes.txt", b"hello");
    scratch.write("blob.bin", b"hello");

    let mut assets = scratch.assets();
    let text = assets.load("notes.txt").expect("notes.txt loads");
    let binary = assets.load("blob.bin").expect("blob.bin loads");

    assert_eq!(assets.text(text), Some("hello"));
    assert_eq!(assets.text(binary), None);
    assert_eq!(assets.bytes(binary), Some(b"hello".as_slice()));
}

#[test]
fn an_async_load_is_reported_once_and_never_again() {
    let scratch = Scratch::new("async");
    scratch.write("broken.png", b"not a png at all");

    let jobs = Jobs::new();
    let mut assets = scratch.assets();
    let id = assets
        .load_async(&jobs, "broken.png")
        .expect("the path resolves");
    // Nothing is decoded yet, so nothing resolves yet.
    assert!(assets.image(id).is_none());

    jobs.wait_for_all();
    let finished = assets.poll();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].0, id);
    assert!(matches!(finished[0].1, Err(Error::Decode(_))));

    // The failure is delivered once: the slot is released, so a second poll
    // has nothing to hand out and the id no longer resolves.
    assert!(assets.poll().is_empty());
    assert!(assets.poll().is_empty());
    assert!(!assets.contains(id));
}

#[test]
fn an_async_load_of_a_good_image_lands_on_poll() {
    let scratch = Scratch::new("async-ok");
    scratch.write("texel.png", &png(1, 1, &[9, 8, 7, 255]));

    let jobs = Jobs::new();
    let mut assets = scratch.assets();
    let id = assets
        .load_async(&jobs, "texel.png")
        .expect("path resolves");

    jobs.wait_for_all();
    let finished = assets.poll();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].0, id);
    assert!(matches!(finished[0].1, Ok(AssetKind::Image)));
    assert_eq!(
        assets.image(id).map(|image| image.rgba8.clone()),
        Some(vec![9, 8, 7, 255])
    );
}

#[test]
fn a_second_async_request_for_a_ready_asset_is_reported_on_poll() {
    let scratch = Scratch::new("async-again");
    scratch.write("notes.txt", b"hi");

    let jobs = Jobs::new();
    let mut assets = scratch.assets();
    let first = assets.load("notes.txt").expect("notes.txt loads");
    let again = assets
        .load_async(&jobs, "notes.txt")
        .expect("path resolves");

    assert_eq!(first, again);
    let finished = assets.poll();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].0, first);
    assert!(matches!(finished[0].1, Ok(AssetKind::Text)));
}
