//! Shared headless device, scene fixtures and readback helpers for the engine
//! render tests.

#![allow(dead_code)]

use std::path::PathBuf;

use bifrost::renderer::CLEAR_COLOR;
use bifrost::{Bifrost, EditorCamera, FrameImage, ViewSource};
use chassis::backend::{BackendChoice, BackendSelection, BackendSource};
use chassis::Gpu;
use engine::{create_gpu, GpuRequest};
use glam::{Mat4, Vec3};
use meats::Scene;
use vault::Assets;

pub const WIDTH: u32 = 64;
pub const HEIGHT: u32 = 64;

/// Frames to render before the readback has a retired submission to return.
/// Larger than any ring depth the renderer has used, so the tests do not depend
/// on that implementation constant.
pub const READBACK_WARMUP: usize = 4;

pub fn extent() -> chassis::Extent2d {
    chassis::Extent2d {
        width: WIDTH,
        height: HEIGHT,
    }
}

/// A scratch directory for the tiny glTFs, removed when the harness drops.
pub struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        // Each test binary owns its own tag and each open adds a unique suffix,
        // so one backend cannot delete another's assets mid-load.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "engine-render-{}-{tag}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("scratch directory");
        Self { root }
    }

    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// One indexed primitive with positions and normals, written as a `.bin`
    /// beside a `.gltf` that names it.
    pub fn write_indexed_mesh(
        &self,
        name: &str,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u16],
    ) {
        assert_eq!(
            positions.len(),
            normals.len(),
            "{name}: position/normal count"
        );
        assert!(!positions.is_empty(), "{name}: no vertices");

        let mut bin = Vec::new();
        for position in positions {
            for value in position {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }
        for normal in normals {
            for value in normal {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }
        for index in indices {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        std::fs::write(self.root.join(format!("{name}.bin")), &bin).expect("mesh buffer");

        let position_bytes = positions.len() * 12;
        let normal_offset = position_bytes;
        let index_offset = position_bytes * 2;
        let byte_length = index_offset + indices.len() * 2;

        let mut min = positions[0];
        let mut max = positions[0];
        for position in positions {
            for axis in 0..3 {
                min[axis] = min[axis].min(position[axis]);
                max[axis] = max[axis].max(position[axis]);
            }
        }

        let gltf = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [ {{ "nodes": [ 0 ] }} ],
  "nodes": [ {{ "name": "{name}", "mesh": 0 }} ],
  "meshes": [ {{
    "name": "{name}",
    "primitives": [ {{
      "attributes": {{ "POSITION": 0, "NORMAL": 1 }},
      "indices": 2
    }} ]
  }} ],
  "buffers": [ {{ "uri": "{name}.bin", "byteLength": {byte_length} }} ],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": {position_bytes}, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {normal_offset}, "byteLength": {position_bytes}, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {index_offset}, "byteLength": {}, "target": 34963 }}
  ],
  "accessors": [
    {{
      "bufferView": 0, "componentType": 5126, "count": {}, "type": "VEC3",
      "min": [{}, {}, {}], "max": [{}, {}, {}]
    }},
    {{ "bufferView": 1, "componentType": 5126, "count": {}, "type": "VEC3" }},
    {{ "bufferView": 2, "componentType": 5123, "count": {}, "type": "SCALAR" }}
  ]
}}"#,
            indices.len() * 2,
            positions.len(),
            min[0],
            min[1],
            min[2],
            max[0],
            max[1],
            max[2],
            normals.len(),
            indices.len(),
        );
        std::fs::write(self.root.join(format!("{name}.gltf")), gltf).expect("mesh glTF");
    }

    /// Two triangles with opposite winding, so one survives either front-face
    /// convention. Both carry +Z normals, so whichever is drawn is lit.
    pub fn write_triangle(&self) {
        let positions = [
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, -2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let normals = [[0.0, 0.0, 1.0]; 6];
        let indices = [0u16, 1, 2, 3, 4, 5];
        self.write_indexed_mesh("tri", &positions, &normals, &indices);
    }

    /// A two-bone skinned quad and a one-second clip that moves the right half.
    ///
    /// The mesh is a plain XY quad at z = 0: its left vertices bind to BoneA and
    /// its right vertices to BoneB. BoneB's clip translation runs from its rest
    /// value to one unit further along +X, so a skinned frame stretches the quad.
    pub fn write_skinned_quad(&self, name: &str) {
        let positions = [
            [-1.0f32, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let joints = [[0u16, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]];
        let weights = [
            [1.0f32, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];
        let indices = [0u16, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2];
        let inverse_bind = [
            [
                1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0,
            ],
        ];
        let times = [0.0f32, 1.0];
        let translations = [[1.0f32, 0.0, 0.0], [2.0, 0.0, 0.0]];

        let mut bin = Vec::new();
        let mut append = |bytes: Vec<u8>| {
            let offset = bin.len();
            bin.extend_from_slice(&bytes);
            offset
        };
        let position_offset = append(f32_bytes(positions.iter().flatten()));
        let normal_offset = append(f32_bytes(normals.iter().flatten()));
        let joint_offset = append(u16_bytes(joints.iter().flatten()));
        let weight_offset = append(f32_bytes(weights.iter().flatten()));
        let index_offset = append(u16_bytes(indices.iter()));
        let bind_offset = append(f32_bytes(inverse_bind.iter().flatten()));
        let time_offset = append(f32_bytes(times.iter()));
        let translation_offset = append(f32_bytes(translations.iter().flatten()));
        let byte_length = bin.len();
        std::fs::write(self.root.join(format!("{name}.bin")), &bin).expect("skin buffer");

        let gltf = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [ {{ "nodes": [ 0, 3 ] }} ],
  "nodes": [
    {{ "name": "Root", "children": [ 1 ] }},
    {{ "name": "BoneA", "children": [ 2 ] }},
    {{ "name": "BoneB", "translation": [ 1.0, 0.0, 0.0 ] }},
    {{ "name": "{name}", "mesh": 0, "skin": 0 }}
  ],
  "meshes": [ {{
    "name": "{name}",
    "primitives": [ {{
      "attributes": {{ "POSITION": 0, "NORMAL": 1, "JOINTS_0": 2, "WEIGHTS_0": 3 }},
      "indices": 4
    }} ]
  }} ],
  "skins": [ {{
    "name": "Skin", "inverseBindMatrices": 5, "joints": [ 1, 2 ], "skeleton": 0
  }} ],
  "animations": [ {{
    "name": "Bend",
    "samplers": [ {{ "input": 6, "output": 7, "interpolation": "LINEAR" }} ],
    "channels": [ {{ "sampler": 0, "target": {{ "node": 2, "path": "translation" }} }} ]
  }} ],
  "buffers": [ {{ "uri": "{name}.bin", "byteLength": {byte_length} }} ],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {position_offset}, "byteLength": 48, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {normal_offset}, "byteLength": 48, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {joint_offset}, "byteLength": 32, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {weight_offset}, "byteLength": 64, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {index_offset}, "byteLength": 24, "target": 34963 }},
    {{ "buffer": 0, "byteOffset": {bind_offset}, "byteLength": 128 }},
    {{ "buffer": 0, "byteOffset": {time_offset}, "byteLength": 8 }},
    {{ "buffer": 0, "byteOffset": {translation_offset}, "byteLength": 24 }}
  ],
  "accessors": [
    {{
      "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
      "min": [ -1.0, -1.0, 0.0 ], "max": [ 1.0, 1.0, 0.0 ]
    }},
    {{ "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3" }},
    {{ "bufferView": 2, "componentType": 5123, "count": 4, "type": "VEC4" }},
    {{ "bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC4" }},
    {{ "bufferView": 4, "componentType": 5123, "count": 12, "type": "SCALAR" }},
    {{ "bufferView": 5, "componentType": 5126, "count": 2, "type": "MAT4" }},
    {{
      "bufferView": 6, "componentType": 5126, "count": 2, "type": "SCALAR",
      "min": [ 0.0 ], "max": [ 1.0 ]
    }},
    {{ "bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC3" }}
  ]
}}"#
        );
        std::fs::write(self.root.join(format!("{name}.gltf")), gltf).expect("skin glTF");
    }

    /// A horizontal quad at y = 0, drawn from either side.
    pub fn write_plane(&self, name: &str, half: f32) {
        let positions = [
            [-half, 0.0, -half],
            [half, 0.0, -half],
            [half, 0.0, half],
            [-half, 0.0, half],
        ];
        let normals = [[0.0, 1.0, 0.0]; 4];
        let indices = [
            0u16, 2, 1, 0, 3, 2, // facing +Y
            0, 1, 2, 0, 2, 3, // facing -Y
        ];
        self.write_indexed_mesh(name, &positions, &normals, &indices);
    }

    /// A vertical quad in the XY plane at `z`, drawn from either side.
    pub fn write_wall(&self, name: &str, half: f32, z: f32) {
        let positions = [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ];
        let normals = [[0.0, 0.0, -1.0]; 4];
        let indices = [
            0u16, 2, 1, 0, 3, 2, // facing -Z
            0, 1, 2, 0, 2, 3, // facing +Z
        ];
        self.write_indexed_mesh(name, &positions, &normals, &indices);
    }

    /// An axis-aligned cube centred on its origin, with every face drawn in both
    /// windings so the depth prepass can cull either way.
    pub fn write_cube(&self, name: &str, half: f32) {
        let h = half;
        let positions = [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ];
        let normals: Vec<[f32; 3]> = positions
            .iter()
            .map(|position| Vec3::from_array(*position).normalize().to_array())
            .collect();
        let quads = [
            [0u16, 3, 2, 1], // -Z
            [4, 5, 6, 7],    // +Z
            [0, 4, 7, 3],    // -X
            [1, 2, 6, 5],    // +X
            [0, 1, 5, 4],    // -Y
            [3, 7, 6, 2],    // +Y
        ];
        let mut indices = Vec::new();
        for [a, b, c, d] in quads {
            indices.extend_from_slice(&[a, b, c, a, c, d]);
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
        self.write_indexed_mesh(name, &positions, &normals, &indices);
    }
}

fn f32_bytes<'a>(values: impl Iterator<Item = &'a f32>) -> Vec<u8> {
    values.flat_map(|value| value.to_le_bytes()).collect()
}

fn u16_bytes<'a>(values: impl Iterator<Item = &'a u16>) -> Vec<u8> {
    values.flat_map(|value| value.to_le_bytes()).collect()
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A renderer, a scene and the device they run on.
pub struct Harness {
    pub context: String,
    pub gpu: Box<dyn Gpu>,
    pub renderer: Bifrost,
    pub scene: Scene,
    pub assets: Assets,
    pub jobs: grind::Jobs,
    pub camera: EditorCamera,
    _scratch: Scratch,
}

impl Harness {
    /// Opens a backend, then asks `build` for the scene and camera to render.
    ///
    /// `None` only when no device opened; a device that opens and then cannot
    /// render is a failure, not a skip.
    pub fn open<F>(choice: BackendChoice, tag: &str, build: F) -> Option<Self>
    where
        F: FnOnce(&str, &Scratch, &mut Assets) -> (Scene, EditorCamera),
    {
        let mut gpu = match create_gpu(GpuRequest {
            selection: BackendSelection {
                choice,
                source: BackendSource::Default,
            },
            engine_name: c"unhuman-engine-render",
            extent: extent(),
        }) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("SKIP: {choice:?}: {error}");
                return None;
            }
        };
        let context = format!("{choice:?}").to_lowercase();

        let scratch = Scratch::new(&format!("{context}-{tag}"));
        let mut assets = Assets::new(scratch.root());
        let (scene, camera) = build(&context, &scratch, &mut assets);

        let mut renderer = Bifrost::new(&mut *gpu, &mut assets)
            .unwrap_or_else(|error| panic!("{context}: Bifrost::new: {error}"));
        for sample in scene.meshes() {
            renderer
                .ensure_mesh(&mut *gpu, &assets, sample.mesh)
                .unwrap_or_else(|error| panic!("{context}: upload mesh: {error}"));
        }

        Some(Self {
            context,
            gpu,
            renderer,
            scene,
            assets,
            jobs: grind::Jobs::new(),
            camera,
            _scratch: scratch,
        })
    }

    /// Turns the grid off without leaving the editor pass with no recorder.
    ///
    /// The renderer refuses a declared pass with no draw behind it, so a line
    /// placed behind the camera keeps the pass alive and invisible.
    pub fn hide_grid(&mut self) {
        use bifrost::passes::LineVertex;
        self.renderer.show_grid(false);
        let vertices = [
            LineVertex {
                position: [0.0, -1000.0, 0.0],
                color: [0.0, 0.0, 0.0, 1.0],
            },
            LineVertex {
                position: [1.0, -1000.0, 0.0],
                color: [0.0, 0.0, 0.0, 1.0],
            },
        ];
        self.renderer
            .set_lines(&mut *self.gpu, &vertices, &[0, 1])
            .unwrap_or_else(|error| panic!("{}: offscreen line: {error}", self.context));
    }

    pub fn render(&mut self, frames: usize) {
        for frame in 0..frames {
            self.renderer
                .render(
                    &mut *self.gpu,
                    &self.jobs,
                    &self.scene,
                    ViewSource::Editor,
                    Some(&self.camera),
                )
                .unwrap_or_else(|error| panic!("{}: render frame {frame}: {error}", self.context));
        }
    }

    pub fn image(&self) -> FrameImage {
        self.renderer
            .frame_image()
            .unwrap_or_else(|| panic!("{}: no readback frame after rendering", self.context))
            .clone()
    }

    /// The device is the authority, so confirm the readback matches what the
    /// renderer was asked to draw.
    pub fn check_extent(&self, image: &FrameImage) {
        assert_eq!(
            (image.width, image.height),
            (chassis::copy_row_pitch(WIDTH, 4) / 4, HEIGHT),
            "{}: readback extent",
            self.context
        );
        assert_eq!(
            image.bgra.len(),
            image.width as usize * image.height as usize * 4,
            "{}: packed readback",
            self.context
        );
    }

    pub fn destroy(mut self) {
        // `Bifrost::destroy` retires the frame ring's submissions itself, so a
        // caller does not hand the GPU resources its frames still read.
        self.renderer.destroy(&mut *self.gpu);
    }
}

/// The bytes an sRGB target holds for a linear clear value.
pub fn srgb8(linear: f32) -> u8 {
    let encoded = if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

/// `Bgra8Srgb` order, which is what the headless composition target uses.
pub fn clear_bgra() -> [u8; 4] {
    [
        srgb8(CLEAR_COLOR[2]),
        srgb8(CLEAR_COLOR[1]),
        srgb8(CLEAR_COLOR[0]),
        255,
    ]
}

pub fn pixel(image: &FrameImage, x: u32, y: u32) -> [u8; 4] {
    let offset = (y as usize * image.width as usize + x as usize) * 4;
    image.bgra[offset..offset + 4]
        .try_into()
        .expect("four bytes per pixel")
}

/// Pixels that are not the clear colour, with a small allowance for rounding.
pub fn drawn_pixels(image: &FrameImage) -> usize {
    let clear = clear_bgra();
    image
        .bgra
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| {
            pixel
                .iter()
                .zip(clear)
                .any(|(actual, expected)| actual.abs_diff(expected) > 2)
        })
        .count()
}

/// Mean absolute difference, largest channel difference, and the fraction of
/// channels past `outlier_delta`.
pub fn difference(a: &FrameImage, b: &FrameImage, outlier_delta: u8) -> (f32, u8, f32) {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "two frames of the same scene differ in size"
    );
    let mut total = 0u64;
    let mut max = 0u8;
    let mut outliers = 0usize;
    for (left, right) in a.bgra.iter().zip(&b.bgra) {
        let delta = left.abs_diff(*right);
        total += delta as u64;
        max = max.max(delta);
        outliers += usize::from(delta > outlier_delta);
    }
    let mean = total as f32 / a.bgra.len().max(1) as f32;
    let fraction = outliers as f32 / a.bgra.len().max(1) as f32;
    (mean, max, fraction)
}

/// Relative luminance of one BGRA sRGB texel, in 0..=255.
pub fn luminance(pixel: [u8; 4]) -> f32 {
    0.0722 * f32::from(pixel[0]) + 0.7152 * f32::from(pixel[1]) + 0.2126 * f32::from(pixel[2])
}

/// The mean luminance of a square region, clamped at the image's edges.
pub fn region_luminance(image: &FrameImage, centre: (u32, u32), radius: u32) -> f32 {
    let x0 = centre.0.saturating_sub(radius);
    let y0 = centre.1.saturating_sub(radius);
    let x1 = (centre.0 + radius).min(image.width - 1);
    let y1 = (centre.1 + radius).min(image.height - 1);
    let mut total = 0.0;
    let mut count = 0u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            total += luminance(pixel(image, x, y));
            count += 1;
        }
    }
    total / count.max(1) as f32
}

/// Where `with` is darker than `without`, for a test that has to find the region
/// an effect landed in rather than assume a pixel.
pub struct Darker {
    pub count: usize,
    pub bounds: ((u32, u32), (u32, u32)),
    pub centroid: (f32, f32),
    pub max_drop: f32,
    pub max_at: (u32, u32),
}

pub fn darker_than(with: &FrameImage, without: &FrameImage, threshold: f32) -> Darker {
    assert_eq!((with.width, with.height), (without.width, without.height));
    let mut count = 0usize;
    let mut min = (with.width, with.height);
    let mut max = (0u32, 0u32);
    let mut sum = (0.0f32, 0.0f32);
    let mut max_drop = 0.0f32;
    let mut max_at = (0u32, 0u32);
    for y in 0..with.height {
        for x in 0..with.width {
            let drop = luminance(pixel(without, x, y)) - luminance(pixel(with, x, y));
            if drop > max_drop {
                max_drop = drop;
                max_at = (x, y);
            }
            if drop > threshold {
                count += 1;
                min.0 = min.0.min(x);
                min.1 = min.1.min(y);
                max.0 = max.0.max(x);
                max.1 = max.1.max(y);
                sum.0 += x as f32;
                sum.1 += y as f32;
            }
        }
    }
    Darker {
        count,
        bounds: (min, max),
        centroid: (sum.0 / count.max(1) as f32, sum.1 / count.max(1) as f32),
        max_drop,
        max_at,
    }
}

/// A world point's pixel coordinate, or `None` when it is behind or outside the
/// view.
pub fn project(view_projection: Mat4, world: Vec3, width: u32, height: u32) -> Option<(u32, u32)> {
    let clip = view_projection * world.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let x = (ndc.x * 0.5 + 0.5) * width as f32;
    let y = (0.5 - ndc.y * 0.5) * height as f32;
    if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
        return None;
    }
    Some((x.round() as u32, y.round() as u32))
}
