//! GPU time per bifrost pass, on Vulkan, and the same frame on wgpu.
//!
//! `#[ignore]` because it is a measurement, not a correctness check. Run it:
//!
//! ```text
//! UHE_VALIDATION=1 cargo test -p stormbreaker --test bench_gpu \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Vulkan timestamps come from `VkQueryPool` (`VK_QUERY_TYPE_TIMESTAMP`) around
//! every pass; that is the backend's `enable_gpu_timing`. wgpu has no timing
//! hook in this tree and `stormglass` is read-only here, so its column is the
//! whole frame. The two totals are what the issue compares, and the per-pass
//! split says where the Vulkan time went.
//!
//! `BENCH_WIDTH`, `BENCH_HEIGHT` and `BENCH_FRAMES` override the defaults.

use std::path::{Path, PathBuf};
use std::time::Instant;

use bifrost::{Bifrost, EditorCamera, ViewSource};
use chassis::{Extent2d, Gpu};
use glam::Vec3;
use meats::{Light, MeshRenderer, Scene};
use stormbreaker::vulkan_gpu::VulkanGpu;
use stormglass::wgpu_gpu::WgpuGpu;
use vault::Assets;

const DEFAULT_WIDTH: u32 = 1998;
const DEFAULT_HEIGHT: u32 = 1135;

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn frames() -> usize {
    env_u32("BENCH_FRAMES", 30) as usize
}

fn extent() -> Extent2d {
    Extent2d {
        width: env_u32("BENCH_WIDTH", DEFAULT_WIDTH),
        height: env_u32("BENCH_HEIGHT", DEFAULT_HEIGHT),
    }
}

/// The model directory, found above this crate because cargo runs from it.
fn asset_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        let candidate = ancestor.join("assets/models");
        if candidate.join("Box.gltf").is_file() {
            return candidate;
        }
    }
    panic!("could not find assets/models above {}", manifest.display());
}

fn scene(assets: &mut Assets) -> (Scene, EditorCamera) {
    let id = assets.load("Box.gltf").expect("load Box.gltf");
    let mut scene = Scene::new();
    let mesh = scene.spawn("box");
    assert!(
        scene.insert(
            mesh,
            MeshRenderer {
                mesh: Some(id),
                cast_shadows: true,
                ..MeshRenderer::default()
            }
        ),
        "box entity"
    );
    let light = scene.spawn("light");
    assert!(scene.insert(light, Light::default()), "light entity");

    let size = extent();
    let mut camera = EditorCamera::new(45.0, size.width as f32 / size.height as f32, 0.1, 1000.0);
    camera.set_distance(6.0);
    camera.set_pitch(0.35);
    camera.set_yaw(0.6);
    camera.set_focal_point(Vec3::new(0.0, 0.0, 0.0));
    (scene, camera)
}

fn build(gpu: &mut dyn Gpu, assets: &mut Assets, scene: &Scene) -> Bifrost {
    let mut renderer = Bifrost::new(gpu, assets).expect("bifrost builds");
    for sample in scene.meshes() {
        renderer
            .ensure_mesh(gpu, assets, sample.mesh)
            .expect("mesh upload");
    }
    renderer
}

fn open_vulkan() -> Option<VulkanGpu> {
    match VulkanGpu::create_headless(c"stormbreaker-bench-gpu", extent()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

fn open_wgpu() -> Option<WgpuGpu> {
    match WgpuGpu::headless(extent()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("SKIP: no wgpu device: {error}");
            None
        }
    }
}

#[test]
#[ignore = "GPU benchmark; run with --ignored"]
fn per_pass_gpu_time() {
    let root = asset_root();
    let count = frames();
    let size = extent();

    if let Some(mut gpu) = open_vulkan() {
        let mut assets = Assets::new(&root);
        let (scene, camera) = scene(&mut assets);
        let mut renderer = build(&mut gpu, &mut assets, &scene);
        let jobs = grind::Jobs::new();

        for _ in 0..3 {
            renderer
                .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
                .expect("warmup frame");
        }
        gpu.wait_idle();

        let start = Instant::now();
        for _ in 0..count {
            renderer
                .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
                .expect("bench frame");
        }
        gpu.wait_idle();
        let wall = start.elapsed().as_secs_f64() * 1e3 / count as f64;
        println!(
            "\nVulkan {}x{}, {count} frames (wall, no timing):",
            size.width, size.height
        );
        println!("  {:<12} {:>9.3}", "wall", wall);

        if !gpu.enable_gpu_timing() {
            eprintln!("SKIP: the Vulkan queue cannot write timestamps");
        } else {
            for _ in 0..3 {
                renderer
                    .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
                    .expect("warmup frame");
            }
            gpu.wait_idle();

            let mut totals = Vec::new();
            let mut names: Vec<String> = Vec::new();
            let mut sums: Vec<f64> = Vec::new();
            let mut wall = 0.0;
            for _ in 0..count {
                let start = Instant::now();
                renderer
                    .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
                    .expect("bench frame");
                gpu.wait_idle();
                wall += start.elapsed().as_secs_f64() * 1e3;
                gpu.collect_gpu_timings().expect("timestamp results");
                totals.push(gpu.gpu_frame_ms().unwrap_or(f32::NAN) as f64);
                if names.is_empty() {
                    names = gpu
                        .gpu_timings()
                        .iter()
                        .map(|timing| timing.name.clone())
                        .collect();
                    sums = vec![0.0; names.len()];
                }
                for (slot, timing) in gpu.gpu_timings().iter().enumerate() {
                    if slot < sums.len() {
                        sums[slot] += f64::from(timing.ms);
                    }
                }
            }

            println!("\nVulkan {}x{}, {count} frames:", size.width, size.height);
            println!("  {:<12} {:>9}", "pass", "GPU ms");
            for (name, sum) in names.iter().zip(&sums) {
                println!("  {name:<12} {:>9.3}", sum / count as f64);
            }
            let frame_mean: f64 = totals.iter().sum::<f64>() / count as f64;
            println!("  {:<12} {:>9.3}", "frame", frame_mean);
            println!("  {:<12} {:>9.3}", "wall", wall / count as f64);
        }
        renderer.destroy(&mut gpu);
    }

    if let Some(mut gpu) = open_wgpu() {
        let mut assets = Assets::new(&root);
        let (scene, camera) = scene(&mut assets);
        let mut renderer = build(&mut gpu, &mut assets, &scene);
        let jobs = grind::Jobs::new();
        for _ in 0..3 {
            renderer
                .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
                .expect("warmup frame");
        }
        gpu.wait_idle();
        let start = Instant::now();
        for _ in 0..count {
            renderer
                .render(&mut gpu, &jobs, &scene, ViewSource::Editor, Some(&camera))
                .expect("bench frame");
        }
        gpu.wait_idle();
        let wall = start.elapsed().as_secs_f64() * 1e3 / count as f64;
        println!("\nwgpu {}x{}, {count} frames:", size.width, size.height);
        println!("  {:<12} {:>9.3}", "wall", wall);
        renderer.destroy(&mut gpu);
    }
}

#[test]
#[ignore = "memory benchmark; run with --ignored"]
fn readback_memory_throughput() {
    use chassis::{BufferDesc, BufferUsage, Resources};

    let size = u64::from(env_u32("BENCH_WIDTH", DEFAULT_WIDTH))
        * u64::from(env_u32("BENCH_HEIGHT", DEFAULT_HEIGHT))
        * 4;
    let Some(mut gpu) = open_vulkan() else {
        return;
    };
    let data = vec![0u8; size as usize];
    let buffer = gpu
        .resources()
        .create_buffer(
            BufferDesc::new(size, BufferUsage::TRANSFER_DST | BufferUsage::MAP_READ),
            Some(&data),
        )
        .expect("readback buffer");
    let frames = frames();
    let start = Instant::now();
    for _ in 0..frames {
        let bytes = gpu.resources().read_buffer(&buffer).expect("read");
        std::hint::black_box(bytes);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1e3 / frames as f64;
    println!(
        "\nreadback {} bytes, {frames} frames: {elapsed:.3} ms/frame ({:.0} MB/s)",
        size,
        size as f64 / elapsed / 1e3
    );
    gpu.resources().destroy_buffer(buffer);
}
