//! Command recording, on the real driver.
//!
//! The triangle goes in through `chassis::CommandList`, comes out in the pixels,
//! and the pixels are read back and checked. Same shape as
//! `UHE/Stormbreaker/vulkan/tests/executor.rs`, which is the point: two backends,
//! one set of assertions.

use chassis::CommandList;
use chassis::Resources;
use stormglass::wgpu_command_buffer::{Recording, WgpuCommandList};
use stormglass::wgpu_device::Adapter;
use stormglass::wgpu_resources::WgpuResources;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// position xyz, one pad, colour rgba -- eight floats against a 32-byte stride.
const VERTICES: [f32; 24] = [
    0.0, 0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, //
    -0.75, -0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, //
    0.75, -0.75, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, //
];

fn asset(relative: &str) -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        let candidate = ancestor.join(relative);
        if candidate.is_file() {
            return candidate;
        }
    }
    panic!("could not find {relative} above {}", manifest.display());
}

fn lit(pixels: &[u8]) -> usize {
    let (pixels, _) = pixels.as_chunks::<4>();
    pixels.iter().filter(|pixel| pixel[0] > 128).count()
}

/// The whole chain: shader, pipeline, target, draw, read back.
#[test]
fn a_triangle_drawn_through_the_command_list_reaches_the_image() {
    let Some(adapter) = Adapter::request(0) else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-draw") else {
        eprintln!("SKIP: no device");
        return;
    };
    let mut resources = WgpuResources::new(&device);

    // The same shader and the same description as the Vulkan suite.
    let compiler =
        forge::slangc::SlangcCompiler::new(forge::Slangc::default()).expect("the vendored slangc");
    let forge = forge::Forge::new(compiler, forge::cache::ShaderCache::disabled());
    let directory = std::env::temp_dir().join("stormglass-draw");
    let (vertex_module, fragment_module) = forge
        .compile_pair(
            &asset("assets/shaders/Triangle.slang"),
            "vs_main",
            "fs_main",
            &directory,
        )
        .expect("Triangle.slang compiles");

    let vertex = resources
        .create_shader("Triangle.vert", &vertex_module.to_bytes())
        .expect("vertex module");
    let fragment = resources
        .create_shader("Triangle.frag", &fragment_module.to_bytes())
        .expect("fragment module");
    let pipeline = resources
        .create_pipeline(chassis::PipelineDesc {
            label: "triangle".to_owned(),
            shaders: vec![
                chassis::ShaderStageDesc {
                    stage: chassis::ShaderStage::Vertex,
                    shader: vertex,
                },
                chassis::ShaderStageDesc {
                    stage: chassis::ShaderStage::Fragment,
                    shader: fragment,
                },
            ],
            set_layouts: Vec::new(),
            vertex: chassis::VertexLayout {
                stride: 32,
                attributes: vec![
                    chassis::VertexAttribute {
                        location: 0,
                        format: chassis::TextureFormat::Rgba32Float,
                        offset: 0,
                    },
                    chassis::VertexAttribute {
                        location: 1,
                        format: chassis::TextureFormat::Rgba32Float,
                        offset: 16,
                    },
                ],
            },
            index_format: None,
            // Culling off: Vulkan's Y-flip means a counter-clockwise NDC
            // triangle is clockwise in framebuffer space, and wgpu flips too.
            // Both backends cull it under these defaults. Both suites set
            // `cull_mode: 0` for the same reason.
            primitive: chassis::PrimitiveState {
                cull_mode: 0,
                ..chassis::PrimitiveState::default()
            },
            depth: chassis::DepthState::default(),
            blend: chassis::BlendState::default(),
            color_targets: vec![chassis::ColorTargetState {
                format: chassis::TextureFormat::Rgba8Unorm,
                blend: false,
                write_mask: 0xf,
            }],
            samples: chassis::SampleCount::X1,
        })
        .expect("a two-stage graphics pipeline");

    let target = resources
        .create_texture(chassis::TextureDesc::new_2d(
            chassis::Extent {
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
            },
            chassis::TextureFormat::Rgba8Unorm,
            chassis::TextureUsage::COLOR_ATTACHMENT | chassis::TextureUsage::TRANSFER_SRC,
        ))
        .expect("target texture");
    // One buffer, as Vulkan does. It took two separate discoveries to get here:
    //
    // 1. wgpu will not map a buffer that lacks `MAP_READ` at *creation*, while
    //    Vulkan reads back by mapping host-visible memory with no flag at all.
    //    `chassis::BufferUsage` had no such flag -- the RHI was incomplete, and
    //    only the second backend found it.
    // 2. wgpu then refused the buffer anyway: `MAP_READ` must be paired with
    //    `COPY_DST` (not `COPY_SRC`, which is the opposite of what the message
    //    reads like), and the whole thing requires the
    //    `MAPPABLE_PRIMARY_BUFFERS` device feature, which this backend was
    //    deliberately not requesting.
    //
    // The rule "request no features" was right for anything optional and wrong
    // for anything load-bearing. Readback is how both backends' tests prove
    // anything, so it is now requested adaptively: only when the adapter has it.
    let readback = resources
        .create_buffer(
            chassis::BufferDesc::new(
                u64::from(WIDTH * HEIGHT * 4),
                chassis::BufferUsage::TRANSFER_DST | chassis::BufferUsage::MAP_READ,
            ),
            None,
        )
        .expect("readback buffer");
    let vertex_buffer = resources
        .create_buffer(
            chassis::BufferDesc::new(
                std::mem::size_of_val(&VERTICES) as u64,
                chassis::BufferUsage::VERTEX | chassis::BufferUsage::TRANSFER_DST,
            ),
            Some(bytemuck::cast_slice(&VERTICES)),
        )
        .expect("vertex buffer");

    // Record.
    let mut recording = Recording::new(device.inner(), "frame");
    {
        let mut list = WgpuCommandList::new(&mut recording, &resources);
        let mut desc = chassis::RenderPassDesc::color(chassis::TextureFormat::Rgba8Unorm);
        desc.colors[0].texture = target;
        desc.label = "draw".to_owned();
        list.begin_rendering(&desc);
        list.set_viewport(chassis::Extent2d {
            width: WIDTH,
            height: HEIGHT,
        });
        list.set_scissor(0, 0, WIDTH, HEIGHT);
        list.bind_pipeline(&pipeline);
        list.bind_vertex_buffer(0, &vertex_buffer, 0);
        list.draw(3, 1);
        list.end_rendering();
    }
    assert!(
        !recording.is_inside_render_pass(),
        "the pass was closed before the buffer was finished",
    );

    // Copy out, then submit. The copy is outside the pass: a copy inside a
    // render pass is a validation error in both APIs.
    {
        let mut list = WgpuCommandList::new(&mut recording, &resources);
        list.copy_texture_to_buffer(&target, &readback);
    }

    let buffer = recording.finish();
    queue.inner().submit([buffer]);
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the queue drains");

    // Read back. `read_buffer` maps, copies and unmaps: the same wgpu steps the
    // long comment above describes, in one call the renderer also uses.
    let pixels = resources
        .read_buffer(&readback)
        .expect("the readback buffer maps after the queue drains");

    let lit = lit(&pixels);
    assert!(
        lit > 0,
        "the image is entirely black: {WIDTH}x{HEIGHT}, {} bytes",
        pixels.len()
    );
    let fraction = lit as f64 / f64::from(WIDTH * HEIGHT);
    assert!(
        (0.02..0.45).contains(&fraction),
        "lit fraction {fraction:.3} is not what a triangle covering that area gives ({lit} of {} pixels)",
        WIDTH * HEIGHT
    );
}

/// A list that ends mid-pass is detectable before it reaches wgpu.
///
/// Not a hypothetical: `end_rendering` is called by the executor, and a pass
/// whose attachment list failed to resolve would otherwise be submitted with the
/// scope still open, which wgpu turns into a runtime error at `finish`.
#[test]
fn an_unclosed_pass_is_visible_on_the_recording() {
    let Some(adapter) = Adapter::request(0) else {
        return;
    };
    let Some((device, _queue)) = adapter.create_device("stormglass-unclosed") else {
        return;
    };
    let resources = WgpuResources::new(&device);
    let mut recording = Recording::new(device.inner(), "unbalanced");

    {
        let mut list = WgpuCommandList::new(&mut recording, &resources);
        // No colour attachment resolves, so nothing opens a real pass -- but
        // the call is still made, and the flag tracks it.
        list.begin_rendering(&chassis::RenderPassDesc::color(
            chassis::TextureFormat::Rgba8Unorm,
        ));
    }

    assert!(
        recording.is_inside_render_pass(),
        "an unclosed pass is tracked, not silently accepted",
    );
}
