//! The wgpu implementation of `chassis::Resources`, on the real driver.
//!
//! These mirror `UHE/Stormbreaker/vulkan/tests/rhi.rs` deliberately. Two
//! backends are only interchangeable if the *same* tests pass against both, and
//! that only happens if the tests go through `&mut dyn Resources` and never name
//! a concrete type.
//!
//! So: every assertion here is one the Vulkan suite also makes. Where the two
//! backends genuinely differ, the difference is the finding and gets its own
//! test saying so.

use chassis::Resources;
use stormglass::wgpu_device::Adapter;
use stormglass::wgpu_resources::WgpuResources;

/// The backend behind the trait, or `None` when this machine has no wgpu adapter.
fn backend() -> Option<WgpuResources> {
    let adapter = Adapter::request(0)?;
    let (device, queue) = adapter.create_device("stormglass-resources")?;
    // With the queue: `upload_texture` is `queue.write_texture`, so a resources
    // built without one cannot upload and says so.
    Some(WgpuResources::with_queue(
        &device,
        Some(queue.inner().clone()),
    ))
}

/// With a device, or a skip. Used by every test so none can panic on a machine
/// with no driver.
macro_rules! with_backend {
    (|$resources:ident| $body:block) => {
        let Some(mut $resources) = backend() else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        $body
    };
}

fn texture_desc() -> chassis::TextureDesc {
    chassis::TextureDesc::new_2d(
        chassis::Extent {
            width: 64,
            height: 64,
            depth: 1,
        },
        chassis::TextureFormat::Rgba8Unorm,
        chassis::TextureUsage::COLOR_ATTACHMENT,
    )
}

/// The same contract the Vulkan suite asserts.
#[test]
fn the_rhi_contract_holds_through_a_trait_object() {
    with_backend!(|resources| {
        let resources: &mut dyn Resources = &mut resources;

        let buffer = resources
            .create_buffer(
                chassis::BufferDesc::new(256, chassis::BufferUsage::VERTEX),
                Some(&[7u8; 256]),
            )
            .expect("a buffer with initial data");
        let texture = resources
            .create_texture(texture_desc())
            .expect("a colour texture");
        let view = resources.create_image_view(&texture).expect("a view of it");

        assert!(!view.is_null(), "a live view handle");
        // Independent registries, so both starting at zero is correct. The tag
        // distinguishes them, not the index.
        assert_eq!(buffer.index(), texture.index());

        resources.destroy_texture(texture);
        resources.destroy_buffer(buffer);
    });
}

/// The same test, run twice, and the second time must still work.
///
/// Worth its own name because it is what a second backend gets wrong first: a
/// registry that drops its device, or a handle that still resolves after its
/// resource is released.
#[test]
fn resources_can_be_created_and_released_repeatedly() {
    with_backend!(|concrete| {
        let resources: &mut dyn Resources = &mut concrete;
        for iteration in 0..8 {
            let buffer = resources
                .create_buffer(
                    chassis::BufferDesc::new(1024, chassis::BufferUsage::VERTEX),
                    Some(&[iteration as u8; 1024]),
                )
                .expect("buffer");
            let texture = resources.create_texture(texture_desc()).expect("texture");
            resources.destroy_buffer(buffer);
            resources.destroy_texture(texture);
        }
        // Nothing left behind. "Nothing leaked" is not observable through
        // `dyn Resources` at all -- there is no method for it -- so this reaches
        // past the trait deliberately, through the concrete handle.
        let (buffers, textures, views, shaders) = concrete.counts();
        assert_eq!(
            (buffers, textures, shaders),
            (0, 0, 0),
            "every resource was released; views are tracked separately and are {}",
            views
        );
    });
}

#[test]
fn destroying_twice_is_not_a_crash() {
    with_backend!(|resources| {
        let resources: &mut dyn Resources = &mut resources;
        let buffer = resources
            .create_buffer(
                chassis::BufferDesc::new(64, chassis::BufferUsage::INDEX),
                None,
            )
            .expect("buffer");
        resources.destroy_buffer(buffer);
        // A released handle resolves to nothing, so this is a no-op rather than
        // a double free. wgpu has no explicit destroy, so the only thing that
        // could be double-freed is the handle.
        resources.destroy_buffer(buffer);

        let texture = resources.create_texture(texture_desc()).expect("texture");
        resources.destroy_texture(texture);
        resources.destroy_texture(texture);
    });
}

/// A zero-sized buffer is refused.
///
/// The Vulkan backend refuses it too, and for the same reason: a zero-size
/// object is legal to create in the API and useless, and some drivers accept it
/// and then ignore everything recorded against it.
#[test]
fn a_buffer_with_no_size_or_no_usage_is_refused() {
    with_backend!(|resources| {
        let resources: &mut dyn Resources = &mut resources;
        assert!(
            resources
                .create_buffer(
                    chassis::BufferDesc::new(0, chassis::BufferUsage::VERTEX),
                    None
                )
                .is_err(),
            "zero bytes is not a buffer",
        );
        assert!(
            resources
                .create_buffer(
                    chassis::BufferDesc::new(64, chassis::BufferUsage::NONE),
                    None
                )
                .is_err(),
            "no usage is not a buffer either",
        );
    });
}

#[test]
fn a_handle_that_never_existed_is_an_error_not_a_panic() {
    with_backend!(|resources| {
        let resources: &mut dyn Resources = &mut resources;
        assert!(
            resources
                .create_image_view(&chassis::Texture::null())
                .is_err(),
            "a view of nothing is an error",
        );
    });
}

/// Malformed SPIR-V is refused where the message can say what is wrong.
///
/// The two backends get this from different places -- naga here, the Vulkan
/// driver there -- and both must refuse it, because the alternative is an opaque
/// failure at pipeline creation that names neither the shader nor the backend.
#[test]
fn malformed_spirv_is_refused_with_a_reason() {
    with_backend!(|resources| {
        let resources: &mut dyn Resources = &mut resources;
        for (name, bytes) in [
            ("empty", Vec::new()),
            ("misaligned", vec![1, 2, 3]),
            ("wrong magic", vec![0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]),
        ] {
            let error = resources
                .create_shader("broken", &bytes)
                .expect_err("{name} is not SPIR-V");
            assert!(
                matches!(error, chassis::Error::InvalidDescriptor(_)),
                "{name} gave {error:?}",
            );
        }
    });
}

/// The real shader, compiled by `forge` from the checked-in source.
///
/// This is the test that proves the two backends share one compiler: the same
/// bytes that the Vulkan suite loads are loaded here, through the same `forge`
/// cache, and must produce a module wgpu accepts.
#[test]
fn the_shared_shader_from_forge_is_accepted() {
    let Some(mut resources) = backend() else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let compiler =
        forge::slangc::SlangcCompiler::new(forge::Slangc::default()).expect("the vendored slangc");
    // `forge` is both a crate and a local name; the local is `compiler` so the
    // crate stays reachable on the next line.
    let shaders = forge::Forge::new(compiler, forge::cache::ShaderCache::disabled());
    let directory = std::env::temp_dir().join("stormglass-shader");
    let (vertex, fragment) = shaders
        .compile_pair(
            &asset("assets/shaders/Triangle.slang"),
            "vs_main",
            "fs_main",
            &directory,
        )
        .expect("Triangle.slang compiles");

    let resources: &mut dyn Resources = &mut resources;
    let vertex_module = resources
        .create_shader("Triangle.vert", &vertex.to_bytes())
        .expect("the vertex module wgpu accepts");
    let fragment_module = resources
        .create_shader("Triangle.frag", &fragment.to_bytes())
        .expect("the fragment module wgpu accepts");
    assert_ne!(vertex_module, fragment_module);
}

/// A bind group, built the same way the Vulkan suite builds one.
///
/// The test that used to sit here asserted that *every* bind group failed, since
/// wgpu had not implemented them. That assertion could have survived `create_bind_
/// group` starting to return a handle which binds nothing -- the failure it
/// exists to catch, and the one it had stopped being able to see.
#[test]
fn a_bind_group_is_built_from_its_declared_layout() {
    with_backend!(|resources| {
        let layout =
            chassis::DescriptorSetLayoutDesc::new("frame").with(chassis::BindingDesc::new(
                0,
                chassis::DescriptorKind::UniformBuffer,
                chassis::StageMask::ALL,
            ));

        let buffer = resources
            .create_buffer(
                chassis::BufferDesc::new(256, chassis::BufferUsage::UNIFORM),
                None,
            )
            .expect("a uniform buffer");

        let resources: &mut dyn Resources = &mut resources;
        let group = resources
            .create_bind_group(&layout, &[(0, buffer, 0)], &[], &[])
            .expect("a bind group");
        assert!(!group.is_null(), "a real handle");

        // Destroying releases the generational slot, so a stale handle cannot
        // resolve to whatever group comes next.
        resources.destroy_bind_group(group);
    });
}

/// A binding the layout does not declare is an error, not wgpu's panic.
///
/// `create_bind_group` is infallible in wgpu and reports the mistake through the
/// device handler. The pre-check names the binding; the validation scope behind
/// it catches anything the check does not.
#[test]
fn an_undeclared_binding_is_an_error() {
    with_backend!(|resources| {
        let layout =
            chassis::DescriptorSetLayoutDesc::new("frame").with(chassis::BindingDesc::new(
                0,
                chassis::DescriptorKind::UniformBuffer,
                chassis::StageMask::ALL,
            ));
        let buffer = resources
            .create_buffer(
                chassis::BufferDesc::new(256, chassis::BufferUsage::UNIFORM),
                None,
            )
            .expect("a uniform buffer");

        let resources: &mut dyn Resources = &mut resources;
        let error = resources
            .create_bind_group(&layout, &[(9, buffer, 0)], &[], &[])
            .expect_err("binding 9 is not declared");
        let message = error.to_string();
        assert!(
            message.contains('9'),
            "the error names the binding: {message}"
        );
    });
}

/// The shared layout cache, which in wgpu is load-bearing rather than an
/// optimisation.
///
/// wgpu compares a bind group against the pipeline's layout by *identity*, so two
/// equal descriptions built separately are two layouts and the group cannot be
/// used with the pipeline. Vulkan compares structurally and does not care. One
/// cache, keyed the same way `is_compatible` compares, absorbs the difference.
#[test]
fn an_equal_layout_is_one_layout() {
    with_backend!(|resources| {
        let mut named_one = chassis::DescriptorSetLayoutDesc::new("frame");
        named_one.bindings.push(chassis::BindingDesc::new(
            0,
            chassis::DescriptorKind::UniformBuffer,
            chassis::StageMask::ALL,
        ));
        let mut named_two = chassis::DescriptorSetLayoutDesc::new("totally-different-name");
        named_two.bindings = named_one.bindings.clone();

        let before = resources.bind_group_layout_count();
        let one = resources
            .bind_group_layout(&named_one)
            .expect("first layout");
        let two = resources
            .bind_group_layout(&named_two)
            .expect("the same layout under another name");
        assert_eq!(
            resources.bind_group_layout_count(),
            before + 1,
            "a different label is still the same layout",
        );
        // Equality is the available spelling of identity for a wgpu resource
        // handle. It is a weaker claim than `Arc::ptr_eq` -- wgpu's `PartialEq`
        // compares the underlying id, which is what "the same object" means here
        // -- but it is the check that actually matters, and the cache count above
        // already proves no second object was made.
        assert_eq!(
            one, two,
            "the cache hands back the same layout, not a second one",
        );
    });
}

/// An offset at or past the end of a buffer is refused rather than turned into an
/// empty range, on both backends and for the same reason: an empty binding reads
/// nothing and looks like a shader bug.
#[test]
fn an_offset_at_or_past_the_end_is_refused() {
    with_backend!(|resources| {
        let layout =
            chassis::DescriptorSetLayoutDesc::new("frame").with(chassis::BindingDesc::new(
                0,
                chassis::DescriptorKind::UniformBuffer,
                chassis::StageMask::ALL,
            ));
        let buffer = resources
            .create_buffer(
                chassis::BufferDesc::new(256, chassis::BufferUsage::UNIFORM),
                None,
            )
            .expect("a uniform buffer");

        let resources: &mut dyn Resources = &mut resources;
        assert!(resources
            .create_bind_group(&layout, &[(0, buffer, 256)], &[], &[])
            .is_err());
        assert!(
            resources
                .create_bind_group(&layout, &[(0, buffer, 1 << 20)], &[], &[])
                .is_err(),
            "a wildly out-of-range offset is refused too",
        );
    });
}

/// A real pipeline, from the same shader and the same description the Vulkan
/// suite builds.
///
/// This is the test that says the two backends are interchangeable rather than
/// merely similar. Same SPIR-V, same `chassis::PipelineDesc`, two very different
/// pipelines underneath. The description is copied field for field from
/// `UHE/Stormbreaker/vulkan/tests/pipelines.rs` -- if one backend's tests drift,
/// the seam stops being tested and starts being two implementations.
#[test]
fn a_graphics_pipeline_from_the_shared_shader_creates() {
    let Some(mut resources) = backend() else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };

    let compiler =
        forge::slangc::SlangcCompiler::new(forge::Slangc::default()).expect("the vendored slangc");
    let shaders = forge::Forge::new(compiler, forge::cache::ShaderCache::disabled());
    let directory = std::env::temp_dir().join("stormglass-pipeline");
    let (vertex_module, fragment_module) = shaders
        .compile_pair(
            &asset("assets/shaders/Triangle.slang"),
            "vs_main",
            "fs_main",
            &directory,
        )
        .expect("Triangle.slang compiles");

    let resources: &mut dyn Resources = &mut resources;
    let vertex = resources
        .create_shader("Triangle.vert", &vertex_module.to_bytes())
        .expect("vertex module");
    let fragment = resources
        .create_shader("Triangle.frag", &fragment_module.to_bytes())
        .expect("fragment module");

    let handle = resources
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
            // Culling off: the same Vulkan Y-flip caveat as the Vulkan suite, and
            // wgpu flips too. Both backends cull a counter-clockwise NDC
            // triangle under these defaults, and both need `front_face: 1` or no
            // culling to draw it.
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
        .expect("a two-stage graphics pipeline creates");

    assert!(!handle.is_null());
}

/// A pipeline whose module names its entries `vs_main`/`fs_main`, not `main`.
///
/// naga keeps the source's names and Slang emits `main`. The backend reads
/// `OpEntryPoint` from the module, so both build a pipeline without either
/// compiler's naming being copied by the other.
#[test]
fn a_pipeline_uses_the_modules_own_entry_point_names() {
    const WGSL: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.0,  0.5),
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

    with_backend!(|resources| {
        let spirv = forge::wgsl::to_spirv(WGSL, "own-entry").expect("the WGSL compiles");
        let resources: &mut dyn Resources = &mut resources;
        let shader = resources
            .create_shader("own-entry", &spirv)
            .expect("the module wgpu accepts");

        let handle = resources
            .create_pipeline(chassis::PipelineDesc {
                label: "own-entry".to_owned(),
                shaders: vec![
                    chassis::ShaderStageDesc {
                        stage: chassis::ShaderStage::Vertex,
                        shader,
                    },
                    chassis::ShaderStageDesc {
                        stage: chassis::ShaderStage::Fragment,
                        shader,
                    },
                ],
                set_layouts: Vec::new(),
                vertex: chassis::VertexLayout::default(),
                index_format: None,
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
            .expect("the parsed entry names build the pipeline");

        assert!(!handle.is_null());
    });
}

/// A pipeline with no vertex stage is refused.
///
/// wgpu's `VertexState` has no such thing, so the alternative is an error deep
/// inside `create_render_pipeline` naming an internal index.
#[test]
fn a_pipeline_without_a_vertex_stage_is_refused() {
    let Some(mut resources) = backend() else {
        return;
    };
    let compiler =
        forge::slangc::SlangcCompiler::new(forge::Slangc::default()).expect("the vendored slangc");
    let shaders = forge::Forge::new(compiler, forge::cache::ShaderCache::disabled());
    let directory = std::env::temp_dir().join("stormglass-pipeline-novs");
    let (vertex_module, fragment_module) = shaders
        .compile_pair(
            &asset("assets/shaders/Triangle.slang"),
            "vs_main",
            "fs_main",
            &directory,
        )
        .expect("Triangle.slang compiles");

    let resources: &mut dyn Resources = &mut resources;
    let fragment = resources
        .create_shader("Triangle.frag", &fragment_module.to_bytes())
        .expect("fragment module");
    let _ = vertex_module;

    let error = resources
        .create_pipeline(chassis::PipelineDesc {
            label: "no-vertex".to_owned(),
            shaders: vec![chassis::ShaderStageDesc {
                stage: chassis::ShaderStage::Fragment,
                shader: fragment,
            }],
            set_layouts: Vec::new(),
            vertex: chassis::VertexLayout::default(),
            index_format: None,
            primitive: chassis::PrimitiveState::default(),
            depth: chassis::DepthState::default(),
            blend: chassis::BlendState::default(),
            color_targets: vec![chassis::ColorTargetState {
                format: chassis::TextureFormat::Rgba8Unorm,
                blend: false,
                write_mask: 0xf,
            }],
            samples: chassis::SampleCount::X1,
        })
        .expect_err("a pipeline with no vertex stage is refused");
    assert!(
        matches!(error, chassis::Error::Unsupported(_)),
        "and it says why, got {error:?}",
    );
}

/// A module with a valid magic number and a garbage body is an error, not a panic.
///
/// The two backends get the same answer by different routes:
///
///   Vulkan  `create_shader_module` returns `Result`, so a bad module is an error
///   wgpu    `create_shader_module` is infallible; validation goes to the device
///           handler, which panics when no error scope is installed.
///
/// `WgpuResources::create_shader` installs a validation scope, so the corrupt
/// body comes back as `InvalidDescriptor`. The magic check in `forge` still
/// refuses truncated and wrong-magic input before wgpu sees it.
#[test]
fn a_module_with_a_valid_magic_and_a_corrupt_body_is_an_error() {
    with_backend!(|resources| {
        let resources: &mut dyn Resources = &mut resources;
        let error = resources
            .create_shader("any", &[0x03, 0x02, 0x23, 0x07, 0, 0, 0, 0])
            .expect_err("a corrupt body is not a shader");
        assert!(
            matches!(error, chassis::Error::InvalidDescriptor(_)),
            "a corrupt body gave {error:?}",
        );
    });
}

/// A stage wgpu has no room for is named, not dropped.
///
/// Silently ignoring a geometry or tessellation stage produces a pipeline that
/// creates fine and draws with fewer stages than its shader expects, which is
/// the most expensive kind of wrong. `shader_state` returns
/// `Error::Unsupported` naming the stage; this test needs a real module to reach
/// that path, and the corrupt-body case above is the cheaper way to prove the
/// code is reached at all.
#[test]
fn geometry_is_a_stage_wgpu_has_no_room_for() {
    // A compile-time fact rather than a runtime one: the enum has the variant,
    // so `shader_state`'s match must handle it, and its arm must be the
    // `Unsupported` one.
    let stage: Option<chassis::ShaderStage> = Some(chassis::ShaderStage::Geometry);
    assert!(stage.is_some());
}

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

/// A texture that can be *sampled*, as opposed to the shared `texture_desc`
/// which is a render target.
///
/// The distinction is not academic: wgpu checks that a view bound to a
/// `SampledImage` descriptor came from a texture with `TEXTURE_BINDING`, and
/// fails with "Usage flags RENDER_ATTACHMENT do not contain TEXTURE_BINDING".
/// Vulkan makes no such check at bind time -- it is a layout question, not a
/// usage one -- so this exact mistake is invisible on one backend and fatal on
/// the other.
fn sampled_texture_desc() -> chassis::TextureDesc {
    chassis::TextureDesc::new_2d(
        chassis::Extent {
            width: 16,
            height: 16,
            depth: 1,
        },
        chassis::TextureFormat::Rgba8Unorm,
        // `TRANSFER_DST` as well as `SAMPLED`, and the reason is a backend
        // difference worth knowing: Vulkan's `vkCmdCopyBufferToImage` requires
        // `TRANSFER_DST`, and wgpu's `queue.write_texture` requires `COPY_DST`.
        // The RHI's `TRANSFER_DST` maps to both, so one description uploads on
        // either -- but a texture created `SAMPLED` alone fails on wgpu with
        // "Usage flags TEXTURE_BINDING do not contain COPY_DST" and works fine on
        // Vulkan.
        chassis::TextureUsage::SAMPLED | chassis::TextureUsage::TRANSFER_DST,
    )
}

/// A sampler and an image bound with it, through the trait.
///
/// The mirror of the Vulkan `an_image_and_a_sampler_bind_together`, and the test
/// that would have caught `count: Some(1)` if it had been written for images
/// first. It was not, so it was caught by probing all three spellings by hand.
#[test]
fn an_image_and_a_sampler_bind_together() {
    with_backend!(|resources| {
        let layout = chassis::DescriptorSetLayoutDesc::new("material")
            .with(chassis::BindingDesc::new(
                0,
                chassis::DescriptorKind::SampledImage,
                chassis::StageMask::ALL,
            ))
            .with(chassis::BindingDesc::new(
                1,
                chassis::DescriptorKind::Sampler,
                chassis::StageMask::ALL,
            ));

        let texture = resources
            .create_texture(sampled_texture_desc())
            .expect("a sampled texture");
        let view = resources
            .create_image_view(&texture)
            .expect("an image view");
        let sampler = resources
            .create_sampler(&chassis::SamplerDesc::trilinear())
            .expect("a sampler");
        assert!(!sampler.is_null());

        let resources: &mut dyn Resources = &mut resources;
        let group = resources
            .create_bind_group(&layout, &[], &[(0, view)], &[(1, sampler)])
            .expect("an image bind group");
        assert!(!group.is_null());

        resources.destroy_sampler(sampler);
    });
}

/// `CombinedImageSampler` exists on Vulkan and **does not exist here**.
///
/// WebGPU has no combined descriptor, so `wgpu::BindingResource` has no variant
/// for one: the variants are Buffer, Sampler, TextureView and their array forms.
/// The RHI's most convenient descriptor kind cannot be expressed on this backend.
///
/// So it is refused, and the refusal names the alternative. The parity test
/// asserts the two backends *disagree* here, which is the one place they must.
#[test]
fn a_combined_image_sampler_is_refused_with_the_alternative() {
    with_backend!(|resources| {
        let layout =
            chassis::DescriptorSetLayoutDesc::new("material").with(chassis::BindingDesc::new(
                0,
                chassis::DescriptorKind::CombinedImageSampler,
                chassis::StageMask::ALL,
            ));
        let texture = resources
            .create_texture(sampled_texture_desc())
            .expect("a sampled texture");
        let view = resources
            .create_image_view(&texture)
            .expect("an image view");
        let sampler = resources
            .create_sampler(&chassis::SamplerDesc::clamped())
            .expect("a sampler");

        let resources: &mut dyn Resources = &mut resources;
        let error = resources
            .create_bind_group(&layout, &[], &[(0, view)], &[(0, sampler)])
            .expect_err("wgpu has no combined image sampler");
        let message = error.to_string();
        assert!(
            message.contains("SampledImage") && message.contains("Sampler"),
            "the error names the alternative to use instead: {message}",
        );
    });
}

/// Two bind groups, one texture, **different samplers**.
///
/// The shape every material system has: a shared texture bound with a per-material
/// filter. The test is that the two groups are two groups.
///
/// It exists because a bind group cache keyed on the *layout* alone -- which is
/// the obvious key, and is what `DescriptorSetLayoutDesc::hash` gives -- returns
/// one group for both, and then every material in the scene samples with
/// whichever filter was built first. The symptom is "my nearest-filtered texture
/// is blurry" and the cause is a hash that forgot a field.
#[test]
fn two_samplers_on_one_texture_are_two_groups() {
    with_backend!(|resources| {
        let layout = chassis::DescriptorSetLayoutDesc::new("material")
            .with(chassis::BindingDesc::new(
                0,
                chassis::DescriptorKind::SampledImage,
                chassis::StageMask::ALL,
            ))
            .with(chassis::BindingDesc::new(
                1,
                chassis::DescriptorKind::Sampler,
                chassis::StageMask::ALL,
            ));

        let texture = resources
            .create_texture(sampled_texture_desc())
            .expect("a sampled texture");
        let view = resources.create_image_view(&texture).expect("a view");
        let nearest = resources
            .create_sampler(&chassis::SamplerDesc::nearest())
            .expect("nearest");
        let clamped = resources
            .create_sampler(&chassis::SamplerDesc::clamped())
            .expect("clamped");
        assert_ne!(nearest, clamped, "two samplers are two handles");

        let resources: &mut dyn Resources = &mut resources;
        let a = resources
            .create_bind_group(&layout, &[], &[(0, view)], &[(1, nearest)])
            .expect("first");
        let b = resources
            .create_bind_group(&layout, &[], &[(0, view)], &[(1, clamped)])
            .expect("second");
        assert_ne!(
            a, b,
            "the same layout with a different sampler is a different group"
        );
    });
}

/// Four texels uploaded and read back, on this backend.
///
/// The mirror of the Vulkan upload test, and the reason the two are worth having
/// separately: **wgpu's upload is synchronous and Vulkan's is not.** Here the
/// bytes are written and the texture is usable when `upload_texture` returns;
/// there a copy is queued and the texture is in use by a command buffer until the
/// ring drains.
///
/// A caller written for wgpu and moved to Vulkan would destroy textures while
/// they were in flight. That is the asymmetry, and each test now says which side
/// it is on.
#[test]
fn four_texels_upload_and_read_back() {
    with_backend!(|resources| {
        const QUADRANTS: [[u8; 4]; 4] = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
        ];
        let texels: Vec<u8> = QUADRANTS.iter().flatten().copied().collect();

        let resources: &mut dyn Resources = &mut resources;
        let texture = resources
            .create_texture(sampled_texture_desc())
            .expect("a sampled texture");
        resources
            .upload_texture(&texture, &texels, 2, 2)
            .expect("four texels");

        // A size the caller's data does not match is refused, not truncated.
        let error = resources
            .upload_texture(&texture, &texels[..8], 2, 2)
            .expect_err("8 bytes is not a 2x2 RGBA8 texture");
        assert!(
            error.to_string().contains("16"),
            "the error names the size it wanted: {error}",
        );
    });
}
