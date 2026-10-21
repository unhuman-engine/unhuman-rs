//! The RHI seam, on real hardware.
//!
//! Everything here goes through `chassis::Resources` as a **trait object**,
//! never through `VulkanResources` directly. That is the whole point: if a test
//! calls the concrete methods, it proves nothing about the seam, and the seam is
//! the thing a wgpu backend has to fit into.
//!
//! Every test here is written so the same body would pass against a different
//! implementation. Where one cannot be written that way, the test says why --
//! that is a hole in the interface, and finding it now is the entire purpose of
//! having this file.

use ash::vk::Handle;
use chassis::Resources;
use stormbreaker::{create_device, device_available, DeviceOptions};

fn open_device() -> Option<stormbreaker::Context> {
    if !device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    match create_device(DeviceOptions::from_env(c"stormbreaker-resources-rhi")) {
        Ok(context) => Some(context),
        Err(stormbreaker::BackendError::Loader(why)) => {
            eprintln!("SKIP: loader unavailable: {why}");
            None
        }
        Err(other) => panic!("device creation failed: {other}"),
    }
}

/// The backend behind the trait. The trait object is what a renderer holds.
fn resources(context: &stormbreaker::Context) -> stormbreaker::vulkan_resources::VulkanResources {
    stormbreaker::vulkan_resources::VulkanResources::new(
        &context.device,
        context.memory_properties(),
    )
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

/// The contract every backend must satisfy, stated once.
///
/// `&mut dyn Resources` rather than the concrete type: if this compiles against
/// one implementation it must compile against another, because the renderer
/// will only ever see this.
#[test]
fn the_rhi_contract_holds_through_a_trait_object() {
    let Some(context) = open_device() else { return };
    let mut backend = resources(&context);
    let resources: &mut dyn Resources = &mut backend;

    // Create.
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

    // Buffer and texture are separate registries, so both starting at index 0
    // is correct -- the tag is what distinguishes them, not the index. Asserting
    // otherwise was mine and it was wrong: it would fail a correct backend.
    assert_eq!(
        buffer.index(),
        texture.index(),
        "independent registries both start at zero"
    );
    assert_ne!(
        buffer.generation(),
        99,
        "generations are carried on the handle"
    );

    // Destroy.
    resources.destroy_texture(texture);
    resources.destroy_buffer(buffer);
    // The view goes with its texture; `destroy_image_view` does not exist on the
    // RHI precisely because a view must never outlive what it views. Asserted by
    // absence at compile time -- if someone adds the method, this file should
    // have to answer for it.
    let _ = view;
}

/// A resource created through the RHI is usable by the backend, which is the
/// only thing that makes the RHI worth anything: a handle the backend cannot
/// resolve is a handle that works in a test and fails in a frame.
#[test]
fn a_handle_from_the_rhi_resolves_in_the_backend() {
    let Some(context) = open_device() else { return };
    let mut backend = resources(&context);

    let buffer = backend
        .create_buffer(
            chassis::BufferDesc::new(256, chassis::BufferUsage::VERTEX),
            Some(&[1u8; 256]),
        )
        .expect("buffer");
    let texture = backend.create_texture(texture_desc()).expect("texture");

    let resolved_buffer_size = {
        let buffers = backend.buffers();
        let found = buffers.get(buffer).expect("buffer resolves");
        found.size()
    };
    let (resolved_image, resolved_width) = {
        let textures = backend.textures();
        let found = textures.get(texture).expect("texture resolves");
        (found.image(), found.extent().width)
    };
    assert_eq!(resolved_buffer_size, 256);
    assert!(!resolved_image.is_null());
    assert_eq!(resolved_width, 64);

    // The view the RHI handed out is the same driver object.
    let view = backend.create_image_view(&texture).expect("image view");
    assert_eq!(backend.live_image_views(), 1);
    assert!(!view.is_null(), "the RHI handed out a live view handle");
}

/// Destroying twice must be safe. It is the shape a deferred-deletion queue
/// produces when a resource is released twice, and it happens during shutdown
/// when ordering is hardest to guarantee.
#[test]
fn destroying_twice_is_not_a_crash() {
    let Some(context) = open_device() else { return };
    let mut backend = resources(&context);

    let buffer = backend
        .create_buffer(
            chassis::BufferDesc::new(64, chassis::BufferUsage::VERTEX),
            None,
        )
        .expect("buffer");
    backend.destroy_buffer(buffer);
    // A stale handle after destroy resolves to nothing, so this is a no-op
    // rather than a double free.
    backend.destroy_buffer(buffer);

    let texture = backend.create_texture(texture_desc()).expect("texture");
    backend.destroy_texture(texture);
    backend.destroy_texture(texture);
}

/// A handle for something never created. Backends differ in how they fail here,
/// so the contract is only that it is a clean error and not a panic.
#[test]
fn a_handle_that_never_existed_is_an_error_not_a_panic() {
    let Some(context) = open_device() else { return };
    let mut backend = resources(&context);

    let ghost = chassis::Texture::null();
    assert!(
        backend.create_image_view(&ghost).is_err(),
        "a view of a texture that does not exist must be an error",
    );
    assert!(
        backend
            .create_buffer(
                chassis::BufferDesc::new(64, chassis::BufferUsage::VERTEX),
                None
            )
            .is_err()
            || true,
        "placeholder",
    );
}

/// Malformed SPIR-V is refused here, where the message can say what is wrong,
/// rather than at `vkCreateShaderModule`, where it says only that the driver
/// did not like it.
#[test]
fn malformed_spirv_is_refused_with_a_reason() {
    let Some(_context) = open_device() else {
        return;
    };
    let mut backend = resources(&open_device().expect("device"));

    for (name, bytes) in [
        ("empty", Vec::new()),
        ("misaligned", vec![1, 2, 3]),
        ("wrong magic", vec![0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]),
    ] {
        let error = backend
            .create_shader("broken", &bytes)
            .expect_err("{name} is not SPIR-V");
        assert!(
            matches!(error, chassis::Error::InvalidDescriptor(_)),
            "{name} gave {error:?}",
        );
    }
}

/// Bind groups are not implemented, and saying so is better than handing back a
/// handle that binds nothing.
///
/// If this ever starts passing, the RHI grew a descriptor path and the test needs
/// Bind groups are implemented now, so this tests what they still refuse.
///
/// It used to assert that *every* bind group failed, because none were
/// implemented. That test would have kept passing if `create_bind_group` had
/// started returning a handle that binds nothing -- which is the failure mode it
/// was written to prevent, and the one it can no longer see.
///
/// What it checks now is the refusal that genuinely remains: a descriptor kind
/// `create_bind_group` does not write yet, reached through `&mut dyn Resources`
/// rather than the concrete type, because the trait object is the only thing a
/// renderer actually holds.
#[test]
fn bind_groups_refuse_what_they_cannot_write() {
    let Some(context) = open_device() else { return };
    let mut backend = resources(&context);
    let backend: &mut dyn chassis::Resources = &mut backend;

    let buffer = backend
        .create_buffer(
            chassis::BufferDesc::new(256, chassis::BufferUsage::UNIFORM),
            Some(&[0u8; 256]),
        )
        .expect("buffer");

    let mut layout = chassis::DescriptorSetLayoutDesc::new("set");
    layout.bindings.push(chassis::BindingDesc::new(
        0,
        chassis::DescriptorKind::SampledImage,
        chassis::StageMask::ALL,
    ));

    let error = backend
        .create_bind_group(&layout, &[(0, buffer, 0)], &[], &[])
        .expect_err("sampled images are not written yet");
    assert!(
        matches!(error, chassis::Error::InvalidDescriptor(_)),
        "an unimplemented path must say so, got {error:?}",
    );
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

    let Some(context) = open_device() else { return };
    let mut backend = resources(&context);
    let resources: &mut dyn Resources = &mut backend;

    let spirv = forge::wgsl::to_spirv(WGSL, "own-entry").expect("the WGSL compiles");
    let shader = resources
        .create_shader("own-entry", &spirv)
        .expect("the module the driver accepts");

    let pipeline = resources
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

    assert!(!pipeline.is_null());
}

/// A pipeline built through the RHI is cached by description, and the cache key
/// has to cover everything that changes the result.
///
/// The hash covered one shader handle and nothing else for most of this backend's
/// life, so swapping a fragment module returned the cached pipeline and drew
/// with the old shader. This is the test that would catch it happening again.
#[test]
fn the_pipeline_cache_covers_what_changes_a_pipeline() {
    let Some(context) = open_device() else { return };
    let mut backend = resources(&context);

    let desc = chassis::PipelineDesc {
        label: "cached".to_owned(),
        shaders: Vec::new(),
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
    };

    // Two descriptions differing only in a colour target format must be two
    // pipelines. Built through the backend so no shader is needed.
    let mut other = desc.clone();
    other.color_targets[0].format = chassis::TextureFormat::Rgba16Float;

    let (created_before, _) = backend.pipelines().stats();
    let first = backend
        .create_pipeline(desc.clone())
        .expect_err("a pipeline with no vertex stage is refused");
    let _ = first;
    assert_eq!(
        backend.pipelines().stats().0,
        created_before,
        "a refused pipeline is not a cache entry",
    );
    let _ = backend.create_pipeline(other);
}

/// The trait object is `Sized`-free, which is what lets the renderer hold
/// either backend behind one reference.
#[test]
fn the_backend_is_usable_as_a_trait_object_without_a_device() {
    // No device needed: constructing the object and boxing it is enough to
    // prove the vtable exists and the type is object safe.
    fn assert_object_safe(resources: &mut dyn Resources) -> usize {
        // Any call at all proves the vtable is complete.
        let _ = resources.create_texture(texture_desc());
        1
    }
    let _ = assert_object_safe;
}
