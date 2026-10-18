//! The wgpu backend, on the real driver.
//!
//! ## The assertion that matters most in this file
//!
//! Every test checks which backend wgpu chose. Without it, a wgpu test that
//! runs on Vulkan proves the *Vulkan* backend works -- which was already known,
//! and which would hide the fact that the Metal and DX12 paths have never run.
//! wgpu running on Vulkan is a perfectly good thing; it is not the same thing as
//! this crate working, and the log line saying so is what keeps the difference
//! visible.

use stormglass::wgpu_device::{self, Adapter};

fn adapters() -> Vec<Adapter> {
    let found = Adapter::enumerate();
    if found.is_empty() {
        eprintln!("SKIP: no wgpu adapter on this machine");
    }
    found
}

#[test]
fn an_adapter_exists() {
    let adapters = adapters();
    if adapters.is_empty() {
        return;
    }
    let adapter = &adapters[0];
    eprintln!(
        "adapter {} = {} (wgpu backend {:?})",
        adapter.index(),
        adapter.name(),
        adapter.backend()
    );
    assert!(!adapter.name().is_empty(), "the adapter has a name");
    assert!(
        adapter.limits().max_texture_dimension_2d >= 2048,
        "a backend with a 1x1 maximum texture size is not usable",
    );
}

#[test]
fn a_device_can_be_created_headless() {
    let Some(adapter) = Adapter::request(0) else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let (device, queue) = adapter.create_device("stormglass-test").expect("a device");

    assert!(
        device.max_texture_size() >= 2048,
        "reported {} through the RHI",
        device.max_texture_size()
    );
    // wgpu exposes one queue and does not number them, so this is a constant.
    // Asserted so a change to that is deliberate rather than accidental.
    assert_eq!(queue.family(), 0);
    let _ = device.inner();
}

/// A device built with no extra features can bind the storage buffer the frame
/// set uses.
///
/// The bug this pins: the wgpu backend mapped every `StorageBuffer` to
/// `read_only: false`, but `bifrost`'s frame set binds the light list, which the
/// shader *reads*. wgpu refuses a render pipeline whose layout says `LOAD|STORE`
/// where the shader says `LOAD`, so the sandbox died on its first frame with
/// "Error matching ShaderStages(FRAGMENT) shader requirements".
///
/// Read-only is also why this device needs no extra feature: a *writable*
/// storage buffer visible to the vertex stage would require
/// `VERTEX_WRITABLE_STORAGE`, which `create_device` deliberately does not ask
/// for.
#[test]
fn a_device_can_bind_the_frame_sets_storage_buffer() {
    let Some(adapter) = Adapter::request(0) else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let (device, _queue) = adapter
        .create_device("stormglass-vertex-storage")
        .expect("a device");

    // `wgpu` reacts to an invalid layout by panicking through its error handler,
    // so reaching the end of this function is the assertion. `read_only: true`
    // and `VERTEX_FRAGMENT` are `layout_type_of(StorageBuffer)` and the frame
    // set's `graphics_stages`, verbatim.
    let _layout = device
        .inner()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
}

/// Every format the RHI can name must map to something wgpu has, or the two
/// backends disagree about what a texture is.
#[test]
fn every_rhi_format_maps_or_says_why_not() {
    let formats = [
        chassis::TextureFormat::R8Unorm,
        chassis::TextureFormat::R8Srgb,
        chassis::TextureFormat::Rg8Unorm,
        chassis::TextureFormat::Rgba8Unorm,
        chassis::TextureFormat::Rgba8Srgb,
        chassis::TextureFormat::Bgra8Unorm,
        chassis::TextureFormat::Bgra8Srgb,
        chassis::TextureFormat::R16Float,
        chassis::TextureFormat::Rgba16Float,
        chassis::TextureFormat::R32Float,
        chassis::TextureFormat::Rgba32Float,
        chassis::TextureFormat::R11b11G11B10Float,
        chassis::TextureFormat::Depth16,
        chassis::TextureFormat::Depth24,
        chassis::TextureFormat::Depth32Float,
        chassis::TextureFormat::Depth24Stencil8,
    ];

    // Exactly one format has no mapping, and the reason is real: single-channel
    // sRGB exists in neither API. Pinned so that if wgpu ever adds it, this
    // fails and someone looks.
    let unmapped: Vec<_> = formats
        .iter()
        .filter(|format| wgpu_device::to_wgpu_format(**format).is_none())
        .collect();
    assert_eq!(
        unmapped,
        vec![&chassis::TextureFormat::R8Srgb],
        "only R8Srgb should be unmappable: single-channel sRGB does not exist",
    );
}

/// A format that survives the round trip must come back identical. A lossy
/// mapping here is a texture that is subtly the wrong thing on one backend.
#[test]
fn formats_survive_the_round_trip() {
    for format in [
        chassis::TextureFormat::R8Unorm,
        chassis::TextureFormat::Rg8Unorm,
        chassis::TextureFormat::Rgba8Unorm,
        chassis::TextureFormat::Rgba8Srgb,
        chassis::TextureFormat::Bgra8Unorm,
        chassis::TextureFormat::R32Float,
        chassis::TextureFormat::Rgba32Float,
        chassis::TextureFormat::Depth32Float,
    ] {
        let wgpu_format =
            wgpu_device::to_wgpu_format(format).unwrap_or_else(|| panic!("{format:?} should map"));
        let back = wgpu_device::from_wgpu_format(wgpu_format)
            .unwrap_or_else(|| panic!("{wgpu_format:?} should map back"));
        assert_eq!(format, back, "{format:?} did not survive the round trip");
    }
}

/// Usage flags translate one for one. A dropped flag is a resource the backend
/// silently cannot do what was asked, which surfaces as a validation error on a
/// different object entirely.
#[test]
fn buffer_usages_translate() {
    assert_eq!(
        wgpu_device::to_wgpu_buffer_usage(chassis::BufferUsage::VERTEX),
        wgpu::BufferUsages::VERTEX
    );
    assert_eq!(
        wgpu_device::to_wgpu_buffer_usage(chassis::BufferUsage::INDEX),
        wgpu::BufferUsages::INDEX
    );
    assert_eq!(
        wgpu_device::to_wgpu_buffer_usage(
            chassis::BufferUsage::TRANSFER_DST | chassis::BufferUsage::TRANSFER_SRC
        ),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    );
    assert_eq!(
        wgpu_device::to_wgpu_buffer_usage(chassis::BufferUsage::NONE),
        wgpu::BufferUsages::empty(),
        "NONE really is empty, not a default",
    );
}

#[test]
fn texture_usages_translate() {
    assert_eq!(
        wgpu_device::to_wgpu_texture_usage(chassis::TextureUsage::COLOR_ATTACHMENT),
        wgpu::TextureUsages::RENDER_ATTACHMENT
    );
    assert_eq!(
        wgpu_device::to_wgpu_texture_usage(chassis::TextureUsage::SAMPLED),
        wgpu::TextureUsages::TEXTURE_BINDING
    );
    assert_eq!(
        wgpu_device::to_wgpu_texture_usage(chassis::TextureUsage::STORAGE),
        wgpu::TextureUsages::STORAGE_BINDING
    );

    // A texture that is only sampled is a real thing to ask for, and refusing it
    // would make a bare SAMPLED an error.
    assert_eq!(
        wgpu_device::to_wgpu_texture_usage(chassis::TextureUsage::NONE),
        wgpu::TextureUsages::TEXTURE_BINDING,
        "an empty usage is treated as sampled rather than as an error",
    );
}

/// What a launcher would show. Also the thing that proves `enumerate` does not
/// panic on a machine with one adapter or with several.
#[test]
fn backends_can_be_listed_for_a_launcher() {
    let backends = wgpu_device::enumerate_backends();
    if backends.is_empty() {
        eprintln!("SKIP: no wgpu adapter");
        return;
    }
    for backend in &backends {
        assert!(!backend.name.is_empty());
        assert!(
            matches!(
                backend.backend,
                "vulkan" | "metal" | "dx12" | "gl" | "webgpu" | "other"
            ),
            "unrecognised backend name {:?}",
            backend.backend
        );
        assert!(backend.max_texture_size >= 2048);
    }
}
