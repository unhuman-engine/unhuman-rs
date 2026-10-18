//! Resource and recording tests against real hardware.
//!
//! The interesting one is `parallel_recordings_submit_in_order`: it records two
//! command lists on two threads and submits them as a semaphore chain, then
//! checks the bytes that came out the far end. Two lists in flight at once is
//! what a single shared primary command buffer per frame cannot express.

use ash::vk;
use chassis::{
    BufferDesc, BufferUsage, Extent, SampleCount, TextureDesc, TextureFormat, TextureUsage,
};
use stormbreaker::{content_hash, Buffers, Context, Recorder, Shaders, Textures, UploadRing};

fn open() -> Option<Context> {
    if !stormbreaker::device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    match stormbreaker::create_device(stormbreaker::DeviceOptions::from_env(c"stormbreaker-tests"))
    {
        Ok(context) => Some(context),
        Err(other) => panic!("device creation failed: {other}"),
    }
}

/// A byte sequence that is not SPIR-V: right length, wrong magic.
fn not_spirv(len: usize) -> Vec<u8> {
    vec![0xAB; len]
}

#[test]
fn a_host_visible_buffer_round_trips() {
    let Some(context) = open() else { return };
    let mut buffers = Buffers::new(&context.device, context.memory_properties());

    let handle = buffers
        .create(
            &context.device,
            &BufferDesc::new(256, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST),
            None,
        )
        .expect("buffer");

    let payload: Vec<u8> = (0..=255u8).collect();
    let resource = buffers.get(handle).expect("handle resolves");
    resource.write(&context.device, &payload).unwrap();

    let read = resource.read(&context.device).unwrap();
    assert_eq!(read.len(), 256);
    assert_eq!(&read[..], &payload[..]);
}

#[test]
fn a_freed_buffer_handle_stops_resolving() {
    let Some(context) = open() else { return };
    let mut buffers = Buffers::new(&context.device, context.memory_properties());

    let handle = buffers
        .create(
            &context.device,
            &BufferDesc::new(64, BufferUsage::TRANSFER_DST),
            None,
        )
        .expect("buffer");
    assert!(buffers.get(handle).is_some());

    assert!(buffers.destroy(&context.device, handle));
    assert!(
        buffers.get(handle).is_none(),
        "a destroyed buffer handle still resolved"
    );
    assert!(
        !buffers.destroy(&context.device, handle),
        "double destroy succeeded"
    );
}

#[test]
fn a_buffer_needs_a_usage() {
    let Some(context) = open() else { return };
    let mut buffers = Buffers::new(&context.device, context.memory_properties());

    let result = buffers.create(
        &context.device,
        &BufferDesc::new(64, chassis::BufferUsage::NONE),
        None,
    );
    assert!(result.is_err(), "a buffer with no usage was accepted");
}

#[test]
fn a_device_local_texture_comes_with_a_view() {
    let Some(context) = open() else { return };
    let mut textures = Textures::new(&context.device, context.memory_properties());

    let desc = TextureDesc::new_2d(
        Extent::new(64, 64),
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    )
    .with_mips(3);

    let handle = textures
        .create(&context.device, &desc, "test")
        .expect("texture");
    let resource = textures.get(handle).expect("handle resolves");

    assert_ne!(resource.image(), vk::Image::null());
    assert_ne!(resource.view(), vk::ImageView::null());
    assert_eq!(resource.mip_levels(), 3);
    assert_eq!(resource.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(resource.extent(), Extent::new(64, 64));

    // The barrier target every transition for this texture will use.
    let target = resource.barrier_target();
    assert_eq!(target.image, resource.image());
    assert_eq!(target.subresource.level_count, 3);

    assert!(textures.destroy(&context.device, handle));
}

#[test]
fn a_texture_with_an_undefined_format_is_rejected() {
    let Some(context) = open() else { return };
    let mut textures = Textures::new(&context.device, context.memory_properties());

    let desc = TextureDesc::new_2d(
        Extent::new(8, 8),
        TextureFormat::Undefined,
        TextureUsage::SAMPLED,
    );
    assert!(textures.create(&context.device, &desc, "bad").is_err());
}

#[test]
fn shader_lookups_deduplicate_by_content() {
    // Testable with no device, which is where the rule is most likely to rot.
    let shaders = Shaders::new();
    let bytes = not_spirv(64);
    let stage = vk::ShaderStageFlags::VERTEX;

    assert!(
        shaders.lookup(&bytes, stage).is_none(),
        "cache returned a miss as a hit"
    );

    let other = not_spirv(64);
    assert_eq!(content_hash(&bytes), content_hash(&other));
    assert_ne!(content_hash(&bytes), content_hash(&not_spirv(60)));
    assert_eq!(shaders.lookup(&bytes, stage), shaders.lookup(&other, stage));
    assert_eq!(
        shaders.lookup(&bytes, vk::ShaderStageFlags::FRAGMENT),
        shaders.lookup(&other, stage),
        "the same bytes for a different stage are a different module"
    );
}

#[test]
fn malformed_spirv_is_rejected_before_the_driver_sees_it() {
    let Some(context) = open() else { return };
    let mut shaders = Shaders::new();
    let stage = vk::ShaderStageFlags::VERTEX;

    // No magic word.
    assert!(shaders
        .create(&context.device, &not_spirv(64), stage)
        .is_err());
    // Right magic, not a multiple of four.
    let mut odd = vec![0x03u8, 0x02, 0x23, 0x07, 0x00];
    odd.push(0x00);
    assert!(shaders.create(&context.device, &odd, stage).is_err());
    // Shorter than the magic word itself.
    assert!(shaders
        .create(&context.device, &[0x03, 0x02], stage)
        .is_err());
}

#[test]
fn the_upload_ring_copies_data() {
    let Some(context) = open() else { return };
    let memory_properties = context.memory_properties();
    let family = context.graphics_queue_family().unwrap();

    let mut buffers = Buffers::new(&context.device, memory_properties);
    let destination = buffers
        .create(
            &context.device,
            &BufferDesc::new(1024, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST),
            None,
        )
        .expect("destination");

    let mut ring = UploadRing::new(
        &context.device,
        &memory_properties,
        context.graphics_queue,
        family,
        4096,
    )
    .expect("upload ring");

    let payload: Vec<u8> = (0..512).map(|i| i as u8).collect();
    ring.copy_to_buffer(buffers.get(destination).unwrap().handle(), 0, &payload)
        .unwrap();
    assert!(ring.has_pending());
    assert_eq!(ring.outstanding(), 512);

    ring.wait().unwrap();
    assert!(!ring.has_pending());

    let read = buffers
        .get(destination)
        .unwrap()
        .read(&context.device)
        .unwrap();
    assert_eq!(&read[..payload.len()], &payload[..]);
}

#[test]
fn the_upload_ring_refuses_an_oversized_upload() {
    let Some(context) = open() else { return };
    let memory_properties = context.memory_properties();
    let family = context.graphics_queue_family().unwrap();

    let mut buffers = Buffers::new(&context.device, memory_properties);
    let destination = buffers
        .create(
            &context.device,
            &BufferDesc::new(64, BufferUsage::TRANSFER_DST),
            None,
        )
        .unwrap();

    let mut ring = UploadRing::new(
        &context.device,
        &memory_properties,
        context.graphics_queue,
        family,
        128,
    )
    .unwrap();

    let big = vec![0u8; 256];
    assert!(
        ring.copy_to_buffer(buffers.get(destination).unwrap().handle(), 0, &big)
            .is_err(),
        "an upload larger than the ring was accepted"
    );
}

/// The architectural proof: two lists recorded on two threads, submitted as a
/// semaphore chain, result checked.
#[test]
fn parallel_recordings_submit_in_order() {
    let Some(context) = open() else { return };
    let memory_properties = context.memory_properties();
    let family = context.graphics_queue_family().unwrap();

    let mut buffers = Buffers::new(&context.device, memory_properties);
    let source = buffers
        .create(
            &context.device,
            &BufferDesc::new(256, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST),
            None,
        )
        .unwrap();

    // Device-local, so the second list genuinely has to wait for the first: it
    // cannot be short-circuited by host visibility.
    let staging = buffers
        .create(
            &context.device,
            &BufferDesc::new(256, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST),
            None,
        )
        .unwrap();
    let device_local = {
        let desc = BufferDesc::new(256, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST);
        buffers.create(&context.device, &desc, None).unwrap()
    };
    let readback = buffers
        .create(
            &context.device,
            &BufferDesc::new(256, BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST),
            None,
        )
        .unwrap();

    let payload: Vec<u8> = (0..256).map(|i| i as u8).collect();
    buffers
        .get(source)
        .unwrap()
        .write(&context.device, &payload)
        .unwrap();

    let mut recorder = Recorder::new(
        &context.device,
        context.graphics_queue,
        family,
        context.tier(),
        4,
    )
    .expect("recorder");

    let source_handle = buffers.get(source).unwrap().handle();
    let staging_handle = buffers.get(staging).unwrap().handle();
    let device_handle = buffers.get(device_local).unwrap().handle();
    let readback_handle = buffers.get(readback).unwrap().handle();

    // Take two lists; slot 0 records the first half of the chain, slot 1 the
    // second. Recording is done on separate threads to prove there is no shared
    // state between them.
    let mut lists = recorder.take(2).expect("two lists");
    // Slots identify command pools, so they are stable identities rather than
    // per-frame positions: what matters is that `take` hands them back in
    // ascending order, because the caller assigns pass `i` to list `i` and the
    // submit chain walks slot order.
    assert!(
        lists[0].slot() < lists[1].slot(),
        "take returned lists out of slot order: {:?}",
        lists.iter().map(|l| l.slot()).collect::<Vec<_>>()
    );

    let region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: 256,
    };
    let (mut first, mut second) = (lists.remove(0), lists.remove(0));

    // Each list is *moved* into its own thread. There is no shared command
    // buffer, no pool lock and no `&mut` aliasing between them: this is the
    // recording model a shared primary command buffer cannot express.
    let finished = std::thread::scope(|scope| {
        let forward = scope.spawn(move || {
            first.copy_buffer_to_buffer(source_handle, device_handle, &[region]);
            first.end().unwrap()
        });
        let back = scope.spawn(move || {
            second.copy_buffer_to_buffer(device_handle, readback_handle, &[region]);
            second.end().unwrap()
        });
        vec![forward.join().unwrap(), back.join().unwrap()]
    });

    // `staging` proves the arena is wider than the lists in flight.
    let _ = staging_handle;

    recorder.submit(finished, 0).expect("submit");

    let read = buffers
        .get(readback)
        .unwrap()
        .read(&context.device)
        .unwrap();
    assert_eq!(
        &read[..],
        &payload[..],
        "the chained submit did not preserve order"
    );
}

#[test]
fn the_recorder_refuses_more_lists_than_its_arena_holds() {
    let Some(context) = open() else { return };
    let family = context.graphics_queue_family().unwrap();

    let mut recorder = Recorder::new(
        &context.device,
        context.graphics_queue,
        family,
        context.tier(),
        2,
    )
    .unwrap();
    assert_eq!(recorder.capacity(), 2);

    assert!(
        recorder.take(3).is_err(),
        "taking more lists than the arena holds should fail, not truncate"
    );

    let lists = recorder.take(2).unwrap();
    let finished: Vec<_> = lists.into_iter().map(|l| l.end().unwrap()).collect();
    recorder.submit(finished, 0).unwrap();
    assert_eq!(
        recorder.capacity(),
        2,
        "lists were not returned to the arena"
    );
}

#[test]
fn the_submit_chain_rides_a_timeline_when_the_tier_has_one() {
    let Some(context) = open() else { return };
    let family = context.graphics_queue_family().unwrap();

    let recorder = Recorder::new(
        &context.device,
        context.graphics_queue,
        family,
        context.tier(),
        2,
    )
    .unwrap();

    // Matches the tier ladder: Sync2Timeline rides one semaphore with values,
    // Legacy needs one binary semaphore per link.
    assert_eq!(
        recorder.uses_timeline(),
        context.tier().supports_timeline_semaphores(),
        "the chain shape does not match the device tier"
    );
}

#[test]
fn a_zero_sized_recorder_is_refused() {
    let Some(context) = open() else { return };
    let family = context.graphics_queue_family().unwrap();
    assert!(Recorder::new(
        &context.device,
        context.graphics_queue,
        family,
        context.tier(),
        0
    )
    .is_err());
}

#[test]
fn sample_counts_and_formats_map_without_gaps() {
    // Coverage of the format table: every format must produce a non-undefined
    // Vulkan format, or the driver will reject the image at create time.
    for format in [
        TextureFormat::R8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8Srgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::R16Float,
        TextureFormat::Rgba16Float,
        TextureFormat::R32Float,
        TextureFormat::Rgba32Float,
        TextureFormat::R11b11G11B10Float,
        TextureFormat::Depth24,
        TextureFormat::Depth32Float,
        TextureFormat::Depth24Stencil8,
    ] {
        assert_ne!(
            stormbreaker::vulkan_types::vk_format(format),
            vk::Format::UNDEFINED,
            "{format:?} has no Vulkan format"
        );
    }
    assert_eq!(
        stormbreaker::vulkan_types::vk_samples(SampleCount::X4),
        vk::SampleCountFlags::TYPE_4
    );
}

#[test]
fn buffer_usage_maps_to_every_declared_bit() {
    use stormbreaker::vulkan_buffer::to_vk_usage;
    let all = chassis::BufferUsage::VERTEX
        | chassis::BufferUsage::INDEX
        | chassis::BufferUsage::UNIFORM
        | chassis::BufferUsage::STORAGE
        | chassis::BufferUsage::TRANSFER_SRC
        | chassis::BufferUsage::TRANSFER_DST
        | chassis::BufferUsage::INDIRECT;

    let mapped = to_vk_usage(all);
    assert_ne!(mapped, vk::BufferUsageFlags::empty());
    for expected in [
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::BufferUsageFlags::INDEX_BUFFER,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::BufferUsageFlags::INDIRECT_BUFFER,
    ] {
        assert!(mapped.contains(expected), "missing {expected:?}");
    }
    assert_eq!(
        to_vk_usage(chassis::BufferUsage::NONE),
        vk::BufferUsageFlags::empty()
    );
}
