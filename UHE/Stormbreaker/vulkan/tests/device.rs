//! Real device tests.
//!
//! These run against whatever GPU is present. When there is no loader or no
//! device they skip rather than fail, which is the same reason the workspace has
//! `default-members`: the pure logic is covered everywhere, and the hardware is
//! covered wherever it exists.

use stormbreaker::vulkan_extension_check::{version_major, version_minor};
use stormbreaker::{create_device, device_available, BackendError, DeviceOptions, SyncTier};

/// Returns a device, or `None` when this machine cannot run the test.
fn open_device() -> Option<stormbreaker::Context> {
    if !device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    match create_device(DeviceOptions::from_env(c"stormbreaker-tests")) {
        Ok(context) => Some(context),
        Err(BackendError::Loader(why)) => {
            eprintln!("SKIP: loader unavailable: {why}");
            None
        }
        Err(other) => panic!("device creation failed: {other}"),
    }
}

#[test]
fn a_real_device_reports_itself() {
    let Some(context) = open_device() else {
        return;
    };
    let caps = &context.capabilities;

    assert!(
        !caps.device_name.is_empty(),
        "driver reported no device name"
    );
    assert!(
        at_least_1_1(caps.api_version),
        "device reports api {} which is below the 1.1 we requested",
        version_major(caps.api_version)
    );
    assert!(caps.max_texture_dimension_2d >= 2048);
    assert!(
        caps.graphics_queue_family.is_some(),
        "no queue family was marked graphics"
    );
    assert!(
        caps.queue_families.iter().any(|f| f.graphics),
        "queue families disagree with the chosen one"
    );

    eprintln!(
        "device: {} api {}.{} tier {:?}",
        caps.device_name,
        version_major(caps.api_version),
        version_minor(caps.api_version),
        caps.sync_tier
    );
}

#[test]
fn the_graphics_family_is_really_graphics_capable() {
    let Some(context) = open_device() else {
        return;
    };
    let family = context.graphics_queue_family().expect("graphics family");

    let properties = context.queue_properties();
    let family = properties
        .get(family as usize)
        .expect("chosen family is out of range");

    assert!(
        family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS),
        "queue family {} has no graphics bit",
        context.graphics_queue_family().unwrap()
    );
}

#[test]
fn the_tier_follows_what_the_device_reports() {
    let Some(context) = open_device() else {
        return;
    };
    let caps = &context.capabilities;

    // A 1.3 device with synchronization 2 must not be driven through the legacy
    // path. Everything modern we have shipped so far qualifies.
    if at_least_1_3(caps.api_version) {
        assert!(
            caps.sync_tier.supports_sync2(),
            "1.3 device fell back to {:?}",
            caps.sync_tier
        );
    }

    if caps.sync_tier == SyncTier::Legacy {
        eprintln!("note: legacy tier, so no timeline semaphores are enabled");
    } else {
        // Enabling sync 2 or timeline without the matching feature bit is a
        // validation error, so a non-legacy tier means creation succeeded with
        // those bits set.
        assert!(caps.sync_tier.supports_sync2());
    }
}

#[test]
fn memory_properties_are_populated() {
    let Some(context) = open_device() else {
        return;
    };
    let memory = context.memory_properties();

    assert!(
        memory.memory_heap_count > 0,
        "device reported no memory heaps"
    );
    // Vulkan imposes no relationship between memory *types* and memory *heaps*:
    // a device commonly reports many types against few heaps. It also permits
    // zero-sized heap slots, so the invariant that matters for an allocator is
    // simply that at least one heap is usable.
    assert!(
        !memory.memory_types.is_empty(),
        "device reported no memory types"
    );
    assert!(
        memory.memory_heaps.iter().any(|heap| heap.size > 0),
        "no memory heap has any size, so nothing can be allocated"
    );
}

#[test]
fn a_device_can_be_created_and_destroyed_repeatedly() {
    // Two live contexts must be independent: a process-wide device handle would
    // let a second context overwrite the first one's, and the first one's
    // destructor would then free memory the second now owns.
    let (Some(first), Some(second)) = (open_device(), open_device()) else {
        return;
    };

    assert_ne!(
        first.physical_device, second.physical_device,
        "two contexts ended up on the same physical device"
    );
    assert_eq!(
        first.device_name(),
        second.device_name(),
        "both contexts should land on the same GPU here"
    );
}

#[test]
fn a_dropped_device_leaves_the_next_one_intact() {
    let Some(previous) = open_device() else {
        return;
    };
    let previous_name = previous.device_name().to_string();
    drop(previous);

    let Some(current) = open_device() else {
        return;
    };
    assert_eq!(
        current.device_name(),
        previous_name,
        "create/destroy/create cycle changed the device"
    );
}

fn at_least_1_1(api_version: u32) -> bool {
    version_major(api_version) > 1
        || (version_major(api_version) == 1 && version_minor(api_version) >= 1)
}

fn at_least_1_3(api_version: u32) -> bool {
    version_major(api_version) > 1
        || (version_major(api_version) == 1 && version_minor(api_version) >= 3)
}
