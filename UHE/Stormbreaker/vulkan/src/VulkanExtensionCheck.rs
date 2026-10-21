//! Device capability detection and sync tier selection.
//!
//! Detection is core-promotion-aware, so an extension promoted into Vulkan 1.3
//! counts even when the instance never enumerated it. Every barrier and submit
//! lowers through one point that picks the modern or legacy encoding from the
//! resulting tier.
//!
//! `select_sync_tier` is a pure function, so the tier ladder is testable with no
//! device.

use ash::vk;

/// Which encoding of barriers and submits this device gets.
///
/// **Re-exported from `chassis`, not defined here.** There were two `SyncTier`
/// enums with the same name and the same three variants, one in the RHI and one
/// in the backend, and `GpuInfo::sync_tier` wanted the RHI's while
/// `Capabilities::sync_tier` held the backend's -- so assigning one to the other
/// was a type error with the fix "which one did you mean". One type, one name.
pub use chassis::SyncTier;

/// Packaged Vulkan 1.0 as 0x00MMmmpp.
pub fn version_major(api_version: u32) -> u32 {
    api_version >> 22
}

pub fn version_minor(api_version: u32) -> u32 {
    (api_version >> 12) & 0x3ff
}

/// The tier ladder.
///
/// Synchronization 2 became core in 1.3, so it does not need to be an
/// enumerated extension on a 1.3 device. Timeline semaphores likewise. Getting
/// this wrong shows up as a device that reports the extension but rejects the
/// entry point, or vice versa.
pub fn select_sync_tier(
    api_version: u32,
    synchronization2_extension: bool,
    timeline_extension: bool,
) -> SyncTier {
    let core_sync2 = version_major(api_version) > 1 || version_minor(api_version) >= 3;
    if !(core_sync2 || synchronization2_extension) {
        return SyncTier::Legacy;
    }
    if timeline_extension {
        SyncTier::Sync2Timeline
    } else {
        SyncTier::Sync2
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub graphics: bool,
    pub compute: bool,
    pub transfer: bool,
    pub present: bool,
    pub timestamp_valid_bits: u32,
}

/// Everything this device advertises, enumerated **once**.
///
/// ## Why this is an object and not a function
///
/// A per-question `vkEnumerateDeviceExtensionProperties` call meant eight driver
/// round-trips during one device creation, where a single cache needs one.
///
/// That is not just slow. `has_device_extension` answers "is this string
/// present", which is the wrong question for any extension promoted into core:
/// a 1.3 driver is *not required* to advertise
/// `VK_KHR_synchronization2`. The sync path happened to be right -- it went
/// through [`select_sync_tier`], which checks the API version -- but the other
/// three went through the string check, so a device that had the capability
/// because of its version rather than its extension list was reported as not
/// having it.
///
/// ## The question this answers
///
/// "Is this capability *usable*" -- which is core-or-extension -- not "does this
/// string exist". Every named query below is that question, so a caller cannot
/// ask the weak one by accident.
#[derive(Clone, Debug, Default)]
pub struct ExtensionCheck {
    /// Every extension the device advertised, as strings.
    advertised: Vec<String>,
    api_version: u32,
}

impl ExtensionCheck {
    /// Enumerates once and caches.
    ///
    /// A device that cannot be enumerated gets an **empty** list, not an error:
    /// every query then answers "no", and device creation fails with a capability
    /// message naming what was missing. Failing here would hide that behind an
    /// enumeration error.
    pub fn probe(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        api_version: u32,
    ) -> Self {
        let advertised = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map(|extensions| {
                extensions
                    .iter()
                    .map(|e| {
                        // SAFETY: the driver returns a NUL-terminated name.
                        unsafe { crate::vulkan_instance::name_of(e.extension_name.as_ptr()) }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            advertised,
            api_version,
        }
    }

    /// The raw string query. True only when the extension is *advertised*.
    ///
    /// Public because a caller sometimes genuinely wants the string -- to build
    /// an enable list -- and because hiding it would mean reaching past this type
    /// for `enumerate_device_extension_properties` again.
    pub fn is_advertised(&self, name: &str) -> bool {
        self.advertised.iter().any(|offered| offered == name)
    }

    /// True when the API version is at least `major.minor`.
    pub fn core_at_least(&self, major: u32, minor: u32) -> bool {
        version_major(self.api_version) > major
            || (version_major(self.api_version) == major
                && version_minor(self.api_version) >= minor)
    }

    /// Core-or-extension. The question every named query asks.
    ///
    /// `promoted_in` is the API version the extension was promoted into, or
    /// `None` for an extension that was never promoted -- for which only the
    /// string counts.
    fn supports(&self, name: &str, promoted_in: Option<(u32, u32)>) -> bool {
        if let Some((major, minor)) = promoted_in {
            if self.core_at_least(major, minor) {
                return true;
            }
        }
        self.is_advertised(name)
    }

    /// `vkCmdPipelineBarrier2` and `vkQueueSubmit2`. Core in 1.3, or
    /// `VK_KHR_synchronization2`.
    pub fn supports_sync2(&self) -> bool {
        self.supports("VK_KHR_synchronization2", Some((1, 3)))
    }

    /// Timeline semaphores. Never promoted, so the string is the whole answer.
    pub fn supports_timeline_semaphores(&self) -> bool {
        self.supports("VK_KHR_timeline_semaphore", None)
    }

    /// Dynamic rendering. Core in 1.3, or `VK_KHR_dynamic_rendering`.
    pub fn supports_dynamic_rendering(&self) -> bool {
        self.supports("VK_KHR_dynamic_rendering", Some((1, 3)))
    }

    /// Non-uniform descriptor indexing. Core in 1.2, or
    /// `VK_EXT_descriptor_indexing`.
    pub fn supports_descriptor_indexing(&self) -> bool {
        self.supports("VK_EXT_descriptor_indexing", Some((1, 2)))
    }

    /// Swapchain. Never promoted.
    pub fn supports_swapchain(&self) -> bool {
        self.supports("VK_KHR_swapchain", None)
    }

    /// Memory budget. Never promoted.
    pub fn supports_memory_budget(&self) -> bool {
        self.supports("VK_KHR_memory_budget", None)
    }

    /// The tier this device gets.
    ///
    /// The same ladder `select_sync_tier` implements, kept as a method so the
    /// device object and the free function cannot drift.
    pub fn sync_tier(&self) -> SyncTier {
        select_sync_tier(
            self.api_version,
            self.supports_sync2(),
            self.supports_timeline_semaphores(),
        )
    }

    /// How many extensions were advertised. For tests and diagnostics.
    pub fn advertised_count(&self) -> usize {
        self.advertised.len()
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Capabilities {
    pub device_name: String,
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub sync_tier: SyncTier,
    pub queue_families: Vec<QueueFamily>,
    pub graphics_queue_family: Option<u32>,
    pub dynamic_rendering: bool,
    pub memory_budget: bool,
    pub descriptor_indexing: bool,
    pub max_texture_dimension_2d: u32,
    /// Nanoseconds per timestamp tick, for the GPU profiler.
    pub timestamp_period: f32,
    /// `VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment`.
    ///
    /// Read once at device creation so anything writing a descriptor can refuse a
    /// misaligned offset itself, naming the binding, rather than leaving
    /// validation to report a limit that points at neither the binding nor the
    /// offset that was wrong.
    pub min_uniform_buffer_offset_alignment: u64,
}

impl Capabilities {
    /// Panics-free: a device without a graphics queue is reported as such, and
    /// device creation refuses it rather than dereferencing a `None`.
    pub fn graphics_queue_family(&self) -> Option<u32> {
        self.graphics_queue_family
    }

    /// Frames in flight. Tied to the swapchain image count in practice; the
    /// backend clamps to what the surface reports.
    pub fn frames_in_flight(&self) -> u32 {
        2
    }
}

#[cfg(test)]
mod extension_check_tests {
    use super::*;

    fn check(version: (u32, u32), advertised: &[&str]) -> ExtensionCheck {
        ExtensionCheck {
            advertised: advertised.iter().map(|s| (*s).to_owned()).collect(),
            api_version: vk::make_api_version(0, version.0, version.1, 0),
        }
    }

    /// **This is the bug the type exists to prevent.**
    ///
    /// A Vulkan 1.3 driver is *not required* to advertise
    /// `VK_KHR_synchronization2` -- the extension was promoted into core, and
    /// drivers stopped listing promoted extensions. `has_device_extension` asked
    /// only the string, so it would have answered "no" on a device that has the
    /// capability because of its version.
    #[test]
    fn a_promoted_extension_counts_by_core_version_alone() {
        let device = check((1, 3), &[]);
        assert!(
            device.supports_sync2(),
            "1.3 core is sync2, advertised or not"
        );
        assert!(
            device.supports_dynamic_rendering(),
            "1.3 core is dynamic rendering"
        );
        assert!(
            device.supports_descriptor_indexing(),
            "1.2 core is descriptor indexing"
        );
    }

    /// And the pre-promotion case still needs the string.
    #[test]
    fn below_the_promotion_the_string_is_the_only_answer() {
        let device = check((1, 2), &["VK_KHR_synchronization2"]);
        assert!(device.supports_sync2(), "the extension is advertised");

        let bare = check((1, 2), &[]);
        assert!(!bare.supports_sync2(), "1.2 with no extension is legacy");
        assert_eq!(bare.sync_tier(), SyncTier::Legacy);
    }

    /// Timeline semaphores were never promoted, so a version alone buys nothing.
    ///
    /// Which is the other half of the rule: `promoted_in: None` means the string
    /// is the whole answer, and getting that wrong would report timeline support
    /// on every Vulkan 1.3 device ever shipped.
    #[test]
    fn a_never_promoted_extension_ignores_the_version() {
        assert!(!check((1, 3), &[]).supports_timeline_semaphores());
        assert!(check((1, 3), &["VK_KHR_timeline_semaphore"]).supports_timeline_semaphores());
    }

    #[test]
    fn the_string_query_is_still_available_and_still_only_about_strings() {
        let device = check((1, 3), &["VK_KHR_timeline_semaphore"]);
        assert!(device.is_advertised("VK_KHR_timeline_semaphore"));
        assert!(!device.is_advertised("VK_KHR_synchronization2"));
        assert_eq!(device.advertised_count(), 1);
    }

    /// The object and the pure function must agree, or a caller gets a different
    /// tier depending on which one it reached for.
    #[test]
    fn the_object_and_the_function_agree() {
        for &(version, sync2, timeline) in &[
            ((1, 1), false, false),
            ((1, 2), true, false),
            ((1, 3), false, false),
            ((1, 3), false, true),
        ] {
            let advertised: Vec<&str> = [
                sync2.then_some("VK_KHR_synchronization2"),
                timeline.then_some("VK_KHR_timeline_semaphore"),
            ]
            .into_iter()
            .flatten()
            .collect();
            let device = check(version, &advertised);
            assert_eq!(
                device.sync_tier(),
                select_sync_tier(
                    vk::make_api_version(0, version.0, version.1, 0),
                    sync2,
                    timeline
                ),
                "version {version:?} sync2={sync2} timeline={timeline}",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_decodes() {
        assert_eq!(version_major(vk_make_version(1, 3, 281)), 1);
        assert_eq!(version_minor(vk_make_version(1, 3, 281)), 3);
        assert_eq!(version_major(vk_make_version(1, 1, 0)), 1);
        assert_eq!(version_minor(vk_make_version(1, 1, 0)), 1);
    }

    #[test]
    fn legacy_device_gets_legacy_encodings() {
        let tier = select_sync_tier(vk_make_version(1, 1, 0), false, false);
        assert_eq!(tier, SyncTier::Legacy);
        assert!(!tier.supports_sync2());
        assert!(!tier.supports_timeline_semaphores());
    }

    #[test]
    fn synchronization2_extension_promotes_a_1_1_device() {
        let tier = select_sync_tier(vk_make_version(1, 1, 0), true, false);
        assert_eq!(tier, SyncTier::Sync2);
        assert!(tier.supports_sync2());
        assert!(!tier.supports_timeline_semaphores());
    }

    #[test]
    fn core_promotion_is_enough_on_1_3() {
        // No extension enumerated, but 1.3 has synchronization 2 in core.
        let tier = select_sync_tier(vk_make_version(1, 3, 0), false, false);
        assert_eq!(tier, SyncTier::Sync2);
    }

    #[test]
    fn timeline_is_the_top_tier() {
        let tier = select_sync_tier(vk_make_version(1, 3, 0), false, true);
        assert_eq!(tier, SyncTier::Sync2Timeline);
        assert!(tier.supports_timeline_semaphores());
    }

    const fn vk_make_version(major: u32, minor: u32, patch: u32) -> u32 {
        (major << 22) | (minor << 12) | patch
    }
}
