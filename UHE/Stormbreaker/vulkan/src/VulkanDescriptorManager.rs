//! Descriptor management.
//!
//! Owns the layouts, the pools and the sets, and is the only entry point the
//! rest of the backend uses:
//!
//! ```text
//!   vulkan_descriptor_set.rs    DescriptorKind, DescriptorSet, writes
//!   vulkan_descriptor_pool.rs   pool sizes and allocation
//!   this file                   caching, lifetime, and the mapping to chassis
//! ```
//!
//! Two deliberate properties:
//!
//! * **Layouts are cached by a structural hash**, so two identical requests share
//!   one `VkDescriptorSetLayout` instead of allocating a layout per bind group.
//! * **Sets are per-pass and bound at an index.** There is no global set, so the
//!   failure mode where every `BindPipeline` rebinds one process-wide set has no
//!   representation.

use std::collections::HashMap;

use ash::vk;
use chassis::{BindGroup, Registry};

use crate::vulkan_descriptor_pool::{DescriptorPool, PoolSizes};
use crate::vulkan_descriptor_set::{self, DescriptorSet};
use crate::vulkan_error::{BackendError, Result};

pub use crate::vulkan_descriptor_set::DescriptorKind as Kind;

/// Sets one pool holds before another is created. Large enough that a frame's
/// worth of bind groups fits in one, small enough that a pool's descriptor memory
/// is not a per-frame allocation.
const POOL_MAX_SETS: u32 = 256;

/// One cached layout, next to the driver object.
struct LayoutEntry {
    layout: vk::DescriptorSetLayout,
    /// Layouts may not be freed while a pipeline referencing them is alive, so
    /// they are counted rather than destroyed on last use.
    ref_count: u32,
}

pub struct Descriptors {
    device: ash::Device,
    layouts: HashMap<u64, LayoutEntry>,
    pool_sizes: PoolSizes,
    /// Every pool created, and which one to allocate from.
    ///
    /// A pool is never destroyed to make room. A live bind group holds a
    /// `VkDescriptorSet` allocated from one, and destroying the pool frees it
    /// under the caller -- which is what wiped the handle registry and made every
    /// earlier bind group alias a later one.
    pools: Vec<DescriptorPool>,
    current: Option<usize>,
    sets: Registry<DescriptorSet, chassis::BindGroupTag>,
    layout_hits: u32,
    layout_misses: u32,
}

impl Descriptors {
    pub fn new(device: &ash::Device) -> Self {
        Self {
            device: device.clone(),
            layouts: HashMap::new(),
            pool_sizes: PoolSizes::default(),
            pools: Vec::new(),
            current: None,
            sets: Registry::new(),
            layout_hits: 0,
            layout_misses: 0,
        }
    }

    /// A layout for this binding set, created on first sight.
    ///
    /// Keyed by the description's own hash and built from it by the same
    /// conversion a pipeline uses -- one description, one layout, whichever side
    /// asks first. Building it from a second, stage-less copy of the description
    /// is what made every set on this backend incompatible with its pipeline.
    pub fn layout(
        &mut self,
        desc: &chassis::DescriptorSetLayoutDesc,
    ) -> Result<vk::DescriptorSetLayout> {
        let hash = desc.hash();
        if let Some(entry) = self.layouts.get_mut(&hash) {
            entry.ref_count += 1;
            self.layout_hits += 1;
            return Ok(entry.layout);
        }
        self.layout_misses += 1;

        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = desc
            .bindings
            .iter()
            .map(|binding| vk::DescriptorSetLayoutBinding {
                binding: binding.binding,
                descriptor_type: vulkan_descriptor_set::descriptor_type_of(binding.kind),
                descriptor_count: binding.count.max(1),
                stage_flags: vulkan_descriptor_set::stage_flags_of(binding.stages),
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            })
            .collect();
        bindings.sort_by_key(|binding| binding.binding);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }
            .map_err(|e| BackendError::vk("vkCreateDescriptorSetLayout", e))?;

        // The sizes grew, so the pool in use may not cover the new descriptor
        // type. The next allocation makes a new pool; this one stays alive for the
        // sets already allocated from it.
        self.pool_sizes.account_for(&desc.bindings);
        self.current = None;

        self.layouts.insert(
            hash,
            LayoutEntry {
                layout,
                ref_count: 1,
            },
        );
        Ok(layout)
    }

    /// Allocates a set. Rebuilds the pool when the accumulated sizes changed or
    /// the current one is full.
    /// The index of a pool with room for another set.
    ///
    /// A new pool is created when the current one is full or was invalidated by a
    /// new descriptor type. The old pool is kept: its sets are live.
    fn pool_index(&mut self) -> Result<usize> {
        if let Some(index) = self.current {
            if self.pools[index].has_room() {
                return Ok(index);
            }
        }

        let pool =
            DescriptorPool::create(&self.device, &self.pool_sizes.as_slice(), POOL_MAX_SETS)?;
        self.pools.push(pool);
        let index = self.pools.len() - 1;
        self.current = Some(index);
        Ok(index)
    }

    /// Allocates a set from `layout`.
    pub fn allocate(&mut self, layout: vk::DescriptorSetLayout) -> Result<BindGroup> {
        let index = self.pool_index()?;
        let set = self.pools[index].allocate(layout)?;
        Ok(self.sets.create(DescriptorSet::new(set, layout, index)))
    }

    /// Frees a bind group's set and forgets its handle.
    ///
    /// The registry entry goes with the set, so a handle used after this resolves
    /// to nothing -- which is the difference between a stale handle that is
    /// reported and one that silently aliases whatever was allocated next.
    pub fn destroy(&mut self, handle: BindGroup) -> bool {
        let Some(entry) = self.sets.remove(handle) else {
            return false;
        };
        if let Some(pool) = self.pools.get_mut(entry.pool()) {
            pool.free(entry.handle());
        }
        true
    }

    /// Applies writes to an allocated set.
    pub fn update(
        &mut self,
        bind_group: BindGroup,
        buffer_writes: &[vk::WriteDescriptorSet<'_>],
        image_writes: &[vk::WriteDescriptorSet<'_>],
    ) -> Result<()> {
        let Some(set) = self.sets.get(bind_group).copied() else {
            return Err(BackendError::InvalidHandle);
        };
        vulkan_descriptor_set::update(&self.device, set, buffer_writes, image_writes)
    }

    pub fn set(&self, bind_group: BindGroup) -> Option<DescriptorSet> {
        self.sets.get(bind_group).copied()
    }

    /// Distinct layouts created. A count that climbs every frame means the hash
    /// is missing a field and each pass is allocating its own layout.
    pub fn layout_count(&self) -> usize {
        self.layouts.len()
    }

    /// Layout requests served from cache versus created. `hits` staying at zero
    /// across frames means the cache is not being consulted.
    pub fn layout_stats(&self) -> (u32, u32) {
        (self.layout_hits, self.layout_misses)
    }

    pub fn pool(&self) -> Option<vk::DescriptorPool> {
        self.current
            .and_then(|index| self.pools.get(index))
            .map(DescriptorPool::handle)
    }

    pub fn set_count(&self) -> usize {
        self.sets.len()
    }
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        // Pools first: they own the sets, and freeing layouts while a pool
        // references them is what the destruction order rules forbid.
        self.pools.clear();
        self.current = None;
        unsafe {
            for entry in self.layouts.values() {
                self.device
                    .destroy_descriptor_set_layout(entry.layout, None);
            }
        }
    }
}
