//! Pipeline description, hashing and the description cache.
//!
//! Holds the state objects, hash, description cache and lifetime. Creation is
//! split out:
//!
//! ```text
//!   vulkan_pipeline_state.rs    this file: description, hash, cache, lifetime
//!   vulkan_graphic_pipeline.rs  graphics pipeline creation
//!   vulkan_compute_pipeline.rs  compute pipeline creation
//!   vulkan_pipeline_cache.rs    the driver's pipeline cache
//! ```
//!
//! Two caches exist and they solve different problems, which is why they are two
//! files:
//!
//! * **This one** asks *us* "is this the same description as last time?", which
//!   avoids the `vkCreateGraphicsPipelines` call entirely.
//! * `vulkan_pipeline_cache.rs` asks the *driver* "have you already compiled
//!   this?", which can still cost the call.
//!
//! Only the driver cache exists without this file, and asking it "have you
//! compiled this?" can miss, so every request would recompile.

use std::collections::HashMap;

use ash::vk;
use chassis::{Pipeline, Registry};

use crate::vulkan_descriptor_manager::Descriptors;
use crate::vulkan_error::Result;
use crate::vulkan_shader::content_hash;

/// Hashes everything that affects the compiled pipeline.
///
/// Attribute offsets and formats change generated code, so they are in. The
/// label is not, so renaming a pass does not invalidate its pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PipelineHash(u64);

impl PipelineHash {
    /// From raw bytes. Used by the compute path, whose description shares no
    /// fields with the graphics one.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(content_hash(bytes))
    }

    pub fn of(desc: &chassis::PipelineDesc) -> Self {
        let mut bytes = Vec::new();
        // Every stage. Hashing only the first would mean changing the fragment
        // module left the pipeline cached under the old description, and the
        // draw used a stale shader with no symptom until the pixels were wrong.
        for stage in &desc.shaders {
            bytes.extend_from_slice(&(stage.stage as u32).to_le_bytes());
            bytes.extend_from_slice(&stage.shader.index().to_le_bytes());
            bytes.extend_from_slice(&stage.shader.generation().to_le_bytes());
        }
        // And every set layout. A pipeline that changed which descriptors it
        // reads is a different pipeline, and the old key could not tell.
        for set in &desc.set_layouts {
            for binding in &set.bindings {
                bytes.extend_from_slice(&binding.binding.to_le_bytes());
                bytes.extend_from_slice(&(binding.kind as u32).to_le_bytes());
                bytes.extend_from_slice(&binding.count.to_le_bytes());
                bytes.extend_from_slice(&binding.stages.0.to_le_bytes());
            }
            bytes.push(0xff);
        }
        bytes.push(0xfe);
        bytes.extend_from_slice(&desc.vertex.stride.to_le_bytes());
        for attribute in &desc.vertex.attributes {
            bytes.extend_from_slice(&attribute.location.to_le_bytes());
            bytes.extend_from_slice(
                &attribute
                    .format
                    .bytes_per_texel()
                    .unwrap_or(0)
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(&attribute.offset.to_le_bytes());
        }
        bytes.extend_from_slice(
            &desc
                .index_format
                .map(|f| f as u32)
                .unwrap_or(0xff)
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&desc.primitive.topology.to_le_bytes());
        bytes.extend_from_slice(&desc.primitive.cull_mode.to_le_bytes());
        bytes.push(desc.depth.write_enabled as u8);
        bytes.push(desc.depth.format.is_some() as u8);
        bytes.push(desc.blend.enabled as u8);
        bytes.extend_from_slice(&(desc.color_targets.len() as u32).to_le_bytes());
        for target in &desc.color_targets {
            bytes.extend_from_slice(&(target.format as u32).to_le_bytes());
            bytes.push(target.blend as u8);
            bytes.push(target.write_mask);
        }
        Self(content_hash(&bytes))
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

pub struct PipelineResource {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    label: String,
}

impl PipelineResource {
    pub fn new(pipeline: vk::Pipeline, layout: vk::PipelineLayout, label: &str) -> Self {
        Self {
            pipeline,
            layout,
            label: label.to_string(),
        }
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Owns every pipeline and maps a description hash to the handle for it.
pub struct Pipelines {
    device: ash::Device,
    registry: Registry<PipelineResource, chassis::PipelineTag>,
    by_description: HashMap<PipelineHash, Pipeline>,
    driver_cache: vk::PipelineCache,
    created: u32,
    cache_hits: u32,
    /// Every `VkPipelineLayout` this object created, for teardown. No key:
    /// nothing looks one up, and the set layouts it references live in
    /// `Descriptors`.
    pipeline_layouts: Vec<vk::PipelineLayout>,
}

impl Pipelines {
    pub fn new(device: &ash::Device, driver_cache: vk::PipelineCache) -> Self {
        Self {
            device: device.clone(),
            registry: Registry::new(),
            by_description: HashMap::new(),
            driver_cache,
            created: 0,
            cache_hits: 0,
            pipeline_layouts: Vec::new(),
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn driver_cache(&self) -> vk::PipelineCache {
        self.driver_cache
    }

    pub fn by_description(&self, hash: PipelineHash) -> Option<Pipeline> {
        self.by_description.get(&hash).copied()
    }

    pub fn mark_cache_hit(&mut self, handle: Pipeline) {
        self.cache_hits += 1;
        let _ = handle;
    }

    /// Records a newly compiled pipeline against the description hash that
    /// missed.
    pub fn insert(&mut self, hash: PipelineHash, resource: PipelineResource) -> Pipeline {
        let handle = self.registry.create(resource);
        self.by_description.insert(hash, handle);
        self.created += 1;
        handle
    }

    /// Build the pipeline layout for a description's descriptor sets.
    ///
    /// Set layouts are deduplicated by shape, not by label: two sets that bind
    /// the same things at the same bindings are the same layout, and splitting
    /// them by name means a rename changes the driver objects for no reason.
    /// The `VkPipelineLayout` for a pipeline's declared sets.
    ///
    /// The set layouts come from [`Descriptors`], which is the one place a layout
    /// is created: a bind group asks the same cache for the same description, so
    /// the two cannot disagree. This used to keep a second cache of its own, built
    /// by a second conversion -- which dropped the stage flags and made every set
    /// incompatible with its pipeline.
    pub fn create_layout(
        &mut self,
        desc: &[chassis::DescriptorSetLayoutDesc],
        descriptors: &mut Descriptors,
    ) -> Result<vk::PipelineLayout> {
        let mut set_layouts = Vec::with_capacity(desc.len());
        for set in desc {
            set_layouts.push(descriptors.layout(set)?);
        }

        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let layout = unsafe { self.device.create_pipeline_layout(&info, None) }
            .map_err(|e| crate::vulkan_error::BackendError::vk("vkCreatePipelineLayout", e))?;

        // Recorded before returning: if the pipeline build that follows fails, the
        // layout is already allocated and nothing else will free it.
        self.pipeline_layouts.push(layout);
        Ok(layout)
    }

    /// Destroy every pipeline layout and set layout this object created.
    ///
    /// Called before the device goes away. A leaked `VkDescriptorSetLayout` is
    /// not reported by validation as an error, it is simply gone, and the next
    /// run gets a different handle for the same shape.
    /// Destroy one pipeline and forget its description.
    ///
    /// The pipeline layout is deliberately *not* destroyed here: layouts are
    /// shared by shape across pipelines and are owned by `destroy_layouts`. A
    /// per-pipeline destroy of the layout frees it while another pipeline still
    /// names it, which is a use-after-free the driver finds on the next draw
    /// rather than at the destroy call.
    ///
    /// Removing the description is the part that is easy to forget: leaving it
    /// means the same description returns a handle to freed memory.
    pub fn destroy(&mut self, device: &ash::Device, handle: chassis::Pipeline) -> bool {
        let Some(resource) = self.registry.get(handle) else {
            return false;
        };
        let pipeline = resource.pipeline;
        let removed = self.registry.remove(handle).is_some();
        if removed {
            self.by_description
                .retain(|_, candidate| *candidate != handle);
            unsafe { device.destroy_pipeline(pipeline, None) };
        }
        removed
    }

    /// Destroys every `VkPipelineLayout` this object created.
    ///
    /// The set layouts are not here: they belong to [`Descriptors`], which created
    /// them and destroys them. This object used to own a second set of them, and
    /// two owners of one layout is how one is destroyed while the other still
    /// references it.
    pub fn destroy_layouts(&mut self) {
        for pipeline_layout in self.pipeline_layouts.drain(..) {
            unsafe { self.device.destroy_pipeline_layout(pipeline_layout, None) };
        }
    }

    pub fn get(&self, handle: Pipeline) -> Option<&PipelineResource> {
        self.registry.get(handle)
    }

    /// Pipelines actually compiled, versus requests served from the description
    /// cache. `hits` at zero while a frame asks for the same pipeline repeatedly
    /// means the hash is missing a field.
    pub fn stats(&self) -> (u32, u32) {
        (self.created, self.cache_hits)
    }

    pub fn len(&self) -> usize {
        self.registry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        unsafe {
            // Pipelines only. Their layouts belong to `destroy_layouts`, which
            // owns every layout this object created -- including ones from a
            // pipeline that failed to build and was never registered. Destroying
            // them here as well is a double free:
            // VUID-vkDestroyPipelineLayout-pipelineLayout-parameter, once per
            // pipeline, reported only at teardown.
            for (_, resource) in self.registry.iter() {
                self.device.destroy_pipeline(resource.pipeline, None);
            }
        }
        // The set layouts too. They were added after this `Drop` was written and
        // forgotten: every pipeline leaked one per descriptor set, which
        // validation reports only at `vkDestroyDevice` as
        // VUID-vkDestroyDevice-device-05137 -- an error about teardown that
        // names nothing about which set leaked.
        self.destroy_layouts();
    }
}
