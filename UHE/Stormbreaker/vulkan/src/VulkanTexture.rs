//! Textures and image views.
//!
//! A texture carries no layout field: the render graph is the only authority on
//! image layout. The texture knows its format and extent, and transitions arrive
//! as barriers it never has to reason about, so the two cannot disagree.

use ash::vk;
use chassis::{Extent, Registry, Texture, TextureDesc, TextureFormat, TextureUsage};

use crate::vulkan_allocation;
use crate::vulkan_error::{BackendError, Result};
use crate::vulkan_types;

pub struct TextureResource {
    image: vk::Image,
    device_memory: vk::DeviceMemory,
    view: vk::ImageView,
    format: TextureFormat,
    extent: Extent,
    mip_levels: u32,
    label: String,
}

impl TextureResource {
    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// The image plus the subresource range every barrier for this texture uses.
    ///
    /// The aspect comes from the texture's own format. `Target::color` would say
    /// "colour" for a depth image, and the barrier encoder intersects that with
    /// the format's aspects, so the depth buffer's barriers carried an empty
    /// aspect mask -- `VUID-VkImageSubresourceRange-aspectMask-requiredbitmask`,
    /// which validation reports as a barrier with no aspect at all.
    pub fn barrier_target(&self) -> crate::vulkan_barrier_encoder::Target {
        crate::vulkan_barrier_encoder::Target::for_format(
            self.image,
            self.format,
            self.mip_levels,
            1,
        )
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.device_memory, None);
        }
    }
}

pub struct Textures {
    device: ash::Device,
    registry: Registry<TextureResource, chassis::TextureTag>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Extra views made by `create_view`. The view inside a `TextureResource`
    /// dies with the resource; these do not.
    views: Vec<crate::vulkan_image_view::ImageView>,
}

impl Textures {
    pub fn new(
        device: &ash::Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        Self {
            device: device.clone(),
            registry: Registry::new(),
            memory_properties,
            views: Vec::new(),
        }
    }

    pub fn create(
        &mut self,
        device: &ash::Device,
        desc: &TextureDesc,
        label: &str,
    ) -> Result<Texture> {
        if desc.format == TextureFormat::Undefined {
            return Err(BackendError::EmptyUsage {
                what: "texture format",
            });
        }
        let usage = to_vk_usage(desc.usage);
        if usage.is_empty() {
            return Err(BackendError::EmptyUsage {
                what: "texture usage",
            });
        }
        let level_count = desc.mip_levels.max(1);

        let image_info = vk::ImageCreateInfo {
            image_type: if desc.extent.depth > 1 {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            },
            format: vulkan_types::vk_format(desc.format),
            extent: vk::Extent3D {
                width: desc.extent.width,
                height: desc.extent.height,
                depth: desc.extent.depth.max(1),
            },
            mip_levels: level_count,
            array_layers: 1,
            samples: vulkan_types::vk_samples(desc.samples),
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = match unsafe { device.create_image(&image_info, None) } {
            Ok(image) => image,
            Err(e) => return Err(BackendError::vk("vkCreateImage", e)),
        };

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let device_memory = match vulkan_allocation::create_image_memory(
            device,
            &self.memory_properties,
            requirements.memory_type_bits,
            requirements.size,
        ) {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                return Err(e);
            }
        };

        if let Err(e) = unsafe { device.bind_image_memory(image, device_memory, 0) } {
            unsafe {
                device.free_memory(device_memory, None);
                device.destroy_image(image, None);
            }
            return Err(BackendError::vk("vkBindImageMemory", e));
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            // UNDEFINED is the only legal layout for a view created alongside
            // the image; the graph transitions it later.
            .format(vulkan_types::vk_format(desc.format))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vulkan_types::aspect(desc.format),
                base_mip_level: 0,
                level_count,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = match unsafe { device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(e) => {
                unsafe {
                    device.free_memory(device_memory, None);
                    device.destroy_image(image, None);
                }
                return Err(BackendError::vk("vkCreateImageView", e));
            }
        };

        Ok(self.registry.create(TextureResource {
            image,
            device_memory,
            view,
            format: desc.format,
            extent: desc.extent,
            mip_levels: level_count,
            label: label.to_string(),
        }))
    }

    pub fn get(&self, handle: Texture) -> Option<&TextureResource> {
        self.registry.get(handle)
    }

    /// A second view of an existing texture.
    ///
    /// `create` already makes one view alongside the image, so this exists for
    /// the case the RHI needs and the texture did not anticipate: a sampler
    /// wants a view with a different aspect or mip range than a render target
    /// does. Returning the existing view would be wrong for that, so a new one is
    /// made and tracked for destruction.
    ///
    /// A null handle is refused rather than producing a null view, which the
    /// driver accepts and the frame then reads as a zero-sized attachment.
    pub fn create_view(
        &mut self,
        device: &ash::Device,
        handle: Texture,
    ) -> Result<crate::vulkan_image_view::ImageView> {
        let Some(resource) = self.registry.get(handle) else {
            return Err(BackendError::InvalidHandle);
        };
        let view = crate::vulkan_image_view::create_default(
            device,
            resource.image(),
            vulkan_types::vk_format(resource.format()),
            vulkan_types::aspect(resource.format()),
            resource.mip_levels(),
            1,
        )?;
        self.views.push(view);
        Ok(view)
    }

    /// Views made by `create_view`, destroyed with the textures.
    ///
    /// Held here rather than by the caller because the RHI's
    /// `create_image_view` hands back a bare `vk::ImageView` and has no destroy
    /// for it -- a view that outlives its texture is a use-after-free, and one
    /// that never dies is a leak validation only mentions at `vkDestroyDevice`.
    pub fn views(&self) -> &[crate::vulkan_image_view::ImageView] {
        &self.views
    }

    pub fn destroy(&mut self, device: &ash::Device, handle: Texture) -> bool {
        match self.registry.remove(handle) {
            Some(resource) => {
                resource.destroy(device);
                true
            }
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.registry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Texture, &TextureResource)> {
        self.registry.iter()
    }
}

impl Drop for Textures {
    fn drop(&mut self) {
        // Views first. A view must not outlive the image it views, and the
        // images are destroyed on the next line.
        //
        // A missed `create_view` view leaks a bare `VkImageView`, which validation
        // reports only at teardown as VUID-vkDestroyDevice-device-05137, naming no
        // object.
        for view in self.views.drain(..) {
            crate::vulkan_image_view::destroy(&self.device, view.handle());
        }
        for (_, resource) in self.registry.iter() {
            resource.destroy(&self.device);
        }
    }
}

pub fn to_vk_usage(usage: TextureUsage) -> vk::ImageUsageFlags {
    let mut out = vk::ImageUsageFlags::empty();
    if usage.contains(TextureUsage::SAMPLED) {
        out |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(TextureUsage::STORAGE) {
        out |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.contains(TextureUsage::COLOR_ATTACHMENT) {
        out |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(TextureUsage::DEPTH_ATTACHMENT) {
        out |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if usage.contains(TextureUsage::TRANSFER_SRC) {
        out |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(TextureUsage::TRANSFER_DST) {
        out |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    out
}
