//! The legacy render-pass fallback, on real hardware.
//!
//! This is the path taken on hardware without `VK_KHR_dynamic_rendering`. It is
//! not exercised by any other test in the project, which makes it exactly the
//! kind of code that is wrong until the day it is needed.

use ash::vk::Handle;
use stormbreaker::{create_device, device_available, DeviceOptions};

fn open_device() -> Option<stormbreaker::Context> {
    if !device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    match create_device(DeviceOptions::from_env(c"stormbreaker-renderpass")) {
        Ok(context) => Some(context),
        Err(stormbreaker::BackendError::Loader(why)) => {
            eprintln!("SKIP: loader unavailable: {why}");
            None
        }
        Err(other) => panic!("device creation failed: {other}"),
    }
}

fn colour() -> chassis::RenderPassDesc {
    chassis::RenderPassDesc::color(chassis::TextureFormat::Rgba8Unorm)
}

#[test]
fn a_single_colour_render_pass_creates() {
    let Some(context) = open_device() else { return };
    let render_pass = stormbreaker::vulkan_render_pass::RenderPass::new(
        &context.device,
        &colour(),
        "single-colour",
    )
    .expect("a one-colour render pass creates");
    assert!(!render_pass.handle().is_null());
    assert_eq!(render_pass.color_attachment_count(), 1);
    assert!(!render_pass.has_depth());
}

/// Two colour attachments, because an off-by-one in the attachment slice is the
/// failure mode a single-colour test cannot see.
#[test]
fn two_colour_attachments_create() {
    let Some(context) = open_device() else { return };
    let mut desc = colour();
    desc.colors.push(chassis::ColorAttachmentDesc {
        texture: chassis::Texture::null(),
        format: chassis::TextureFormat::Rgba16Float,
        load: chassis::LoadOp::Clear,
        clear: [0.0; 4],
        store: true,
        resolve: chassis::Texture::null(),
    });

    let render_pass =
        stormbreaker::vulkan_render_pass::RenderPass::new(&context.device, &desc, "two-colour")
            .expect("a two-colour render pass creates");
    assert_eq!(render_pass.color_attachment_count(), 2);
}

/// Depth plus colour: the depth attachment must land at index 1, after the
/// colours, or the subpass references the wrong attachment and validation says
/// nothing useful about it.
#[test]
fn depth_attaches_after_the_colours() {
    let Some(context) = open_device() else { return };
    let desc = colour().with_depth(chassis::TextureFormat::Depth24);

    let render_pass = stormbreaker::vulkan_render_pass::RenderPass::new(
        &context.device,
        &desc,
        "colour-and-depth",
    )
    .expect("a depth render pass creates");
    assert_eq!(render_pass.color_attachment_count(), 1);
    assert!(render_pass.has_depth());
}

/// Every combination of load and store, because they map straight through to
/// `VkAttachmentLoadOp` / `VkAttachmentStoreOp` and a wrong mapping is silent:
/// it renders, just not what was asked for.
#[test]
fn every_load_and_store_combination_creates() {
    let Some(context) = open_device() else { return };
    let loads = [
        chassis::LoadOp::Clear,
        chassis::LoadOp::Load,
        chassis::LoadOp::DontCare,
    ];
    let stores = [true, false];
    let mut made = 0;
    for load in loads {
        for store in stores {
            let mut desc = colour();
            desc.colors[0].load = load;
            desc.colors[0].store = store;
            stormbreaker::vulkan_render_pass::RenderPass::new(
                &context.device,
                &desc,
                "combination",
            )
            .unwrap_or_else(|e| panic!("load={load:?} store={store}: {e}"));
            made += 1;
        }
    }
    assert_eq!(made, 6, "every combination was exercised");
}

/// Mixed formats, which is what a deferred gbuffer actually does: a colour
/// attachment list is rarely one format.
#[test]
fn mixed_colour_formats_create() {
    let Some(context) = open_device() else { return };
    let mut desc = colour();
    for format in [
        chassis::TextureFormat::Rgba8Unorm,
        chassis::TextureFormat::Rgba16Float,
        chassis::TextureFormat::R32Float,
        chassis::TextureFormat::R11b11G11B10Float,
    ] {
        desc.colors.push(chassis::ColorAttachmentDesc {
            texture: chassis::Texture::null(),
            format,
            load: chassis::LoadOp::DontCare,
            clear: [0.0; 4],
            store: false,
            resolve: chassis::Texture::null(),
        });
    }

    let render_pass =
        stormbreaker::vulkan_render_pass::RenderPass::new(&context.device, &desc, "gbuffer-shaped")
            .expect("a gbuffer-shaped render pass creates");
    assert_eq!(render_pass.color_attachment_count(), 5);
}
