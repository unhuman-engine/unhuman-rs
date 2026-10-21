//! A compute dispatch, on the real driver.
//!
//! The shader writes one word and the test reads it back. A dispatch that
//! returns without error is not evidence: it can be recorded into a buffer that
//! is never submitted, bound to a pipeline created for the wrong bind point, or
//! rejected by validation after the count was incremented. The value is the
//! assertion.

use chassis::{CommandList, Resources};
use stormbreaker::vulkan_command_buffer::CommandList as VulkanCommandList;
use stormbreaker::vulkan_frame_context::Recorder;
use stormbreaker::{create_device, device_available, DeviceOptions};

const WRITTEN: u32 = 7;

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

fn compile_compute() -> Vec<u8> {
    let compiler =
        forge::slangc::SlangcCompiler::new(forge::Slangc::default()).expect("the vendored slangc");
    let forge = forge::Forge::new(compiler, forge::cache::ShaderCache::disabled());
    let source = asset("assets/shaders/StoreIndex.slang");
    let output = std::env::temp_dir().join("stormbreaker-compute.spv");
    forge
        .compile(
            &forge::CompileRequest::new(&source, "cs_main", forge::Stage::Compute),
            &output,
        )
        .expect("StoreIndex.slang compiles")
        .to_bytes()
}

fn open_device() -> Option<stormbreaker::Context> {
    if !device_available() {
        eprintln!("SKIP: no Vulkan loader");
        return None;
    }
    match create_device(DeviceOptions::from_env(c"stormbreaker-compute")) {
        Ok(context) => Some(context),
        Err(stormbreaker::BackendError::Loader(why)) => {
            eprintln!("SKIP: loader unavailable: {why}");
            None
        }
        Err(other) => panic!("device creation failed: {other}"),
    }
}

#[test]
fn a_dispatch_through_the_command_list_writes_the_buffer() {
    let Some(context) = open_device() else {
        return;
    };
    let mut backend = stormbreaker::vulkan_resources::VulkanResources::new(
        &context.device,
        context.memory_properties(),
    );

    let shader = backend
        .create_shader("StoreIndex", &compile_compute())
        .expect("the compute module");
    let set = chassis::DescriptorSetLayoutDesc::new("output").with(chassis::BindingDesc::new(
        0,
        chassis::DescriptorKind::StorageBufferReadWrite,
        chassis::StageMask::of(chassis::ShaderStage::Compute),
    ));
    let pipeline = backend
        .create_compute_pipeline(
            chassis::ComputePipelineDesc::new("store-index", shader).with_set(set.clone()),
        )
        .expect("a compute pipeline");
    let output = backend
        .create_buffer(
            chassis::BufferDesc::new(
                4,
                chassis::BufferUsage::STORAGE | chassis::BufferUsage::TRANSFER_DST,
            ),
            Some(&[0u8; 4]),
        )
        .expect("a storage buffer");
    let group = backend
        .create_bind_group(&set, &[(0, output, 0)], &[], &[])
        .expect("a bind group for the storage buffer");

    let family = context.graphics_queue_family().expect("a graphics queue");
    let mut recorder = Recorder::new(
        &context.device,
        context.graphics_queue,
        family,
        context.tier(),
        1,
    )
    .expect("a recorder");
    let mut recording = recorder.take(1).expect("one list").remove(0);

    {
        let pipelines = backend.pipelines();
        let textures = backend.textures();
        let buffers = backend.buffers();
        let descriptors = backend.descriptors();
        let mut list = VulkanCommandList::new(
            &mut recording,
            &pipelines,
            &textures,
            &buffers,
            &descriptors,
        );
        assert!(matches!(
            list.set_push_constants(&[0; 4]),
            Err(chassis::Error::Unsupported(_))
        ));
        list.begin_compute();
        list.bind_pipeline(&pipeline);
        list.bind_group(0, &group);
        list.dispatch(1, 1, 1);
        list.end_compute();
    }

    let finished = recording.end().expect("the recording ends");
    let token = recorder
        .submit(vec![finished], 0)
        .expect("the list is submitted");
    recorder.wait(token).expect("the submit retires");

    let bytes = backend
        .read_buffer(&output)
        .expect("the buffer is readable");
    assert_eq!(
        u32::from_ne_bytes(bytes[..4].try_into().expect("four bytes")),
        WRITTEN
    );
}
