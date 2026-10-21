//! A compute dispatch, on the real driver.
//!
//! The shader writes one word and the test reads it back. A dispatch that
//! returns without error is not evidence: it can be recorded into a pass that
//! never closes, bound to a pipeline with no work, or silently skipped. The
//! value is the assertion.

use chassis::{CommandList, Resources};
use stormglass::wgpu_command_buffer::{Recording, WgpuCommandList};
use stormglass::wgpu_device::Adapter;
use stormglass::wgpu_resources::WgpuResources;

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
    let output = std::env::temp_dir().join("stormglass-compute.spv");
    forge
        .compile(
            &forge::CompileRequest::new(&source, "cs_main", forge::Stage::Compute),
            &output,
        )
        .expect("StoreIndex.slang compiles")
        .to_bytes()
}

#[test]
fn a_dispatch_through_the_command_list_writes_the_buffer() {
    let Some(adapter) = Adapter::request(0) else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let Some((device, queue)) = adapter.create_device("stormglass-compute") else {
        eprintln!("SKIP: no device");
        return;
    };
    let mut resources = WgpuResources::new(&device);

    let shader = resources
        .create_shader("StoreIndex", &compile_compute())
        .expect("the compute module");
    let set = chassis::DescriptorSetLayoutDesc::new("output").with(chassis::BindingDesc::new(
        0,
        chassis::DescriptorKind::StorageBufferReadWrite,
        chassis::StageMask::of(chassis::ShaderStage::Compute),
    ));
    let pipeline = resources
        .create_compute_pipeline(
            chassis::ComputePipelineDesc::new("store-index", shader).with_set(set.clone()),
        )
        .expect("a compute pipeline");

    // `MAP_READ` for the test's readback; `STORAGE` so the shader can write.
    // wgpu allows the pair only with `MAPPABLE_PRIMARY_BUFFERS`, which
    // `create_device` requests when the adapter has it.
    let output = resources
        .create_buffer(
            chassis::BufferDesc::new(
                4,
                chassis::BufferUsage::STORAGE
                    | chassis::BufferUsage::TRANSFER_DST
                    | chassis::BufferUsage::MAP_READ,
            ),
            None,
        )
        .expect("a storage buffer");
    let group = resources
        .create_bind_group(&set, &[(0, output, 0)], &[], &[])
        .expect("a bind group for the storage buffer");

    let mut recording = Recording::new(device.inner(), "compute");
    {
        let mut list = WgpuCommandList::new(&mut recording, &resources);
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
    assert!(
        !recording.is_inside_compute_pass(),
        "the compute pass was closed before the buffer was finished",
    );

    let commands = recording.finish();
    queue.inner().submit([commands]);
    device
        .inner()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the queue drains");

    let bytes = resources
        .read_buffer(&output)
        .expect("the readback buffer maps after the queue drains");
    assert_eq!(
        u32::from_ne_bytes(bytes[..4].try_into().expect("four bytes")),
        WRITTEN
    );
}
