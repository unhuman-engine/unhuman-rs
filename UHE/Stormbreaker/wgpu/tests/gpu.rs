//! Backend selection, and `chassis::Gpu` as a trait object.
//!
//! `chassis::Gpu` had **zero** implementations until now. These tests are the
//! ones that make it a seam rather than a description.

use chassis::{
    backend::{flag_value, BackendChoice, BackendSelection, BackendSource},
    Gpu, GpuInfo, Swapchain,
};
use stormglass::wgpu_device::Adapter;
use stormglass::wgpu_gpu::WgpuGpu;

fn arguments(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_owned()).collect()
}

// -- selection

/// wgpu is the default.
///
/// A stability decision, not a performance one: when a frame goes wrong the
/// first question is "the backend or the engine", and the default should be the
/// one that usually works.
#[test]
fn wgpu_is_the_default_backend() {
    assert_eq!(BackendChoice::default(), BackendChoice::Wgpu);
    assert_eq!(BackendSelection::default().choice, BackendChoice::Wgpu);
    assert_eq!(
        BackendSelection::default().source,
        BackendSource::Default,
        "and it says it was a default, not a choice",
    );
}

/// The flag wins over everything.
///
/// The case that silently stops being true when someone adds a fourth source,
/// which is why it is a test rather than a comment.
#[test]
fn the_flag_beats_the_environment() {
    let selection =
        BackendSelection::resolve(&arguments(&["--backend", "vulkan"]), Some("wgpu"), None)
            .expect("both are valid");
    assert_eq!(selection.choice, BackendChoice::Vulkan);
    assert_eq!(selection.source, BackendSource::Flag);
}

/// Both flag spellings. A parser that takes one and not the other is a trap that
/// only shows up on the invocation nobody tested.
#[test]
fn both_flag_spellings_work() {
    for form in [
        arguments(&["--backend", "vulkan"]),
        arguments(&["--backend=vulkan"]),
    ] {
        let selection = BackendSelection::resolve(&form, None, None).expect("valid");
        assert_eq!(selection.choice, BackendChoice::Vulkan, "for {form:?}");
        assert_eq!(selection.source, BackendSource::Flag);
    }
}

#[test]
fn the_environment_is_used_when_there_is_no_flag() {
    let selection = BackendSelection::resolve(&[], Some("vulkan"), None).expect("valid");
    assert_eq!(selection.choice, BackendChoice::Vulkan);
    assert_eq!(selection.source, BackendSource::Environment);

    // An empty value is nobody saying anything, so the default stands rather than
    // an error -- an exported-but-blank variable is a normal accident.
    let blank = BackendSelection::resolve(&[], Some("  "), None).expect("blank is not a choice");
    assert_eq!(blank.choice, BackendChoice::Wgpu);
    assert_eq!(blank.source, BackendSource::Default);
}

/// An unrecognised backend is an error, not a silent fallback.
///
/// A launcher that passes `--backend vlkan` and silently gets wgpu has a bug that
/// will be reported as "the wrong backend", and nothing would point here.
#[test]
fn a_misspelled_backend_is_an_error_naming_what_was_asked_for() {
    let error = BackendSelection::resolve(&arguments(&["--backend", "vlkan"]), None, None)
        .expect_err("a typo is not a backend");
    assert!(
        error.to_string().contains("vlkan"),
        "the message quotes what was asked for, got {error}",
    );
    // And it says what would have worked.
    assert!(error.to_string().contains("wgpu"));
    assert!(error.to_string().contains("vulkan"));

    // An empty flag value is nobody saying anything, and it must not parse as a
    // backend: `--backend ""` is a launcher bug, not a request for the default.
    assert!("".parse::<BackendChoice>().is_err());
}

#[test]
fn names_are_matched_case_insensitively_and_trimmed() {
    for spelling in ["vulkan", "VULKAN", " Vulkan ", "\tvk\n"] {
        let selection = BackendSelection::resolve(&arguments(&["--backend", spelling]), None, None)
            .unwrap_or_else(|e| panic!("{spelling:?} should parse: {e}"));
        assert_eq!(selection.choice, BackendChoice::Vulkan, "for {spelling:?}");
    }
}

/// A flag with no value after it falls through rather than panicking.
///
/// `--backend` alone is a launcher bug, and the useful response is to use the
/// default and say so -- not to crash before the window opens.
#[test]
fn a_flag_with_no_value_falls_through_to_the_default() {
    let selection =
        BackendSelection::resolve(&arguments(&["--backend"]), None, None).expect("does not panic");
    assert_eq!(selection.choice, BackendChoice::Wgpu);
}

#[test]
fn unrelated_arguments_are_left_alone() {
    assert_eq!(flag_value(&arguments(&["--width", "1280"])), None);
    assert_eq!(
        flag_value(&arguments(&["--backend", "vulkan", "--width", "1"])),
        Some("vulkan".into())
    );
}

// -- the trait object

fn open_gpu() -> Option<WgpuGpu> {
    if Adapter::request(0).is_none() {
        eprintln!("SKIP: no wgpu adapter");
        return None;
    }
    WgpuGpu::headless(chassis::Extent2d {
        width: 64,
        height: 64,
    })
    .ok()
}

/// The first implementation of `chassis::Gpu` in this project, reached as
/// `dyn Gpu`.
///
/// Everything above the RHI will hold this, not `WgpuGpu`. That is the claim
/// the whole two-backend design rests on and it has never compiled until now.
#[test]
fn a_wgpu_backend_is_usable_as_dyn_gpu() {
    let Some(gpu) = open_gpu() else { return };
    let mut gpu: Box<dyn Gpu> = Box::new(gpu);

    let info: &GpuInfo = gpu.info();
    assert!(!info.name.is_empty(), "the device reported a name");
    assert!(info.max_texture_size >= 2048);
    assert_eq!(
        info.frames_in_flight, 1,
        "one, until more are actually wired"
    );

    gpu.frame_begin().expect("frame begin");
    gpu.submit().expect("submit with no frontend attached");
    gpu.wait_idle();

    // Resources are reachable through the trait, and objects made through it are
    // real. A `dyn Resources` that cannot create anything is not a seam.
    let resources: &mut dyn chassis::Resources = gpu.resources();
    let buffer = resources
        .create_buffer(
            chassis::BufferDesc::new(256, chassis::BufferUsage::VERTEX),
            Some(&[3u8; 256]),
        )
        .expect("a buffer through dyn Resources");
    let texture = resources
        .create_texture(chassis::TextureDesc::new_2d(
            chassis::Extent {
                width: 32,
                height: 32,
                depth: 1,
            },
            chassis::TextureFormat::Rgba8Unorm,
            chassis::TextureUsage::COLOR_ATTACHMENT,
        ))
        .expect("a texture through dyn Resources");
    resources.destroy_texture(texture);
    resources.destroy_buffer(buffer);

    let _: &mut dyn chassis::RenderGraph = gpu.graph();
}

/// A headless backend has no swapchain and says so.
///
/// `Gpu::swapchain` returns an `Option` for exactly this, and this is the first
/// implementation that takes the opportunity: a dummy would make every caller
/// carry one.
#[test]
fn a_headless_backend_has_no_swapchain() {
    let Some(gpu) = open_gpu() else { return };
    let gpu: &dyn Gpu = &gpu;
    assert!(
        gpu.swapchain().is_none(),
        "headless is a real mode, not a degraded one",
    );
}

/// The same object is both `Gpu` and `Swapchain`, and `Gpu::swapchain` wins.
///
/// Trivial, and worth pinning: the two impls coexisting is what makes the
/// `Option` above possible without a wrapper type.
#[test]
fn the_concrete_type_is_both_a_gpu_and_a_swapchain() {
    let Some(gpu) = open_gpu() else { return };
    let as_swapchain: &dyn Swapchain = &gpu;
    assert_eq!(as_swapchain.extent().width, 64);
    assert!(
        !as_swapchain.is_out_of_date(),
        "a headless frame is never out of date"
    );
}

/// A handle made through `dyn Resources` must resolve inside the executor.
///
/// The bug this pins: `WgpuGpu` used to build one `WgpuResources` for itself
/// and hand a *second* to the executor, so every texture created through the
/// trait had a handle the executor could not resolve. Nothing caught it, because
/// the tests that created resources never recorded a pass with them -- the draw
/// test built its own single registry by hand.
#[test]
fn a_resource_made_through_the_trait_resolves_in_the_executor() {
    let Some(mut gpu) = open_gpu() else { return };
    let extent = chassis::Extent2d {
        width: 32,
        height: 32,
    };

    let texture = {
        let resources: &mut dyn chassis::Resources = gpu.resources();
        resources
            .create_texture(chassis::TextureDesc::new_2d(
                chassis::Extent {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                chassis::TextureFormat::Rgba8Unorm,
                chassis::TextureUsage::COLOR_ATTACHMENT,
            ))
            .expect("a texture through dyn Resources")
    };

    // The same handle, asked of the executor. Different registries would give
    // `None` here, which is exactly the failure the two-registry bug produced.
    let resolved = gpu
        .executor()
        .resources()
        .get_texture(texture)
        .expect("the handle resolves in the executor");
    assert_eq!(resolved.extent().width, extent.width);
    assert_eq!(resolved.format(), chassis::TextureFormat::Rgba8Unorm);

    // And through the slot table the executor records from.
    gpu.slots_mut().set_texture(0, texture);
    assert_eq!(gpu.slots().texture(0), Some(texture));
    assert!(
        gpu.executor().resources().get_texture(texture).is_some(),
        "the slot the executor will look it up in holds the handle it can resolve",
    );
}

/// The engine's own `Gpu` can upload a texture.
///
/// The bug this pins: `WgpuGpu` assembled its executor registry with
/// `WgpuResources::new`, which carries no queue, while the gpu it was building
/// held one. `upload_texture` is `queue.write_texture`, so it refused -- and
/// `bifrost` uploads its 1x1 textures during plugin build, so the editor and the
/// sandbox failed to start on any machine *with* a working GPU. No test caught
/// it because every test that uploaded built its own `WgpuResources::with_queue`.
#[test]
fn the_engine_gpu_can_upload_a_texture() {
    let Some(mut gpu) = open_gpu() else { return };

    let resources: &mut dyn chassis::Resources = gpu.resources();
    let texture = resources
        .create_texture(chassis::TextureDesc::new_2d(
            chassis::Extent {
                width: 2,
                height: 2,
                depth: 1,
            },
            chassis::TextureFormat::Rgba8Unorm,
            // `TRANSFER_DST` too: `queue.write_texture` requires `COPY_DST`.
            chassis::TextureUsage::SAMPLED | chassis::TextureUsage::TRANSFER_DST,
        ))
        .expect("a sampled texture through dyn Resources");
    resources
        .upload_texture(&texture, &[255u8; 2 * 2 * 4], 2, 2)
        .expect("the engine's resources carry the queue the renderer uploads through");
}

/// The settings file is the third source, and it is last.
///
/// `BackendSource::Settings` existed from the start and nothing could ever
/// produce it -- `resolve` took only two arguments. So the variant was a claim
/// about a feature that did not exist, and this is the test that makes it true.
///
/// First-one-that-says-anything wins, checked pairwise rather than in a chain:
/// "flag beats environment" passing says nothing about "flag beats settings",
/// and the failing pair is the one nobody thinks to test.
#[test]
fn the_settings_file_is_the_last_source() {
    let from_file = || BackendSelection::resolve(&[], None, Some("wgpu")).expect("valid");

    // The flag beats both.
    let selection = BackendSelection::resolve(
        &arguments(&["--backend", "vulkan"]),
        Some("wgpu"),
        Some("wgpu"),
    )
    .expect("valid");
    assert_eq!(selection.choice, BackendChoice::Vulkan);
    assert_eq!(selection.source, BackendSource::Flag);

    // The environment beats the settings file.
    let selection = BackendSelection::resolve(&[], Some("vulkan"), Some("wgpu")).expect("valid");
    assert_eq!(selection.choice, BackendChoice::Vulkan);
    assert_eq!(selection.source, BackendSource::Environment);

    // And with only the file, the file decides -- and says so.
    let selection = from_file();
    assert_eq!(selection.choice, BackendChoice::Wgpu);
    assert_eq!(selection.source, BackendSource::Settings);
    assert!(
        selection.source.to_string().contains("settings"),
        "the log line names where the answer came from: {}",
        selection.source,
    );
}

/// A blank source is nobody saying anything, not somebody saying "blank".
///
/// All three sources can be blank -- an empty environment variable, an empty
/// `UHE_BACKEND`, a settings file with `backend =` and nothing after it -- and
/// all three have to fall through rather than fail. This is the case that makes
/// "first one that says anything" true instead of "first one that exists".
#[test]
fn a_blank_source_falls_through_rather_than_failing() {
    for blank in ["", "   ", "\t"] {
        let selection = BackendSelection::resolve(&[], Some(blank), Some(blank)).expect("blank");
        assert_eq!(selection.choice, BackendChoice::Wgpu);
        assert_eq!(
            selection.source,
            BackendSource::Default,
            "{blank:?} is nobody saying anything",
        );
    }

    // A blank environment with a real settings file still gets the settings.
    let selection = BackendSelection::resolve(&[], Some("  "), Some("vulkan")).expect("valid");
    assert_eq!(selection.source, BackendSource::Settings);
}

/// A settings file with a backend nobody has heard of is an error, like any
/// other source.
///
/// The alternative -- falling through to wgpu -- would make a typo in a file the
/// user deliberately edited mean "wgpu, mysteriously", and the error would name
/// the flag instead of the file.
#[test]
fn a_misspelled_backend_in_settings_is_an_error() {
    let error = BackendSelection::resolve(&[], None, Some("vlkan"))
        .expect_err("a typo in the settings file is still a typo");
    assert!(error.0.contains("vlkan"), "names what was asked: {error}");
}

/// The settings file really does feed the selection, end to end.
///
/// `resolve` takes a string rather than a file so it stays pure, which means
/// something has to do the reading. This checks that `Settings::parse` produces
/// something `resolve` accepts, because those are two functions that could
/// disagree about what a `backend` key means.
#[test]
fn a_parsed_settings_file_drives_the_choice() {
    let settings = chassis::settings::Settings::parse(
        "# unhuman.settings\nwindow.width = 1280\nbackend = vulkan\n",
    );
    let selection = BackendSelection::resolve(&[], None, settings.backend()).expect("valid");
    assert_eq!(selection.choice, BackendChoice::Vulkan);
    assert_eq!(selection.source, BackendSource::Settings);

    // And a file that does not mention a backend leaves the choice alone.
    let settings = chassis::settings::Settings::parse("window.width = 1280\n");
    assert_eq!(settings.backend(), None);
    let selection = BackendSelection::resolve(&[], None, settings.backend()).expect("valid");
    assert_eq!(selection.source, BackendSource::Default);
}
