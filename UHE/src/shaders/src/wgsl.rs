//! WGSL in, SPIR-V out.
//!
//! The second half of `forge`, and the half the wgpu backend needs: the
//! renderer's own passes are written in WGSL, and both backends take SPIR-V
//! (`chassis::Resources::create_shader` is bytes for a shader module either
//! way), so one compile path serves both. Slang stays the path for the vendored
//! shaders; this exists because a Slang compiler is a binary this machine does
//! not have, and a renderer that cannot compile its own shaders is a renderer
//! that cannot be checked.
//!
//! Validation is not optional here. naga's validator is the same set of rules
//! wgpu enforces at shader-module creation, so a shader that passes here fails
//! on neither backend for a *shader* reason, and an error names the line rather
//! than surfacing later as an opaque pipeline failure.

use crate::Stage;

/// What can go wrong, with the label of the shader that went wrong.
#[derive(Debug)]
pub enum WgslError {
    Parse { label: String, message: String },
    Validate { label: String, message: String },
    Backend { label: String, message: String },
    UnsupportedStage { label: String, stage: Stage },
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslError::Parse { label, message } => write!(f, "{label}: {message}"),
            WgslError::Validate { label, message } => {
                write!(f, "{label}: validation: {message}")
            }
            WgslError::Backend { label, message } => write!(f, "{label}: spir-v: {message}"),
            WgslError::UnsupportedStage { label, stage } => {
                write!(f, "{label}: {stage:?} has no SPIR-V stage")
            }
        }
    }
}

impl std::error::Error for WgslError {}

fn validate(source: &str, label: &str) -> Result<naga::valid::ModuleInfo, WgslError> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| WgslError::Parse {
        label: label.to_owned(),
        message: e.to_string(),
    })?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );

    validator
        .validate(&module)
        .map_err(|e| WgslError::Validate {
            label: label.to_owned(),
            message: e.to_string(),
        })
}

fn shader_stage(stage: Stage, label: &str) -> Result<naga::ShaderStage, WgslError> {
    match stage {
        Stage::Vertex => Ok(naga::ShaderStage::Vertex),
        Stage::Fragment => Ok(naga::ShaderStage::Fragment),
        Stage::Compute => Ok(naga::ShaderStage::Compute),
        other => Err(WgslError::UnsupportedStage {
            label: label.to_owned(),
            stage: other,
        }),
    }
}

fn words_to_bytes(words: Vec<u32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Every entry point in the source, as one SPIR-V module.
///
/// A single-entry shader is the case that matters and this is the wrong shape
/// for it; [`to_spirv_entry`] is the one to reach for. This exists for a compute
/// shader that carries several kernels.
pub fn to_spirv(source: &str, label: &str) -> Result<Vec<u8>, WgslError> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| WgslError::Parse {
        label: label.to_owned(),
        message: e.to_string(),
    })?;
    let info = validate(source, label)?;

    let options = naga::back::spv::Options::default();
    let words = naga::back::spv::write_vec(&module, &info, &options, None).map_err(|e| {
        WgslError::Backend {
            label: label.to_owned(),
            message: format!("{e:?}"),
        }
    })?;

    Ok(words_to_bytes(words))
}

/// One entry point, which is what a shader module is.
///
/// The pipeline stage comes from `forge::Stage`, not from naga, so a caller does
/// not have to depend on the SPIR-V backend to describe a shader.
///
/// The emitted module names its entry point [`crate::ENTRY_POINT`], not the name
/// the source declared. Slang renames it the same way -- `Triangle.slang`
/// declares `vs_main` and the module's `OpEntryPoint` is `main` -- and both
/// backends' pipeline creation names it when it builds a stage. A module that
/// kept the source name creates no pipeline on Vulkan and reports it as a shader
/// error.
pub fn to_spirv_entry(
    source: &str,
    label: &str,
    entry_point: &str,
    stage: Stage,
) -> Result<Vec<u8>, WgslError> {
    let mut module = naga::front::wgsl::parse_str(source).map_err(|e| WgslError::Parse {
        label: label.to_owned(),
        message: e.to_string(),
    })?;
    let info = validate(source, label)?;

    let entry_name = match module
        .entry_points
        .iter_mut()
        .find(|entry| entry.name == entry_point)
    {
        Some(entry) => {
            entry.name = crate::ENTRY_POINT.to_owned();
            crate::ENTRY_POINT
        }
        // Left alone, so naga reports the name the caller asked for.
        None => entry_point,
    };

    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: shader_stage(stage, label)?,
        entry_point: entry_name.to_owned(),
    };

    let options = naga::back::spv::Options::default();
    let words =
        naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).map_err(|e| {
            WgslError::Backend {
                label: label.to_owned(),
                message: format!("{e:?}"),
            }
        })?;

    Ok(words_to_bytes(words))
}

/// The entry points a source declares, in declaration order.
///
/// A caller that wants one module per stage needs their names; guessing them
/// from the file name is how a shader silently stops being compiled.
pub fn entry_points(source: &str, label: &str) -> Result<Vec<(String, Stage)>, WgslError> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| WgslError::Parse {
        label: label.to_owned(),
        message: e.to_string(),
    })?;

    Ok(module
        .entry_points
        .iter()
        .map(|entry| {
            let stage = match entry.stage {
                naga::ShaderStage::Vertex => Stage::Vertex,
                naga::ShaderStage::Fragment => Stage::Fragment,
                naga::ShaderStage::Compute => Stage::Compute,
                naga::ShaderStage::Task => Stage::Task,
                naga::ShaderStage::Mesh => Stage::Mesh,
                naga::ShaderStage::RayGeneration => Stage::Raygen,
                naga::ShaderStage::Miss => Stage::Miss,
                naga::ShaderStage::AnyHit => Stage::AnyHit,
                naga::ShaderStage::ClosestHit => Stage::ClosestHit,
            };
            (entry.name.clone(), stage)
        })
        .collect())
}

/// Parses and validates without emitting anything: the cheap half of a build
/// step, and the half a test can call.
pub fn check(source: &str, label: &str) -> Result<(), WgslError> {
    validate(source, label).map(|_| ())
}
