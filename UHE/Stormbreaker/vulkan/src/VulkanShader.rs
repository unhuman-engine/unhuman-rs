//! Shader modules.
//!
//! Modules are keyed by a content hash, so a shader whose source did not change
//! is never recompiled or re-uploaded. `forge` produces the SPIR-V and the hash;
//! this file never parses shader source.
//!
//! The driver's own cache is `pipeline_cache.rs`, and the "have we compiled this
//! description before" cache is in `pipeline.rs`. Three layers, three files.

use std::collections::HashMap;
use std::path::Path;

use ash::vk;
use chassis::{Registry, Shader};
use forge::{CompileRequest, Forge, Stage as SlangStage};

use crate::vulkan_error::{BackendError, Result};

/// A `forge` stage as the driver wants it.
///
/// Only pipeline stages map: `forge::Stage` covers ray tracing and mesh stages,
/// which have no `VkShaderStageFlag` here. Returning `None` rather than zero is
/// deliberate, because `vk::ShaderStageFlags::empty()` is a silently valid
/// argument to `vkCreateGraphicsPipelines` that produces a pipeline which draws
/// nothing.
pub fn shader_stage_flags(stage: SlangStage) -> Option<vk::ShaderStageFlags> {
    let flags = match stage {
        SlangStage::Vertex => vk::ShaderStageFlags::VERTEX,
        SlangStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        SlangStage::Compute => vk::ShaderStageFlags::COMPUTE,
        SlangStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        SlangStage::Hull => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        SlangStage::Domain => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        // No task, mesh, amplification, ray tracing or intersection.
        //
        // Not an omission: ash 0.38 does not expose those in core
        // `VkShaderStageFlagBits`, because mesh shading and ray tracing are
        // extensions. Mapping them here would need an extension-gated flag and a
        // feature check, and reporting them as bindable without that produces a
        // pipeline that creates fine and draws nothing. `forge::Stage` can parse
        // them for a compiler that accepts them; the driver cannot take them yet.
        SlangStage::Task
        | SlangStage::Mesh
        | SlangStage::Amplification
        | SlangStage::Raygen
        | SlangStage::Miss
        | SlangStage::AnyHit
        | SlangStage::ClosestHit
        | SlangStage::Callable
        | SlangStage::Intersection => return None,
    };
    Some(flags)
}

/// Reinterprets SPIR-V bytes as the `u32` words Vulkan wants, rejecting anything
/// that is not a module.
///
/// Every SPIR-V module starts with the magic word `0x07230203`. A buffer without
/// it is not SPIR-V, and letting that reach the driver produces a much worse
/// error message than refusing it here.
fn spirv_to_words(spirv: &[u8]) -> Result<&[u32]> {
    if spirv.len() < 4 || !spirv.len().is_multiple_of(4) {
        return Err(BackendError::MalformedSpirv { len: spirv.len() });
    }
    if u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]) != 0x0723_0203 {
        return Err(BackendError::MalformedSpirv { len: spirv.len() });
    }
    // A byte buffer that is 4-byte aligned and a multiple of 4 long can be read
    // as u32 on every target Vulkan runs on.
    let (prefix, words, suffix) = unsafe { spirv.align_to::<u32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(BackendError::MalformedSpirv { len: spirv.len() });
    }
    Ok(words)
}

/// Every SPIR-V module starts with this. Checked on every load, because a
/// buffer without it is not a module and letting it reach the driver produces a
/// worse message than refusing it here.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Cheap content hash (FNV-1a). Not cryptographic: it only has to notice that
/// the words changed.
pub fn content_hash_words(words: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Hash a byte buffer as if it were the words it decodes to, so a handle looked
/// up by bytes matches one looked up by words.
pub fn content_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub struct ShaderResource {
    module: vk::ShaderModule,
    hash: u64,
    stage: vk::ShaderStageFlags,
    entries: Vec<chassis::EntryPoint>,
}

impl ShaderResource {
    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn stage(&self) -> vk::ShaderStageFlags {
        self.stage
    }

    /// The module's name for `stage`, which is what a pipeline must ask for.
    pub fn entry(&self, stage: chassis::ShaderStage) -> Option<&str> {
        self.entries
            .iter()
            .find(|entry| entry.stage == stage)
            .map(|entry| entry.name.as_str())
    }
}

/// The Vulkan flag for a stage the RHI can name.
pub fn stage_flags(stage: chassis::ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        chassis::ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        chassis::ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        chassis::ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        chassis::ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        chassis::ShaderStage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        chassis::ShaderStage::TessellationEvaluation => {
            vk::ShaderStageFlags::TESSELLATION_EVALUATION
        }
    }
}

/// Owns shader modules, deduplicated by content hash.
pub struct Shaders {
    /// `None` when there is no device, in which case `create` fails and
    /// `Drop` has nothing to destroy. Better than a null handle that is
    /// dereferenced later.
    device: Option<ash::Device>,
    registry: Registry<ShaderResource, chassis::ShaderTag>,
    by_hash: HashMap<u64, Shader>,
    compiled: u32,
    reused: u32,
}

impl Shaders {
    /// A registry with no device behind it: fine for dedup tests, which never
    /// reach the driver, but nothing created through it can be destroyed.
    pub fn new() -> Self {
        Self::with_device(None)
    }

    pub fn with_device(device: Option<&ash::Device>) -> Self {
        Self {
            device: device.cloned(),
            registry: Registry::new(),
            by_hash: HashMap::new(),
            compiled: 0,
            reused: 0,
        }
    }

    /// The cached module for this SPIR-V, if one was loaded.
    ///
    /// Split out from `create` so the dedup rule is testable without a device,
    /// which is the part most likely to be wrong.
    pub fn lookup(&self, spirv: &[u8]) -> Option<Shader> {
        self.by_hash.get(&content_hash(spirv)).copied()
    }

    /// Creates a module from words `forge` already validated.
    ///
    /// The preferred entry point, because `forge::Module` has already checked
    /// the magic number and produced words, so this skips the byte
    /// reinterpretation `create` has to do. A shader that arrived from Slang
    /// should never be re-validated or re-aligned to reach the driver.
    pub fn create_from_words(&mut self, device: &ash::Device, words: &[u32]) -> Result<Shader> {
        if words.first().copied() != Some(SPIRV_MAGIC) {
            return Err(BackendError::MalformedSpirv {
                len: words.len() * 4,
            });
        }
        self.upload(device, words)
    }

    /// Compile a Slang source through `forge` and create the module.
    ///
    /// The seam between shader source and the GPU. `forge` owns the cache and the
    /// diagnostics; this owns the driver object and the dedup, so neither has to
    /// know about the other.
    pub fn load_slang(
        &mut self,
        device: &ash::Device,
        forge: &Forge,
        request: CompileRequest,
        scratch: &Path,
    ) -> Result<Shader> {
        let module = forge
            .compile(&request, scratch)
            .map_err(|error| BackendError::ShaderCompilation(error.to_string()))?;
        self.create_from_words(device, &module.words)
    }

    /// Creates a module, or returns the existing one when the same SPIR-V was
    /// already loaded this run.
    pub fn create(&mut self, device: &ash::Device, spirv: &[u8]) -> Result<Shader> {
        let words = spirv_to_words(spirv)?;
        self.upload(device, words)
    }

    /// The shared dedup-then-upload path behind every `create*`.
    fn upload(&mut self, device: &ash::Device, words: &[u32]) -> Result<Shader> {
        // Hash the words, not the bytes: both entry points see the same module,
        // so hashing the input type would mean hashing a different buffer for
        // the same content.
        let hash = content_hash_words(words);
        if let Some(existing) = self.by_hash.get(&hash) {
            self.reused += 1;
            return Ok(*existing);
        }

        let entries = chassis::spirv::entry_points(words)
            .filter(|entries| !entries.is_empty())
            .ok_or(BackendError::MalformedSpirv {
                len: words.len() * 4,
            })?;
        let stage = entries
            .iter()
            .fold(vk::ShaderStageFlags::empty(), |flags, entry| {
                flags | stage_flags(entry.stage)
            });

        let create_info = vk::ShaderModuleCreateInfo::default().code(words);
        let module = unsafe { device.create_shader_module(&create_info, None) }
            .map_err(|e| BackendError::vk("vkCreateShaderModule", e))?;

        let handle = self.registry.create(ShaderResource {
            module,
            hash,
            stage,
            entries,
        });
        self.by_hash.insert(hash, handle);
        self.compiled += 1;
        Ok(handle)
    }

    pub fn get(&self, handle: Shader) -> Option<&ShaderResource> {
        self.registry.get(handle)
    }

    pub fn destroy(&mut self, device: &ash::Device, handle: Shader) -> bool {
        let Some(resource) = self.registry.get(handle) else {
            return false;
        };
        self.by_hash.remove(&resource.hash);
        if let Some(resource) = self.registry.remove(handle) {
            unsafe { device.destroy_shader_module(resource.module, None) };
            true
        } else {
            false
        }
    }

    /// Modules actually created this run, versus how many requests were served
    /// from the content-hash cache. A `reused` that never grows means the cache
    /// is not being consulted.
    pub fn stats(&self) -> (u32, u32) {
        (self.compiled, self.reused)
    }

    pub fn len(&self) -> usize {
        self.registry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }
}

impl Drop for Shaders {
    fn drop(&mut self) {
        let Some(device) = &self.device else {
            return;
        };
        unsafe {
            for (_, resource) in self.registry.iter() {
                device.destroy_shader_module(resource.module, None);
            }
        }
        // A leaked `VkShaderModule` is only reported at `vkDestroyDevice`, as
        // VUID-vkDestroyDevice-device-05137. That error names teardown and
        // nothing about which object was left alive, so it costs a bisect every
        // time it appears.
    }
}

impl Default for Shaders {
    fn default() -> Self {
        Self::new()
    }
}
