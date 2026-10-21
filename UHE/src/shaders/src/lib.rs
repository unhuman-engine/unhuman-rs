//! Shader compilation. Slang in, SPIR-V out.
//!
//! The point of this crate is that a shader and the pipeline that uses it are
//! compiled by *one* compiler. The tree vendors a built Slang distribution and
//! `forge` drives that exact binary, so a bug reproduced with `slangc` on the
//! command line is the same bug the engine hit.
//!
//! [`ShaderCompiler`] is a trait for that reason, not for pluggability's sake:
//! the first implementation shells out to the driver (see [`slangc`]), and an
//! in-process `libslang.so` binding replaces it without touching a caller.
//! The driver exists first because the Slang C API is a vtable of virtual
//! interfaces, so linking it in process means binding that layout across a
//! crate boundary for no benefit until compile time actually matters.

pub mod cache;
pub mod request;
pub mod slangc;
pub mod stage;
pub mod wgsl;

use std::path::Path;

use crate::cache::{CacheKey, ShaderCache};
use crate::request::CompiledModule;
use crate::slangc::SlangcCompiler;

pub use cache::ShaderCache as Cache;
pub use request::{CompileRequest, CompiledModule as Module};
pub use request::{Define, Diagnostic, MatrixLayout, Optimization, Severity};
pub use slangc::{compile_once, Slangc};
pub use stage::Stage;

/// What can go wrong, with enough detail to act on.
#[derive(Debug)]
pub enum ForgeError {
    /// The compiler could not be started at all: not vendored, not on `PATH`,
    /// not executable.
    Spawn {
        program: std::path::PathBuf,
        error: std::io::Error,
    },
    /// The compiler ran and rejected the shader.
    ///
    /// Carries the diagnostics rather than a string, because the whole value of
    /// a shader compiler is a message that points at a line.
    Compile {
        /// Boxed: a `CompileRequest` holds `Vec<PathBuf>` and `Vec<Define>`, and
        /// an `Err` this large is copied on every `?`.
        request: Box<CompileRequest>,
        diagnostics: Vec<Diagnostic>,
    },
    /// Output that is not SPIR-V: truncated, misaligned, or wrong magic.
    ///
    /// Reached only on a corrupt cache entry or a compiler bug, and it has to be
    /// its own error because the alternative is a `vkCreateShaderModule` error
    /// that blames the driver for a file that was never valid.
    NotSpirv {
        path: std::path::PathBuf,
    },
    Io(std::io::Error),
}

impl std::fmt::Display for ForgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForgeError::Spawn { program, error } => {
                write!(f, "cannot run the Slang compiler at {program:?}: {error}")
            }
            ForgeError::Compile {
                request,
                diagnostics,
            } => {
                writeln!(
                    f,
                    "Slang rejected {} ({}, entry {}):",
                    request.source.display(),
                    request.stage,
                    request.entry
                )?;
                for diagnostic in diagnostics {
                    match &diagnostic.file {
                        Some(file) => writeln!(
                            f,
                            "  {}:{}:{}: {}",
                            file.display(),
                            diagnostic.line.unwrap_or(0),
                            diagnostic.column.unwrap_or(0),
                            diagnostic.message
                        )?,
                        None => writeln!(f, "  {}", diagnostic.message)?,
                    }
                }
                Ok(())
            }
            ForgeError::NotSpirv { path } => {
                write!(f, "{} is not a SPIR-V module", path.display())
            }
            ForgeError::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ForgeError {}

impl From<std::io::Error> for ForgeError {
    fn from(error: std::io::Error) -> Self {
        ForgeError::Io(error)
    }
}

/// Compiles shaders. One trait so the driver and an in-process FFI are
/// interchangeable.
pub trait ShaderCompiler: Send + Sync {
    /// A version string that participates in the cache key.
    fn version(&self) -> &str;

    /// Compile one entry point to SPIR-V, using the cache.
    ///
    /// `output` is where the module is written. It is required even on a cache
    /// hit, so the caller can hand the same path to `vkCreateShaderModule`
    /// either way and never has to ask which happened.
    fn compile(
        &self,
        request: &CompileRequest,
        output: &Path,
    ) -> Result<CompiledModule, ForgeError>;
}

/// The cache plus a driver.
pub struct Forge {
    compiler: SlangcCompiler,
    cache: ShaderCache,
}

impl Forge {
    pub fn new(compiler: SlangcCompiler, cache: ShaderCache) -> Self {
        Self { compiler, cache }
    }

    /// The vendored compiler, with a cache under `target/shader-cache`.
    pub fn discover() -> Result<Self, ForgeError> {
        Ok(Self::new(
            SlangcCompiler::new(Slangc::default())?,
            ShaderCache::new("target/shader-cache"),
        ))
    }

    pub fn version(&self) -> &str {
        self.compiler.version()
    }

    pub fn cache(&self) -> &ShaderCache {
        &self.cache
    }

    /// Compile, or load from cache.
    ///
    /// The cache is consulted before the compiler is run, and the key includes
    /// the source bytes, every flag and the compiler version, so a hit is
    /// guaranteed to be the module the source would have produced.
    pub fn compile(
        &self,
        request: &CompileRequest,
        output: &Path,
    ) -> Result<CompiledModule, ForgeError> {
        let Some(key) = CacheKey::new(request, self.compiler.version()) else {
            // No readable source, so no cache hit is possible. Fall through and
            // let the compiler produce the real diagnostic.
            return compile_once(&self.compiler, request, output);
        };

        if let Some(mut module) = self.cache.load(&key) {
            module.stage = request.stage;
            module.entry = request.entry.clone();
            // Still write the file: the caller asked for it, and a caller that
            // trusts the cache to have placed it is the kind of caller that
            // works until the cache is cleared.
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(output, module.to_bytes())?;
            return Ok(module);
        }

        let module = compile_once(&self.compiler, request, output)?;
        self.cache.store(&key, &module)?;
        Ok(module)
    }

    /// Compile a vertex/fragment pair, which is the unit Vulkan actually
    /// consumes.
    ///
    /// Vulkan has no linked shader stage: a pair is two modules with two entry
    /// points compiled from one source. A convenience that hides that is how you
    /// get a pipeline with a fragment stage that is actually the vertex stage.
    pub fn compile_pair(
        &self,
        source: &Path,
        vertex_entry: &str,
        fragment_entry: &str,
        output_dir: &Path,
    ) -> Result<(CompiledModule, CompiledModule), ForgeError> {
        let vertex = self.compile(
            &CompileRequest::new(source, vertex_entry, Stage::Vertex),
            &output_dir.join("shader.vert.spv"),
        )?;
        let fragment = self.compile(
            &CompileRequest::new(source, fragment_entry, Stage::Fragment),
            &output_dir.join("shader.frag.spv"),
        )?;
        Ok((vertex, fragment))
    }
}
