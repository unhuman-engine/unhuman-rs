//! One logging setup, so every binary prints the same shape and any caller can
//! attach a sink to read the same records back.
//!
//! One line per record:
//!
//! ```text
//! [2026-10-16 16:43:50.123] [info ] editor::app: editor renderer: AMD Radeon Graphics
//! ```
//!
//! Default level is `info`. Third-party crates that log at `info` are quieted
//! to `warn`, so a default run is the engine's output rather than the
//! accessibility bus and the Wayland span trace. `RUST_LOG` overrides it, read
//! exactly as `env_logger` reads it.
//!
//! Records go to stderr, to a file (so a process started without a terminal
//! still leaves a trace), and to every sink attached at runtime. Sinks are the
//! editor's way of showing engine records in its console: attach one and it
//! receives records on whichever thread emitted them.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use log::LevelFilter;

/// Crates whose `info` output is noise for an engine, all arriving with the
/// windowing stack. A caller who wants them sets `RUST_LOG`.
const QUIET: &[&str] = &[
    "zbus",
    "zvariant",
    "tracing",
    "accesskit",
    "calloop",
    "sctk",
    "smithay_client_toolkit",
];

/// Identifies an attached sink, so it can be detached without comparing trait
/// objects.
pub type SinkId = u64;

/// A runtime destination for the process log, besides stderr and the file.
///
/// The logger calls [`Sink::log`] on the thread that emitted the record, so an
/// implementation must be `Send + Sync`. It must not log, or it re-enters.
pub trait Sink: Send + Sync {
    /// Whether this sink wants the record. The logger asks before calling
    /// [`Sink::log`], so a filtered record is never formatted or copied for
    /// this sink. Other sinks and the file do not see the answer.
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>);
}

static SINKS: RwLock<Vec<(SinkId, Arc<dyn Sink>)>> = RwLock::new(Vec::new());
static NEXT_SINK: AtomicU64 = AtomicU64::new(0);
static LOGGER: OnceLock<Tee> = OnceLock::new();
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Attaches a sink and returns the id [`detach`] takes.
pub fn attach(sink: Arc<dyn Sink>) -> SinkId {
    let id = NEXT_SINK.fetch_add(1, Ordering::Relaxed);
    let mut sinks = SINKS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    sinks.push((id, sink));
    id
}

/// Detaches the sink with `id`. Returns whether one was attached.
pub fn detach(id: SinkId) -> bool {
    let mut sinks = SINKS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let before = sinks.len();
    sinks.retain(|(attached, _)| *attached != id);
    sinks.len() != before
}

/// The file this process is logging to, if [`init`] or [`init_with_file`] ran.
pub fn log_path() -> Option<PathBuf> {
    if !INSTALLED.load(Ordering::Acquire) {
        return None;
    }
    LOGGER.get().and_then(|tee| tee.path.clone())
}

/// Install the process logger with a file in the user's state directory.
///
/// Idempotent on purpose: a second call returns the path already in use instead
/// of installing a second logger. The path is returned so a caller can show it;
/// a process launched from a desktop entry has stderr nobody reads.
pub fn init() -> Option<PathBuf> {
    let (file, path) = open_default();
    install(file, path)
}

/// Install the process logger with its file at `path`, for tests or a caller
/// that owns the location. Returns the path actually used.
pub fn init_with_file(path: impl AsRef<Path>) -> Option<PathBuf> {
    let path = path.as_ref().to_path_buf();
    let file = open(&path);
    install(file, Some(path))
}

fn install(file: Option<File>, path: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(existing) = LOGGER.get() {
        return if INSTALLED.load(Ordering::Acquire) {
            existing.path.clone()
        } else {
            None
        };
    }

    let tee = Tee {
        stderr: builder().build(),
        file: file.map(file_logger),
        path,
    };
    // `env_logger`'s logger owns the filter; the max level is what lets records
    // reach it at all, and its own `enabled` does the per-module filtering.
    let max = tee.stderr.filter();
    if LOGGER.set(tee).is_err() {
        return LOGGER.get().and_then(|tee| tee.path.clone());
    }
    let tee = LOGGER.get().expect("just set");
    if log::set_logger(tee).is_err() {
        return None;
    }
    INSTALLED.store(true, Ordering::Release);
    log::set_max_level(max);
    tee.path.clone()
}

struct Tee {
    stderr: env_logger::Logger,
    file: Option<env_logger::Logger>,
    path: Option<PathBuf>,
}

impl log::Log for Tee {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        self.stderr.enabled(metadata)
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        self.stderr.log(record);
        if let Some(file) = &self.file {
            file.log(record);
        }
        dispatch(record);
    }

    fn flush(&self) {
        self.stderr.flush();
        if let Some(file) = &self.file {
            file.flush();
        }
    }
}

fn dispatch(record: &log::Record<'_>) {
    // Clone the sinks out so a slow or re-entrant sink is not called while the
    // registry is locked.
    let sinks: Vec<Arc<dyn Sink>> = {
        let attached = SINKS
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        attached.iter().map(|(_, sink)| Arc::clone(sink)).collect()
    };
    for sink in sinks {
        // The filter belongs to the sink, so a console can drop debug without
        // changing what the file or another sink receives.
        if sink.enabled(record.metadata()) {
            sink.log(record);
        }
    }
}

fn builder() -> env_logger::Builder {
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    for name in QUIET {
        builder.filter_module(name, LevelFilter::Warn);
    }
    builder.format(|buf, record| {
        // `env_logger`'s timestamp is RFC3339; the space and the dropped `Z`
        // are the shape this log uses.
        let stamp = buf
            .timestamp_millis()
            .to_string()
            .replace('T', " ")
            .trim_end_matches('Z')
            .to_string();
        let level = record.level();
        // Colour the level and reset after it; the style is empty when the
        // output is not a terminal, so a piped log carries no escape codes.
        let style = buf.default_level_style(level);
        writeln!(
            buf,
            "[{stamp}] [{style}{level:<5}{style:#}] {}: {}",
            record.target(),
            record.args()
        )
    });
    builder
}

fn file_logger(file: File) -> env_logger::Logger {
    let mut builder = builder();
    builder.target(env_logger::Target::Pipe(Box::new(file)));
    // A file is never a terminal; keep the escape codes out of it.
    builder.write_style(env_logger::WriteStyle::Never);
    builder.build()
}

fn open_default() -> (Option<File>, Option<PathBuf>) {
    let preferred = default_path();
    if let Some(file) = open(&preferred) {
        return (Some(file), Some(preferred));
    }
    let fallback = std::env::temp_dir().join("unhuman.log");
    if let Some(file) = open(&fallback) {
        return (Some(file), Some(fallback));
    }
    (None, None)
}

fn default_path() -> PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("unhuman").join("unhuman.log")
}

fn open(path: &Path) -> Option<File> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Append: two processes from the same desktop session should not truncate
    // each other's log.
    OpenOptions::new().create(true).append(true).open(path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_log_sits_under_a_state_or_temp_directory() {
        let path = default_path();
        assert!(path.ends_with("unhuman.log"));
        assert!(path.components().count() > 1);
    }

    #[test]
    fn a_sink_can_be_attached_and_detached_by_id() {
        let sink: Arc<dyn Sink> = Arc::new(Collector);
        let id = attach(sink);
        assert!(detach(id));
        assert!(!detach(id));
    }

    /// A sink that only counts, so the registry's lifetime is testable without
    /// installing a process logger.
    #[derive(Default)]
    struct Collector;

    impl Sink for Collector {
        fn log(&self, _record: &log::Record<'_>) {}
    }
}
