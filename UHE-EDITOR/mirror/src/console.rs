//! The console panel's model: one bounded buffer for editor and engine records.
//!
//! `adam::logging` attaches this as a sink, so a `bifrost` warning arrives here
//! on whichever worker thread emitted it rather than only in the terminal the
//! editor was launched from. The buffer is shared, and the panel paints a
//! snapshot so the logging thread never waits on the widget tree.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

/// How many lines are kept. A bound because an editor left open for a day would
/// otherwise keep every message it ever wrote.
pub const DEFAULT_CAPACITY: usize = 512;

/// How many typed commands are kept for up/down.
pub const DEFAULT_HISTORY: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<log::Level> for Level {
    fn from(level: log::Level) -> Self {
        match level {
            log::Level::Trace => Level::Trace,
            log::Level::Debug => Level::Debug,
            log::Level::Info => Level::Info,
            log::Level::Warn => Level::Warn,
            log::Level::Error => Level::Error,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LogEntry {
    pub level: Level,
    pub target: String,
    pub message: String,
    /// How many identical consecutive records this line stands for. A warning
    /// repeated every frame is one line with a count, not sixty lines.
    pub count: u32,
}

/// A shareable, thread-safe log buffer.
///
/// Clones share the buffer, which is what lets the engine's logging thread and
/// the panel's paint see the same list.
#[derive(Clone)]
pub struct Console {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    lines: VecDeque<LogEntry>,
    capacity: usize,
    /// Records evicted because the buffer was full. Shown above the lines: a
    /// console that silently lost records lies about what happened.
    dropped: u64,
    /// Minimum level this console accepts. The logger asks before it formats
    /// or copies a record for this sink.
    level: Level,
    /// Target prefix, or empty for every target. `bifrost` admits
    /// `bifrost::passes` as well.
    target: String,
    history: Vec<String>,
    /// Index into `history` while walking it, `None` while editing.
    history_index: Option<usize>,
    command: String,
}

impl Inner {
    fn admits(&self, level: Level, target: &str) -> bool {
        level >= self.level && (self.target.is_empty() || target.starts_with(&self.target))
    }
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

impl Console {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// A capacity of zero keeps nothing, which is how a headless run turns the
    /// console off without a second code path.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                lines: VecDeque::new(),
                capacity,
                dropped: 0,
                level: Level::Info,
                target: String::new(),
                history: Vec::new(),
                history_index: None,
                command: String::new(),
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Editor messages carry the `editor` target; engine records carry theirs.
    pub fn push(&self, level: Level, message: impl Into<String>) {
        self.push_target(level, "editor", message);
    }

    pub fn push_target(&self, level: Level, target: impl Into<String>, message: impl Into<String>) {
        let target = target.into();
        let message = message.into();
        let mut inner = self.lock();
        if inner.capacity == 0 || !inner.admits(level, &target) {
            return;
        }

        if let Some(last) = inner.lines.back_mut() {
            if last.level == level && last.target == target && last.message == message {
                last.count = last.count.saturating_add(1);
                return;
            }
        }

        inner.lines.push_back(LogEntry {
            level,
            target,
            message,
            count: 1,
        });
        while inner.lines.len() > inner.capacity {
            inner.lines.pop_front();
            inner.dropped += 1;
        }
    }

    pub fn info(&self, message: impl Into<String>) {
        self.push(Level::Info, message);
    }

    pub fn warn(&self, message: impl Into<String>) {
        self.push(Level::Warn, message);
    }

    pub fn error(&self, message: impl Into<String>) {
        self.push(Level::Error, message);
    }

    /// A snapshot of the buffer, oldest first.
    ///
    /// A snapshot rather than an iterator because the engine may append while
    /// the panel paints; holding the lock across a whole widget tree would make
    /// the logger wait on the UI.
    pub fn lines(&self) -> Vec<LogEntry> {
        let inner = self.lock();
        let mut lines: Vec<LogEntry> = inner.lines.iter().cloned().collect();
        if inner.dropped > 0 {
            lines.insert(0, dropped_entry(inner.dropped));
        }
        lines
    }

    pub fn len(&self) -> usize {
        self.lock().lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().lines.is_empty()
    }

    /// The last retained message, without the dropped notice.
    pub fn last(&self) -> Option<LogEntry> {
        self.lock().lines.back().cloned()
    }

    /// Records lost because the buffer was full.
    pub fn dropped(&self) -> u64 {
        self.lock().dropped
    }

    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.lines.clear();
        inner.dropped = 0;
    }

    /// Drops records below `level` before they reach this console. Other sinks
    /// and the file keep their own filters.
    pub fn set_level(&self, level: Level) {
        self.lock().level = level;
    }

    pub fn level(&self) -> Level {
        self.lock().level
    }

    /// Admits only targets that start with `target`. Empty admits every target.
    pub fn set_target(&self, target: impl Into<String>) {
        self.lock().target = target.into();
    }

    pub fn target(&self) -> String {
        self.lock().target.clone()
    }

    // ---- the command line --------------------------------------------------

    pub fn command_line(&self) -> String {
        self.lock().command.clone()
    }

    /// Replaces the line being edited, for a caller that owns a buffer.
    pub fn set_command_line(&self, line: impl Into<String>) {
        let mut inner = self.lock();
        inner.command = line.into();
        inner.history_index = None;
    }

    pub fn command_insert(&self, text: &str) {
        self.lock().command.push_str(text);
    }

    pub fn command_backspace(&self) {
        let mut inner = self.lock();
        inner.command.pop();
        inner.history_index = None;
    }

    pub fn command_clear(&self) {
        let mut inner = self.lock();
        inner.command.clear();
        inner.history_index = None;
    }

    /// Moves one step towards older history and puts it on the line.
    pub fn history_up(&self) {
        let mut inner = self.lock();
        if inner.history.is_empty() {
            return;
        }
        let index = match inner.history_index {
            Some(index) => index.saturating_sub(1),
            None => inner.history.len() - 1,
        };
        inner.history_index = Some(index);
        inner.command = inner.history[index].clone();
    }

    /// Moves one step towards newer history; the newest step leaves an empty
    /// line, so typing after Enter starts cleanly.
    pub fn history_down(&self) {
        let mut inner = self.lock();
        let Some(index) = inner.history_index else {
            return;
        };
        if index + 1 < inner.history.len() {
            inner.history_index = Some(index + 1);
            inner.command = inner.history[index + 1].clone();
        } else {
            inner.history_index = None;
            inner.command.clear();
        }
    }

    /// Records a submitted line and clears the buffer. A repeat of the last
    /// command is one history entry, not a run of identical ones.
    pub fn remember_command(&self, line: &str) {
        let line = line.trim();
        let mut inner = self.lock();
        inner.command.clear();
        inner.history_index = None;
        if line.is_empty() || inner.history.last().is_some_and(|last| last == line) {
            return;
        }
        inner.history.push(line.to_string());
        if inner.history.len() > DEFAULT_HISTORY {
            inner.history.remove(0);
        }
    }
}

fn dropped_entry(dropped: u64) -> LogEntry {
    let noun = if dropped == 1 { "record" } else { "records" };
    LogEntry {
        level: Level::Warn,
        target: "console".to_string(),
        message: format!("{dropped} {noun} dropped"),
        count: 1,
    }
}

impl adam::logging::Sink for Console {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        let inner = self.lock();
        inner.admits(metadata.level().into(), metadata.target())
    }

    fn log(&self, record: &log::Record<'_>) {
        self.push_target(
            record.level().into(),
            record.target(),
            record.args().to_string(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam::logging::Sink as _;

    #[test]
    fn push_keeps_level_target_and_order() {
        let console = Console::new();
        console.info("first");
        console.warn("second");
        console.error("third");

        let lines = console.lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].level, Level::Info);
        assert_eq!(lines[1].level, Level::Warn);
        assert_eq!(lines[2].level, Level::Error);
        assert_eq!(lines[0].target, "editor");
        assert_eq!(console.last().unwrap().message, "third");
    }

    #[test]
    fn the_oldest_lines_fall_off_at_capacity() {
        let console = Console::with_capacity(3);
        for index in 0..5 {
            console.info(format!("line {index}"));
        }
        let messages: Vec<String> = console
            .lines()
            .into_iter()
            .filter(|line| line.target == "editor")
            .map(|line| line.message)
            .collect();
        assert_eq!(messages, vec!["line 2", "line 3", "line 4"]);
    }

    #[test]
    fn a_full_buffer_counts_and_shows_what_it_dropped() {
        let console = Console::with_capacity(2);
        console.info("a");
        console.info("b");
        console.info("c");

        assert_eq!(console.dropped(), 1);
        let lines = console.lines();
        assert_eq!(lines[0].target, "console");
        assert_eq!(lines[0].message, "1 record dropped");
    }

    #[test]
    fn identical_consecutive_records_are_counted_not_listed() {
        let console = Console::new();
        for _ in 0..60 {
            console.warn("repeat");
        }
        let lines = console.lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].count, 60);
    }

    #[test]
    fn a_different_record_breaks_the_run() {
        let console = Console::new();
        console.warn("same");
        console.info("other");
        console.warn("same");
        assert_eq!(console.len(), 3);
    }

    #[test]
    fn engine_records_keep_their_level_and_target() {
        let console = Console::new();
        let record = log::Record::builder()
            .args(format_args!("mesh failed"))
            .level(log::Level::Warn)
            .target("bifrost")
            .build();
        console.log(&record);

        let lines = console.lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].level, Level::Warn);
        assert_eq!(lines[0].target, "bifrost");
        assert_eq!(lines[0].message, "mesh failed");
    }

    #[test]
    fn records_from_worker_threads_share_one_buffer() {
        let console = Console::with_capacity(1024);
        let threads: Vec<_> = (0..8)
            .map(|worker| {
                let console = console.clone();
                std::thread::spawn(move || {
                    for index in 0..50 {
                        console.info(format!("worker {worker}: {index}"));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(console.len(), 400);
    }

    #[test]
    fn zero_capacity_keeps_nothing() {
        let console = Console::with_capacity(0);
        console.error("gone");
        assert!(console.is_empty());
        assert_eq!(console.len(), 0);
    }

    #[test]
    fn clear_empties_the_buffer() {
        let console = Console::new();
        console.info("a");
        console.clear();
        assert!(console.is_empty());
        assert!(console.last().is_none());
        assert_eq!(console.dropped(), 0);
    }

    #[test]
    fn a_level_filter_rejects_lower_records_before_they_are_stored() {
        let console = Console::new();
        console.set_level(Level::Info);
        let record = log::Record::builder()
            .args(format_args!("noise"))
            .level(log::Level::Debug)
            .target("bifrost")
            .build();
        assert!(!console.enabled(record.metadata()));

        console.push_target(Level::Debug, "bifrost", "noise");
        console.push_target(Level::Info, "bifrost", "kept");
        let messages: Vec<String> = console
            .lines()
            .into_iter()
            .map(|line| line.message)
            .collect();
        assert_eq!(messages, vec!["kept"]);
    }

    #[test]
    fn a_target_filter_matches_a_prefix() {
        let console = Console::new();
        console.set_target("bifrost");
        console.push_target(Level::Info, "bifrost", "exact");
        console.push_target(Level::Info, "bifrost::passes", "child");
        console.push_target(Level::Info, "editor", "other");

        let targets: Vec<String> = console
            .lines()
            .into_iter()
            .map(|line| line.target)
            .collect();
        assert_eq!(targets, vec!["bifrost", "bifrost::passes"]);
    }

    #[test]
    fn command_history_walks_older_and_newer_lines() {
        let console = Console::new();
        console.remember_command("undo");
        console.remember_command("redo");
        assert_eq!(console.command_line(), "");

        console.history_up();
        assert_eq!(console.command_line(), "redo");
        console.history_up();
        assert_eq!(console.command_line(), "undo");
        console.history_down();
        assert_eq!(console.command_line(), "redo");
        console.history_down();
        assert_eq!(console.command_line(), "");
    }

    #[test]
    fn a_repeated_command_is_one_history_entry() {
        let console = Console::new();
        console.remember_command("undo");
        console.remember_command("undo");
        console.history_up();
        assert_eq!(console.command_line(), "undo");
        console.history_up();
        assert_eq!(console.command_line(), "undo");
        console.history_down();
        assert_eq!(console.command_line(), "");
    }
}
