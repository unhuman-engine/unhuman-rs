//! The console panel's model: a bounded buffer of editor messages.
//!
//! The process log goes to stderr through `adam::logging`; that sink belongs to
//! the logger and cannot be read back. This is the editor's own log, written by
//! the actions that change the scene, so the console shows what the user did
//! rather than everything the engine ever printed.

use std::collections::VecDeque;

/// How many lines are kept. A bound because an editor left open for a day would
/// otherwise keep every message it ever wrote.
pub const DEFAULT_CAPACITY: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LogEntry {
    pub level: Level,
    pub message: String,
}

pub struct Console {
    lines: VecDeque<LogEntry>,
    capacity: usize,
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
            lines: VecDeque::new(),
            capacity,
        }
    }

    pub fn push(&mut self, level: Level, message: impl Into<String>) {
        if self.capacity == 0 {
            return;
        }
        self.lines.push_back(LogEntry {
            level,
            message: message.into(),
        });
        while self.lines.len() > self.capacity {
            self.lines.pop_front();
        }
    }

    pub fn info(&mut self, message: impl Into<String>) {
        self.push(Level::Info, message);
    }

    pub fn warn(&mut self, message: impl Into<String>) {
        self.push(Level::Warn, message);
    }

    pub fn error(&mut self, message: impl Into<String>) {
        self.push(Level::Error, message);
    }

    pub fn lines(&self) -> impl Iterator<Item = &LogEntry> {
        self.lines.iter()
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// The last message, which the status bar can show without walking the list.
    pub fn last(&self) -> Option<&LogEntry> {
        self.lines.back()
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_keeps_level_and_order() {
        let mut console = Console::new();
        console.info("first");
        console.warn("second");
        console.error("third");

        let lines: Vec<_> = console.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].level, Level::Info);
        assert_eq!(lines[1].level, Level::Warn);
        assert_eq!(lines[2].level, Level::Error);
        assert_eq!(console.last().unwrap().message, "third");
    }

    #[test]
    fn the_oldest_lines_fall_off_at_capacity() {
        let mut console = Console::with_capacity(3);
        for index in 0..5 {
            console.info(format!("line {index}"));
        }
        let messages: Vec<&str> = console.lines().map(|line| line.message.as_str()).collect();
        assert_eq!(messages, vec!["line 2", "line 3", "line 4"]);
    }

    #[test]
    fn zero_capacity_keeps_nothing() {
        let mut console = Console::with_capacity(0);
        console.error("gone");
        assert!(console.is_empty());
        assert_eq!(console.len(), 0);
    }

    #[test]
    fn clear_empties_the_buffer() {
        let mut console = Console::new();
        console.info("a");
        console.clear();
        assert!(console.is_empty());
        assert!(console.last().is_none());
    }
}
