//! Command line for the headless binary.
//!
//! Pure: `parse` reads a slice of arguments and returns a [`Config`], so every
//! flag combination is testable without a process, an engine or a GPU.

use std::fmt;
use std::path::PathBuf;

/// Frames a run simulates when nothing else is said.
pub const DEFAULT_STEPS: u64 = 10_000;

/// `--steps 0`: run until a plugin asks to quit.
pub const STEPS_UNTIL_QUIT: u64 = 0;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Config {
    pub steps: u64,
    pub scene: Option<PathBuf>,
    pub record: Option<PathBuf>,
    pub replay: Option<PathBuf>,
    pub list_plugins: bool,
    pub help: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            steps: DEFAULT_STEPS,
            scene: None,
            record: None,
            replay: None,
            list_plugins: false,
            help: false,
        }
    }
}

impl Config {
    pub fn parse(arguments: &[String]) -> Result<Self, ParseError> {
        let mut config = Self::default();
        let mut rest = arguments.iter();

        while let Some(argument) = rest.next() {
            let (name, inline) = split_flag(argument);
            match name {
                "--steps" => {
                    let value = value_of(name, inline, &mut rest)?;
                    config.steps = value.parse().map_err(|_| {
                        ParseError(format!("--steps wants a frame count, got {value:?}"))
                    })?;
                }
                "--scene" => {
                    let value = value_of(name, inline, &mut rest)?;
                    config.scene = Some(PathBuf::from(value));
                }
                "--record" => {
                    let value = value_of(name, inline, &mut rest)?;
                    config.record = Some(PathBuf::from(value));
                }
                "--replay" => {
                    let value = value_of(name, inline, &mut rest)?;
                    config.replay = Some(PathBuf::from(value));
                }
                "--list-plugins" => config.list_plugins = true,
                "--help" | "-h" => config.help = true,
                other => {
                    return Err(ParseError(format!("unknown argument {other:?}")));
                }
            }
        }

        Ok(config)
    }

    /// Frames to simulate, or `None` for "until a plugin quits".
    pub fn frames(&self) -> Option<u64> {
        match self.steps {
            STEPS_UNTIL_QUIT => None,
            steps => Some(steps),
        }
    }
}

/// `--steps=100` splits into its name and its value; `--steps` has no value yet.
fn split_flag(argument: &str) -> (&str, Option<&str>) {
    match argument.split_once('=') {
        Some((name, value)) => (name, Some(value)),
        None => (argument, None),
    }
}

fn value_of<'a, I>(name: &str, inline: Option<&'a str>, rest: &mut I) -> Result<&'a str, ParseError>
where
    I: Iterator<Item = &'a String>,
{
    if let Some(value) = inline {
        return Ok(value);
    }
    match rest.next() {
        Some(value) => Ok(value.as_str()),
        None => Err(ParseError(format!("{name} wants a value"))),
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParseError(String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

// Used by the binary; the test target includes this file whole and only reaches
// the parser, so the help text reads as dead there.
#[allow(dead_code)]
pub const USAGE: &str = "\
nocap - headless frame loop, no window and no GPU

USAGE:
    nocap [OPTIONS]

OPTIONS:
    --steps N        frames to simulate (0 = until a plugin quits) [default: 10000]
    --scene PATH     load a .unhuman scene before the plugins build
    --record PATH    record N frames of scripted input to PATH as a capture
    --replay PATH    replay a capture from PATH and check every frame digest
    --list-plugins   print the plugin names in registration order and exit
    -h, --help       print this help

A capture is a deterministic frame sequence: the input delivered to each frame,
its fixed step, and a digest of the resulting scene. `--replay` runs the same
engine with no window and no GPU and fails on the first frame whose digest
differs.";
