//! Command line for the sandbox.
//!
//! Pure: `parse` reads a slice of arguments and returns a [`Config`], so every
//! flag combination is testable without a process, a window or a GPU.

use std::fmt;
use std::path::PathBuf;

/// Frames a headless run simulates when nothing else is said.
pub const DEFAULT_STEPS: u64 = 10_000;

/// `--steps 0`: run until a plugin asks to quit.
pub const STEPS_UNTIL_QUIT: u64 = 0;

/// What the process was asked to do.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Config {
    pub headless: bool,
    pub steps: u64,
    pub scene: Option<PathBuf>,
    /// The raw arguments, so the engine factory can resolve `--backend` itself
    /// rather than this parser keeping a second copy of that rule.
    pub raw: Vec<String>,
    pub help: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            headless: false,
            steps: DEFAULT_STEPS,
            scene: None,
            raw: Vec::new(),
            help: false,
        }
    }
}

impl Config {
    pub fn parse(arguments: &[String]) -> Result<Self, ParseError> {
        let mut config = Self {
            raw: arguments.to_vec(),
            ..Self::default()
        };
        let mut rest = arguments.iter();

        while let Some(argument) = rest.next() {
            let (name, inline) = split_flag(argument);
            match name {
                "--headless" => config.headless = true,
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
                // `--backend` is validated by `BackendSelection`; the flag itself
                // is accepted here so the raw list reaches the factory intact.
                "--backend" => {
                    value_of(name, inline, &mut rest)?;
                }
                "--help" | "-h" => config.help = true,
                other => return Err(ParseError(format!("unknown argument {other:?}"))),
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
sandbox - the real engine loop, windowed or headless

USAGE:
    sandbox [OPTIONS]

OPTIONS:
    --headless       no window; the frame loop still runs
    --steps N        headless frames to simulate (0 = until a plugin quits)
                     [default: 10000]
    --scene PATH     load a .unhuman scene before the plugins build
    --backend NAME   wgpu (default) or vulkan
    -h, --help       print this help

The backend is also readable from UHE_BACKEND and from the `backend` key in the
settings file; the flag wins.";
