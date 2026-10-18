//! Argument parsing, which is a pure function and needs no process.

#[path = "../src/config.rs"]
mod config;

use std::path::PathBuf;

use config::{Config, ParseError, DEFAULT_STEPS, STEPS_UNTIL_QUIT};

fn parse(arguments: &[&str]) -> Result<Config, ParseError> {
    let arguments: Vec<String> = arguments.iter().map(|a| a.to_string()).collect();
    Config::parse(&arguments)
}

#[test]
fn default_is_ten_thousand_steps_and_nothing_else() {
    let config = Config::default();
    assert_eq!(config.steps, DEFAULT_STEPS);
    assert_eq!(config.steps, 10_000);
    assert_eq!(config.scene, None);
    assert_eq!(config.record, None);
    assert_eq!(config.replay, None);
    assert!(!config.list_plugins);
    assert!(!config.help);
}

#[test]
fn no_arguments_is_the_default() {
    assert_eq!(parse(&[]).unwrap(), Config::default());
}

#[test]
fn steps_takes_a_value_either_way() {
    assert_eq!(parse(&["--steps", "250"]).unwrap().steps, 250);
    assert_eq!(parse(&["--steps=250"]).unwrap().steps, 250);
}

#[test]
fn zero_steps_means_until_a_plugin_quits() {
    let config = parse(&["--steps", "0"]).unwrap();
    assert_eq!(config.steps, STEPS_UNTIL_QUIT);
    assert_eq!(config.frames(), None);
    assert_eq!(Config::default().frames(), Some(10_000));
}

#[test]
fn scene_takes_a_path_either_way() {
    assert_eq!(
        parse(&["--scene", "assets/scene/aimlab.unhuman"])
            .unwrap()
            .scene,
        Some(PathBuf::from("assets/scene/aimlab.unhuman"))
    );
    assert_eq!(
        parse(&["--scene=assets/scene/aimlab.unhuman"])
            .unwrap()
            .scene,
        Some(PathBuf::from("assets/scene/aimlab.unhuman"))
    );
}

#[test]
fn list_plugins_and_help_are_flags() {
    assert!(parse(&["--list-plugins"]).unwrap().list_plugins);
    assert!(parse(&["--help"]).unwrap().help);
    assert!(parse(&["-h"]).unwrap().help);
}

#[test]
fn record_and_replay_take_paths_either_way() {
    assert_eq!(
        parse(&["--record", "run.cap"]).unwrap().record,
        Some(PathBuf::from("run.cap"))
    );
    assert_eq!(
        parse(&["--record=run.cap"]).unwrap().record,
        Some(PathBuf::from("run.cap"))
    );
    assert_eq!(
        parse(&["--replay=run.cap"]).unwrap().replay,
        Some(PathBuf::from("run.cap"))
    );
    assert!(parse(&["--record"]).is_err());
    assert!(parse(&["--replay"]).is_err());
}

#[test]
fn flags_combine_in_any_order() {
    let config = parse(&["--list-plugins", "--steps=3", "--scene=a.unhuman"]).unwrap();
    assert!(config.list_plugins);
    assert_eq!(config.steps, 3);
    assert_eq!(config.scene, Some(PathBuf::from("a.unhuman")));
}

#[test]
fn an_unknown_argument_is_an_error_that_names_it() {
    let error = parse(&["--frobnicate"]).unwrap_err();
    assert!(error.to_string().contains("--frobnicate"), "{error}");
}

#[test]
fn a_flag_without_its_value_is_an_error() {
    assert!(parse(&["--steps"]).is_err());
    assert!(parse(&["--scene"]).is_err());
}

#[test]
fn a_step_count_that_is_not_a_number_is_an_error() {
    let error = parse(&["--steps", "lots"]).unwrap_err();
    assert!(error.to_string().contains("lots"), "{error}");
    assert!(parse(&["--steps=-1"]).is_err());
}
