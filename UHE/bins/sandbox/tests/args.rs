//! Argument parsing, which is a pure function and needs no process.

#[path = "../src/args.rs"]
mod args;

use std::path::PathBuf;

use args::{Config, ParseError, DEFAULT_STEPS, STEPS_UNTIL_QUIT};

fn parse(arguments: &[&str]) -> Result<Config, ParseError> {
    let arguments: Vec<String> = arguments.iter().map(|a| a.to_string()).collect();
    Config::parse(&arguments)
}

#[test]
fn default_is_windowed_at_ten_thousand_steps() {
    let config = Config::default();
    assert!(!config.headless);
    assert_eq!(config.steps, DEFAULT_STEPS);
    assert_eq!(config.scene, None);
    assert!(!config.help);
}

#[test]
fn no_arguments_is_the_default_apart_from_the_raw_list() {
    assert_eq!(parse(&[]).unwrap(), Config::default());
}

#[test]
fn headless_is_a_flag() {
    assert!(parse(&["--headless"]).unwrap().headless);
}

#[test]
fn steps_takes_a_value_either_way() {
    assert_eq!(parse(&["--steps", "250"]).unwrap().steps, 250);
    assert_eq!(parse(&["--steps=250"]).unwrap().steps, 250);
}

#[test]
fn zero_steps_means_until_a_plugin_quits() {
    let config = parse(&["--headless", "--steps", "0"]).unwrap();
    assert_eq!(config.steps, STEPS_UNTIL_QUIT);
    assert_eq!(config.frames(), None);
    assert_eq!(Config::default().frames(), Some(DEFAULT_STEPS));
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
fn backend_is_kept_in_the_raw_list_for_the_factory() {
    let config = parse(&["--backend=vulkan"]).unwrap();
    assert_eq!(config.raw, vec!["--backend=vulkan".to_string()]);
}

#[test]
fn backend_without_its_value_is_an_error() {
    assert!(parse(&["--backend"]).is_err());
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
