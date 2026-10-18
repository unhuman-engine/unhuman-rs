//! Headless binary: the engine loop with no window and no GPU, plus a
//! deterministic capture/replay layer.
//!
//! `--record` writes a frame sequence to a file; `--replay` runs it back and
//! checks it. Both are windowless and need no device, which is what makes a
//! capture replayable on a machine that has none.

mod capture;
mod config;
mod event;
mod input;
mod plugins;
mod runner;
mod session;

use config::Config;

fn main() {
    adam::logging::init();

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let config = match Config::parse(&arguments) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("nocap: {error}");
            std::process::exit(2);
        }
    };

    if config.help {
        println!("{}", config::USAGE);
        return;
    }

    let result = match (&config.record, &config.replay) {
        (Some(path), _) => runner::record(&config, path),
        (None, Some(path)) => runner::replay(&config, path),
        (None, None) => runner::summary(&config),
    };

    if let Err(error) = result {
        eprintln!("nocap: {error}");
        std::process::exit(1);
    }
}
