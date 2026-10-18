//! Sandbox: a thin composition root over `adam::App`.
//!
//! Windowed by default, headless with `--headless`. All behavior lives in the
//! plugin crates; this binary names them and picks the surface.

mod args;
mod headless;
mod plugins;
mod windowed;

use args::Config;

fn main() {
    adam::logging::init();

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let config = match Config::parse(&arguments) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("sandbox: {error}");
            std::process::exit(2);
        }
    };

    if config.help {
        println!("{}", args::USAGE);
        return;
    }

    let run = if config.headless {
        headless::run(&config)
    } else {
        windowed::run(&config)
    };
    if let Err(error) = run {
        eprintln!("sandbox: {error}");
        std::process::exit(1);
    }
}
