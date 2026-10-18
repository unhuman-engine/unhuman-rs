//! Record and replay, end to end through the built binary.
//!
//! The binary is the unit under test: these run the real data path with no
//! window and no GPU, which is exactly what a replay on a GPU-less machine is.

use std::path::PathBuf;
use std::process::{Command, Output};

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nocap-test-{}-{name}.cap", std::process::id()))
}

fn nocap(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nocap"))
        .args(args)
        .output()
        .expect("run nocap")
}

fn record(path: &str, steps: &str) {
    let output = nocap(&["--record", path, "--steps", steps]);
    assert!(
        output.status.success(),
        "record failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn recording_twice_is_byte_identical() {
    let first = temp("determinism-a");
    let second = temp("determinism-b");
    record(first.to_str().unwrap(), "40");
    record(second.to_str().unwrap(), "40");

    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap(),
        "the same input must produce the same capture"
    );

    let _ = std::fs::remove_file(&first);
    let _ = std::fs::remove_file(&second);
}

#[test]
fn a_recorded_capture_replays_headless() {
    let path = temp("replay");
    let path = path.to_str().unwrap();
    record(path, "60");

    let output = nocap(&["--replay", path]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "replay failed: {stdout}");
    assert!(stdout.contains("digests match"), "{stdout}");

    let _ = std::fs::remove_file(path);
}

/// A tampered digest must fail at that frame. Without this the "digests match"
/// line could come from a check that compares nothing.
#[test]
fn a_tampered_digest_fails_at_that_frame() {
    let path = temp("tampered");
    let path = path.to_str().unwrap();
    record(path, "10");

    let text = std::fs::read_to_string(path).unwrap();
    std::fs::write(path, tamper_first_digest(&text)).unwrap();

    let output = nocap(&["--replay", path]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "tampered capture replayed");
    assert!(stderr.contains("diverged"), "{stderr}");

    let _ = std::fs::remove_file(path);
}

/// A capture recorded with a different plugin set must be refused rather than
/// compared against a different engine.
#[test]
fn a_capture_from_another_composition_is_refused() {
    let path = temp("composition");
    let path = path.to_str().unwrap();
    std::fs::write(
        path,
        "nocap 1\nplugins something-else\nframe 16666 0000000000000000\n",
    )
    .unwrap();

    let output = nocap(&["--replay", path]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "mismatched plugins were accepted");
    assert!(stderr.contains("do not match"), "{stderr}");

    let _ = std::fs::remove_file(path);
}

fn tamper_first_digest(text: &str) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    for line in &mut lines {
        if let Some(rest) = line.strip_prefix("frame ") {
            let digest = rest
                .split_whitespace()
                .nth(1)
                .expect("a frame digest")
                .to_string();
            let flipped = format!(
                "{:016x}",
                u64::from_str_radix(&digest, 16).unwrap() ^ 0xdead
            );
            *line = line.replacen(&digest, &flipped, 1);
            break;
        }
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}
