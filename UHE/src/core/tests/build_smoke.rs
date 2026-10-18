// Build smoke tests.
//
// These test that the workspace builds in the shapes it will be built in:
// feature combinations, dependency builds, targets. A behavioural suite cannot
// catch those.

use std::process::Command;

/// Run cargo and capture its combined output.
///
/// `--offline` is deliberately not used: a smoke test that cannot fetch is
/// testing the wrong thing, and CI has network.
fn cargo(args: &[&str]) -> (bool, String) {
    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("could not run cargo {args:?}: {e}"));
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

fn assert_cargo(args: &[&str], what: &str) {
    let (ok, output) = cargo(args);
    assert!(
        ok,
        "{what} failed.\ncommand: cargo {}\n\n{}",
        args.join(" "),
        output.lines().rev().take(40).collect::<Vec<_>>().join("\n"),
    );
}

/// Every crate in the workspace compiles with all targets, tests included.
///
/// The gate's `cargo build --workspace` skips test targets, so a broken test
/// file can sit unnoticed.
#[test]
fn the_workspace_builds_with_all_targets() {
    assert_cargo(
        &["build", "--workspace", "--all-targets"],
        "workspace build",
    );
}

/// Every crate compiles with `--release`. Debug and release differ in what the
/// optimiser rejects, and a backend full of `unsafe` and raw pointers is exactly
/// where a release-only build failure hides.
#[test]
fn the_workspace_builds_in_release() {
    assert_cargo(&["build", "--workspace", "--release"], "release build");
}

/// Doc tests build and run; `--no-run` is not accepted for them. They are the
/// only tests that execute code inside a `//!` comment, so a broken example
/// compiles fine and fails only here.
#[test]
fn doc_tests_run() {
    let (ok, output) = cargo(&["test", "--workspace", "--doc"]);
    assert!(
        ok,
        "doc tests failed:\n{}",
        output.lines().rev().take(30).collect::<Vec<_>>().join("\n"),
    );
}

/// The pure-logic crates build with no Vulkan SDK, no cmake and no driver.
///
/// This lets a fresh clone run tests on a laptop, and silently breaks when a
/// backend crate acquires a build dependency.
#[test]
fn the_logic_crates_build_without_any_driver() {
    assert_cargo(
        &[
            "build",
            "-p",
            "grind",
            "-p",
            "chassis",
            "-p",
            "adam",
            "-p",
            "forge",
            "--all-targets",
        ],
        "logic crates without a driver",
    );
}

/// The workspace resolves with no network and no registry, which is what
/// `--locked` proves: `Cargo.lock` is complete and current.
///
/// A lock file that drifts is how a CI run and a local run end up building
/// different versions of the same dependency.
#[test]
fn the_lock_file_is_current() {
    assert_cargo(
        &["build", "--workspace", "--locked", "--all-targets"],
        "locked build",
    );
}

/// The workspace is formatted. rustfmt-clean and clippy-clean are different
/// properties.
#[test]
fn the_workspace_is_formatted() {
    assert_cargo(&["fmt", "--all", "--check"], "rustfmt");
}

/// Clippy is clean with warnings denied.
#[test]
fn clippy_is_clean_with_warnings_denied() {
    assert_cargo(
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        "clippy",
    );
}

/// Every crate the workspace claims exists. A member listed in `Cargo.toml` but
/// missing from disk produces a confusing error deep inside cargo, and a crate on
/// disk but not listed is silently never tested.
#[test]
fn every_declared_member_exists_on_disk() {
    // `CARGO_MANIFEST_DIR` is this crate's directory, not the workspace root,
    // so the manifest it points at has no `[workspace]` table. Walk up to the
    // one that does.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = root
        .ancestors()
        .map(|ancestor| ancestor.join("Cargo.toml"))
        .find(|path| std::fs::read_to_string(path).is_ok_and(|text| text.contains("[workspace]")))
        .expect("a workspace root with a Cargo.toml");
    let text = std::fs::read_to_string(&manifest).expect("readable workspace manifest");
    let root = manifest
        .parent()
        .expect("the manifest has a parent")
        .to_path_buf();

    let mut checked = 0;
    for line in text.lines() {
        // A member line is a quoted path ending in `",`. The path is rooted at
        // the workspace, so it may be under `UHE/` or `UHE-EDITOR/` and the
        // parse must not assume either.
        let line = line.trim();
        let Some(rest) = line.strip_prefix('"') else {
            continue;
        };
        let Some(path) = rest.strip_suffix("\",") else {
            continue;
        };
        let directory = root.join(path);
        assert!(
            directory.join("Cargo.toml").is_file(),
            "Cargo.toml lists {path} but it has no Cargo.toml on disk",
        );
        checked += 1;
    }
    assert!(
        checked >= 5,
        "only {checked} members found; the parse is wrong"
    );
}

/// The backend crate keeps its file map in step with the sibling header tree.
///
/// Skips rather than fails when that tree is absent. When present, a class with
/// no Rust counterpart is the kind of gap that goes unnoticed: `VulkanRenderPass`
/// sat on a "dead code, skip it" list though it was the only path that worked on
/// a 1.0 device.
#[test]
fn the_backend_file_map_is_accounted_for() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // The sibling tree can sit at any depth. Walk up rather than assume one: a
    // depth-sensitive path that resolves to nothing makes the test skip while
    // reporting success.
    let Some(cpp) = find_upwards(root, "UHE/src/Platform/Vulkan") else {
        eprintln!("SKIP: the C++ tree is not present");
        return;
    };

    let rust_src = root.join("UHE/Stormbreaker/vulkan/src");
    let rust: Vec<String> = walk(&rust_src)
        .into_iter()
        .map(|p| {
            p.file_stem()
                .expect("a .rs file has a stem")
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    let cpp_classes: Vec<String> = walk(&cpp)
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "h"))
        .map(|p| {
            p.file_stem()
                .expect("a .h file has a stem")
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    // Every class deliberately skipped, and why. A new entry is a decision,
    // which is why it is a list rather than a pattern.
    let skipped: &[(&str, &str)] = &[
        (
            "VulkanBinaryFence",
            "VulkanFence.h:16 says it overlaps with VulkanFence and should be consolidated",
        ),
        (
            "VulkanBinarySemaphore",
            "its .cpp says 'Intentionally empty: ... is an unimplemented placeholder'",
        ),
        (
            "VulkanShaderManager",
            "Init() and ShutDown() are both empty bodies",
        ),
        (
            "VulkanImGuiBackend",
            "not skipped -- it belongs with the UI layer, not the backend",
        ),
        (
            "VulkanImGuiPass",
            "not skipped -- it belongs with the UI layer",
        ),
    ];

    let mut missing = Vec::new();
    for class in &cpp_classes {
        if rust.contains(class) {
            continue;
        }
        if let Some((_, reason)) = skipped.iter().find(|(name, _)| name == class) {
            eprintln!("skipped {class}: {reason}");
            continue;
        }
        missing.push(class.clone());
    }

    assert!(
        missing.is_empty(),
        "C++ classes with no Rust counterpart and no recorded reason: {missing:?}. \
         Either port them or add a line to `skipped` saying why.",
    );
}

/// Find `relative` in any ancestor. Depth-sensitive paths have caused repeated
/// test failures in this repository.
fn find_upwards(from: &std::path::Path, relative: &str) -> Option<std::path::PathBuf> {
    for ancestor in from.ancestors() {
        let candidate = ancestor.join(relative);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

fn walk(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}
