//! One log, two sinks: an engine record reaches the console the editor owns,
//! and the file log is named on the console's first line.

#[test]
fn engine_records_reach_the_console_and_the_log_path_is_first() {
    let directory = std::env::temp_dir().join(format!(
        "mirror-logging-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let path = directory.join("unhuman.log");

    assert_eq!(
        adam::logging::init_with_file(&path).as_deref(),
        Some(path.as_path())
    );

    let editor = mirror::Editor::new();
    let first = editor
        .console
        .lines()
        .into_iter()
        .next()
        .expect("a first console line");
    assert!(first.message.contains("log file:"));
    assert!(first.message.contains(&path.display().to_string()));

    log::warn!(target: "bifrost", "a frame failed to record");

    let lines = editor.console.lines();
    let record = lines
        .iter()
        .find(|line| line.target == "bifrost")
        .expect("the engine record reached the console");
    assert_eq!(record.level, mirror::Level::Warn);
    assert_eq!(record.message, "a frame failed to record");
    drop(lines);

    // The same record is in the file for a launch with no terminal.
    log::logger().flush();
    let text = std::fs::read_to_string(&path).expect("the file log exists");
    assert!(text.contains("a frame failed to record"));

    // The console's filter is the sink's, so it drops the record before the
    // buffer sees it while the file keeps writing it.
    editor.console.set_level(mirror::Level::Info);
    let debug = log::Record::builder()
        .args(format_args!("debug noise"))
        .level(log::Level::Debug)
        .target("bifrost")
        .build();
    assert!(!adam::logging::Sink::enabled(
        &editor.console,
        debug.metadata()
    ));

    editor.console.set_target("bifrost");
    log::info!(target: "bifrost::passes", "prefix kept");
    log::info!(target: "other", "prefix dropped");
    log::logger().flush();

    let lines = editor.console.lines();
    assert!(lines.iter().any(|line| line.message == "prefix kept"));
    assert!(!lines.iter().any(|line| line.message == "prefix dropped"));
    drop(lines);

    let text = std::fs::read_to_string(&path).expect("the file log exists");
    assert!(text.contains("prefix dropped"));

    drop(editor);
    let _ = std::fs::remove_dir_all(&directory);
}
