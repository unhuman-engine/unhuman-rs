//! One test, because a process has one logger: it proves the same record
//! reaches the file and a sink attached at runtime, from another thread.

use std::sync::{Arc, Mutex};

use adam::logging::{self, Sink};

#[derive(Default)]
struct Collector(Mutex<Vec<(log::Level, String, String)>>);

impl Sink for Collector {
    fn log(&self, record: &log::Record<'_>) {
        self.0.lock().unwrap().push((
            record.level(),
            record.target().to_string(),
            record.args().to_string(),
        ));
    }
}

#[test]
fn one_log_reaches_the_file_and_an_attached_sink() {
    let directory = std::env::temp_dir().join(format!(
        "adam-logging-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let path = directory.join("unhuman.log");

    assert_eq!(
        logging::init_with_file(&path).as_deref(),
        Some(path.as_path())
    );
    assert_eq!(logging::log_path().as_deref(), Some(path.as_path()));

    let collector = Arc::new(Collector::default());
    let id = logging::attach(collector.clone());

    log::error!(target: "logging-test", "from the main thread");
    let workers: Vec<_> = (0..2)
        .map(|worker| {
            std::thread::spawn(move || {
                log::error!(target: "logging-test", "from worker {worker}");
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }

    log::logger().flush();

    let records = collector.0.lock().unwrap();
    assert!(records.iter().any(|(level, target, message)| {
        *level == log::Level::Error && target == "logging-test" && message == "from the main thread"
    }));
    assert!(records
        .iter()
        .any(|(_, _, message)| message == "from worker 0"));
    assert!(records
        .iter()
        .any(|(_, _, message)| message == "from worker 1"));
    drop(records);

    let text = std::fs::read_to_string(&path).expect("the file log exists");
    assert!(text.contains("from the main thread"));
    assert!(text.contains("from worker 1"));

    assert!(logging::detach(id));
    let _ = std::fs::remove_dir_all(&directory);
}
