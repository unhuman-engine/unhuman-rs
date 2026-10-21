use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use std::time::Duration;

use std::sync::atomic::AtomicBool;

use grind::{CancelToken, JobCounter, Jobs, Priority, TaskGraph};

#[test]
fn execute_and_wait_runs_the_job() {
    let jobs = Jobs::new();
    let ran = Arc::new(AtomicUsize::new(0));

    let counter = Arc::new(JobCounter::new());
    let flag = Arc::clone(&ran);
    jobs.execute(
        Box::new(move || {
            flag.fetch_add(1, Ordering::AcqRel);
        }),
        Some(Arc::clone(&counter)),
    );

    jobs.wait(&counter);
    assert_eq!(ran.load(Ordering::Acquire), 1);
    assert!(counter.is_complete());
}

#[test]
fn parallel_for_covers_every_index_exactly_once() {
    let jobs = Jobs::new();
    let seen = Arc::new(Mutex::new(Vec::new()));

    let sink = Arc::clone(&seen);
    jobs.parallel_for(1_000, move |i| sink.lock().unwrap().push(i), 1);

    let seen = seen.lock().unwrap().clone();
    let unique: HashSet<u32> = seen.iter().copied().collect();

    assert_eq!(seen.len(), 1_000, "an index ran more than once");
    assert_eq!(unique.len(), 1_000, "an index never ran");
    assert_eq!(*seen.iter().max().unwrap(), 999);
}

#[test]
fn parallel_for_on_zero_is_a_noop() {
    let jobs = Jobs::new();
    jobs.parallel_for(0, |i| assert!(i == 0), 1);
}

#[test]
fn wait_for_all_drains_everything() {
    let jobs = Jobs::new();
    let done = Arc::new(AtomicUsize::new(0));

    for _ in 0..64 {
        let flag = Arc::clone(&done);
        jobs.execute(
            Box::new(move || {
                flag.fetch_add(1, Ordering::AcqRel);
            }),
            None,
        );
    }

    jobs.wait_for_all();
    assert_eq!(done.load(Ordering::Acquire), 64);
    assert!(!jobs.is_busy());
}

#[test]
fn thread_index_is_in_range() {
    let jobs = Jobs::new();
    assert!(jobs.thread_count() >= 2, "expected at least one worker");
    assert_eq!(grind::Jobs::current_thread_index(), 0);

    let observed = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    jobs.execute(
        Box::new(move || sink.lock().unwrap().push(Jobs::current_thread_index())),
        None,
    );
    jobs.wait_for_all();

    let indices = observed.lock().unwrap().clone();
    for index in indices {
        assert!(index < jobs.thread_count());
    }
}

/// The invariant behind `dep_count == 0` root scanning: a task is dispatched
/// exactly once, by the root scan or by its last arriving predecessor, never
/// both. A violation shows up as a task running twice.
#[test]
fn every_task_runs_exactly_once() {
    let jobs = Arc::new(Jobs::new());
    let mut graph = TaskGraph::new();

    let runs = Arc::new(Mutex::new(vec![0u32; 64]));

    let mut ids = Vec::new();
    for i in 0..64 {
        let runs = Arc::clone(&runs);
        ids.push(graph.create_task(move || runs.lock().unwrap()[i] += 1));
    }
    // A diamond shape plus a wide fan-in, so several predecessors race to
    // release the same dependent.
    for &tail in &ids[32..] {
        for &head in &ids[..32] {
            graph.add_dependency(head, tail);
        }
    }
    graph.add_dependency(ids[0], ids[32]);

    graph.execute(Arc::clone(&jobs));

    let runs = runs.lock().unwrap().clone();
    assert!(
        runs.iter().all(|&n| n == 1),
        "a task ran more than once: {runs:?}"
    );
}

#[test]
fn dependencies_are_respected() {
    let jobs = Arc::new(Jobs::new());
    let mut graph = TaskGraph::new();

    let order = Arc::new(Mutex::new(Vec::new()));
    let ids: Vec<_> = (0..32)
        .map(|i| {
            let order = Arc::clone(&order);
            graph.create_task(move || order.lock().unwrap().push(i))
        })
        .collect();

    // Each task depends on the previous one: a chain forces strict ordering.
    for pair in ids.windows(2) {
        graph.add_dependency(pair[0], pair[1]);
    }

    graph.execute(Arc::clone(&jobs));

    let order = order.lock().unwrap().clone();
    assert_eq!(order.len(), 32, "a task did not run");
    let expected: Vec<u32> = (0..32).collect();
    assert_eq!(order, expected, "chain ran out of order");
}

#[test]
fn graph_can_be_executed_more_than_once() {
    let jobs = Arc::new(Jobs::new());
    let mut graph = TaskGraph::new();

    let runs = Arc::new(AtomicUsize::new(0));
    let a = {
        let runs = Arc::clone(&runs);
        graph.create_task(move || {
            runs.fetch_add(1, Ordering::AcqRel);
        })
    };
    let b = {
        let runs = Arc::clone(&runs);
        graph.create_task(move || {
            runs.fetch_add(1, Ordering::AcqRel);
        })
    };
    graph.add_dependency(a, b);

    for _ in 0..5 {
        graph.execute(Arc::clone(&jobs));
    }

    assert_eq!(
        runs.load(Ordering::Acquire),
        10,
        "counters were not re-armed"
    );
}

#[test]
fn empty_graph_is_fine() {
    let jobs = Arc::new(Jobs::new());
    let mut graph = TaskGraph::new();
    graph.execute(jobs);
    assert!(graph.is_empty());
}

#[test]
fn reset_clears_the_graph() {
    let mut graph = TaskGraph::new();
    graph.create_task(|| {});
    assert_eq!(graph.len(), 1);
    graph.reset();
    assert_eq!(graph.len(), 0);
}

#[test]
#[should_panic(expected = "cannot depend on itself")]
fn self_dependency_is_rejected() {
    let mut graph = TaskGraph::new();
    let id = graph.create_task(|| {});
    graph.add_dependency(id, id);
}

#[test]
#[should_panic(expected = "out of range")]
fn out_of_range_dependency_is_rejected() {
    let mut graph = TaskGraph::new();
    graph.create_task(|| {});
    graph.add_dependency(0, 7);
}
/// Regression: `wait` used to be written as
/// `while !counter.is_complete() && !self.try_pop_job()`, so the loop exited as
/// soon as the calling thread work-helped a single job and stranded the rest.
/// These bodies are slow enough that the caller always pops at least one.
#[test]
fn wait_does_not_return_early_when_the_caller_helps() {
    let jobs = Jobs::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let counter = Arc::new(JobCounter::new());

    for _ in 0..32 {
        let flag = Arc::clone(&ran);
        jobs.execute(
            Box::new(move || {
                std::thread::sleep(Duration::from_micros(500));
                flag.fetch_add(1, Ordering::AcqRel);
            }),
            Some(Arc::clone(&counter)),
        );
    }

    jobs.wait(&counter);
    assert_eq!(ran.load(Ordering::Acquire), 32, "wait returned early");
    assert!(!jobs.is_busy(), "jobs still in flight after wait");
}

#[test]
fn wait_for_all_does_not_return_early_when_the_caller_helps() {
    let jobs = Jobs::new();
    let ran = Arc::new(AtomicUsize::new(0));

    for _ in 0..32 {
        let flag = Arc::clone(&ran);
        jobs.execute(
            Box::new(move || {
                std::thread::sleep(Duration::from_micros(500));
                flag.fetch_add(1, Ordering::AcqRel);
            }),
            None,
        );
    }

    jobs.wait_for_all();
    assert_eq!(
        ran.load(Ordering::Acquire),
        32,
        "wait_for_all returned early"
    );
}

#[test]
fn a_cancelled_job_does_not_run_but_its_counter_settles() {
    let jobs = Jobs::new();
    let ran = Arc::new(AtomicBool::new(false));
    let token = CancelToken::new();
    token.cancel();

    let counter = Arc::new(JobCounter::new());
    let flag = Arc::clone(&ran);
    jobs.execute_cancellable(
        Box::new(move || flag.store(true, Ordering::Release)),
        Some(Arc::clone(&counter)),
        Priority::Low,
        token,
    );

    // A waiter must not block on work that was abandoned.
    jobs.wait(&counter);
    assert!(!ran.load(Ordering::Acquire), "a cancelled job ran");
}

#[test]
fn high_priority_jobs_run_and_settle_their_counter() {
    let jobs = Jobs::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let counter = Arc::new(JobCounter::new());

    for _ in 0..16 {
        let flag = Arc::clone(&ran);
        jobs.execute_priority(
            Box::new(move || {
                flag.fetch_add(1, Ordering::AcqRel);
            }),
            Some(Arc::clone(&counter)),
            Priority::High,
        );
    }

    jobs.wait(&counter);
    assert_eq!(ran.load(Ordering::Acquire), 16);
}
