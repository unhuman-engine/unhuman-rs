use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

pub type Job = Box<dyn FnOnce() + Send + 'static>;

/// Scheduling urgency, so a frame's recording is not queued behind a decode.
///
/// The pool drains high before normal before low. Jobs within one priority stay
/// FIFO; this is not a scheduler, only a way to say a decode may be a frame
/// late while a pass may not.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub enum Priority {
    /// A pass the frame is about to submit.
    High,
    #[default]
    Normal,
    /// A texture decode, mesh extract or other work a frame can wait out.
    Low,
}

/// A cooperative cancellation flag, shared between the submitter and the pool.
///
/// The pool checks it once before running a job; a job already executing is not
/// interrupted, because a decode stopped mid-way would have to unwind its own
/// state. A cancelled job is dropped without running, and its counter still
/// settles, so a waiter never blocks on it.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

struct QueuedJob {
    job: Job,
    counter: Option<Arc<JobCounter>>,
    cancel: Option<CancelToken>,
}

/// Three FIFO lanes, drained high, then normal, then low.
#[derive(Default)]
struct Queue {
    high: VecDeque<QueuedJob>,
    normal: VecDeque<QueuedJob>,
    low: VecDeque<QueuedJob>,
}

impl Queue {
    fn push(&mut self, job: QueuedJob, priority: Priority) {
        match priority {
            Priority::High => self.high.push_back(job),
            Priority::Normal => self.normal.push_back(job),
            Priority::Low => self.low.push_back(job),
        }
    }

    fn pop(&mut self) -> Option<QueuedJob> {
        self.high
            .pop_front()
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
    }

    fn is_empty(&self) -> bool {
        self.high.is_empty() && self.normal.is_empty() && self.low.is_empty()
    }
}

thread_local! {
    /// Unique index of the calling thread. Main thread is 0, workers are 1..=N.
    static THREAD_INDEX: Cell<u32> = const { Cell::new(0) };
}

/// A lightweight counter to wait on a group of jobs.
///
/// This is the core of the task graph: jobs can depend on counters.
#[derive(Default)]
pub struct JobCounter(AtomicUsize);

impl JobCounter {
    pub fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    pub fn pending(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }

    pub fn is_complete(&self) -> bool {
        self.pending() == 0
    }

    fn inc(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }

    /// `AcqRel`, not `Release`.
    ///
    /// A waiter observes zero via a single `Acquire` load, which only synchronises
    /// with the thread that performed the final decrement. `AcqRel` on every
    /// decrement chains the RMWs into one release sequence, so the last carries
    /// every prior job's writes.
    fn dec(&self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    running: AtomicBool,
    in_flight: AtomicI32,
}

/// Runs one dequeued job and settles both counters. Shared by `try_pop_job` and
/// the worker loop so there is exactly one place that decrements them.
///
/// A cancelled job does not run, but its counter still settles: a waiter must
/// not block on work that was abandoned, and a load's result arrives (or does
/// not) through its own channel. The counter settles after the body returns but
/// before the closure is destroyed, so a counter means "the body finished", not
/// "the captures are gone".
fn run_queued(job: QueuedJob, shared: &Shared) {
    let QueuedJob {
        job,
        counter,
        cancel,
    } = job;
    let cancelled = matches!(&cancel, Some(token) if token.is_cancelled());
    if !cancelled {
        job();
    }
    if let Some(counter) = counter {
        counter.dec();
    }
    // AcqRel for the same reason as `JobCounter::dec`.
    shared.in_flight.fetch_sub(1, Ordering::AcqRel);
}

fn pop(shared: &Shared) -> Option<QueuedJob> {
    shared.queue.lock().unwrap().pop()
}

/// A thread pool. Index 0 is the calling (main) thread; spawned workers are 1..=N.
///
/// `thread_count()` exists because per-thread Vulkan command pools need one
/// command list per thread, and the mapping has to be known at submit time.
pub struct Jobs {
    shared: Arc<Shared>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl Jobs {
    /// Spawns `hardware_concurrency() - 1` workers, leaving one core for the
    /// calling thread. The `- 1` matters on big.LITTLE targets.
    pub fn new() -> Self {
        let cores = thread::available_parallelism().map_or(4, |n| n.get());
        let num_workers = cores.saturating_sub(1).max(1);

        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
            running: AtomicBool::new(true),
            in_flight: AtomicI32::new(0),
        });

        let mut workers = Vec::with_capacity(num_workers);
        for index in 0..num_workers {
            let shared = Arc::clone(&shared);
            workers.push(
                thread::Builder::new()
                    .name(format!("grind-worker-{index}"))
                    .spawn(move || worker_loop(index as u32 + 1, shared))
                    .expect("failed to spawn job worker"),
            );
        }

        Self { shared, workers }
    }

    /// Kicks a single normal-priority job. If a counter is attached it is
    /// incremented here, and decremented when the job finishes.
    pub fn execute(&self, job: Job, counter: Option<Arc<JobCounter>>) {
        self.submit(job, counter, Priority::Normal, None);
    }

    /// Kicks a job at an explicit [`Priority`].
    pub fn execute_priority(&self, job: Job, counter: Option<Arc<JobCounter>>, priority: Priority) {
        self.submit(job, counter, priority, None);
    }

    /// Kicks a job that a [`CancelToken`] can drop before it runs.
    pub fn execute_cancellable(
        &self,
        job: Job,
        counter: Option<Arc<JobCounter>>,
        priority: Priority,
        cancel: CancelToken,
    ) {
        self.submit(job, counter, priority, Some(cancel));
    }

    fn submit(
        &self,
        job: Job,
        counter: Option<Arc<JobCounter>>,
        priority: Priority,
        cancel: Option<CancelToken>,
    ) {
        // Count the job as in-flight BEFORE it becomes visible to other
        // threads, so is_busy()/wait_for_all() can never miss it.
        self.shared.in_flight.fetch_add(1, Ordering::AcqRel);

        if let Some(counter) = &counter {
            counter.inc();
        }

        self.shared.queue.lock().unwrap().push(
            QueuedJob {
                job,
                counter,
                cancel,
            },
            priority,
        );
        self.shared.wake.notify_one();
    }

    /// Splits `count` into contiguous batches, running `f(i)` for each, and
    /// blocks until all iterations complete. Contiguous slices beat one job per
    /// element when the body is cheap.
    pub fn parallel_for<F>(&self, count: u32, f: F, min_batch_size: u32)
    where
        F: Fn(u32) + Send + Sync + 'static,
    {
        if count == 0 {
            return;
        }
        let min_batch_size = min_batch_size.max(1);
        let worker_count = self.workers.len() as u32;

        // Single iteration or single worker: run inline, no scheduling overhead.
        if count == 1 || worker_count == 1 {
            for i in 0..count {
                f(i);
            }
            return;
        }

        let counter = Arc::new(JobCounter::new());
        let f = Arc::new(f);

        // Aim for ~2 batches per worker so stragglers can be picked up by
        // threads that finish early.
        let mut batch_count = count.div_ceil(min_batch_size);
        batch_count = batch_count.min(worker_count * 2).max(1);
        let batch_size = count.div_ceil(batch_count);

        for batch in 0..batch_count {
            let begin = batch * batch_size;
            if begin >= count {
                break;
            }
            let end = (begin + batch_size).min(count);

            let f = Arc::clone(&f);
            // High: these batches are a pass a frame is waiting to submit, not
            // background work.
            self.execute_priority(
                Box::new(move || {
                    for i in begin..end {
                        f(i);
                    }
                }),
                Some(Arc::clone(&counter)),
                Priority::High,
            );
        }

        self.wait(&counter);
    }

    /// Blocks until `counter` reaches zero. While waiting the calling thread
    /// steals queued jobs and runs them, so waiting is never idle CPU time, and a
    /// counter whose jobs are all still queued still completes on a
    /// single-worker machine instead of deadlocking.
    pub fn wait(&self, counter: &JobCounter) {
        // The counter is re-checked every iteration and a successful pop does
        // NOT exit the loop. Folding the two into `while !done && !pop()` would
        // return as soon as the caller work-helped a single job, stranding the
        // rest of the batch.
        while !counter.is_complete() {
            if !self.try_pop_job() {
                // Queue is empty: the job is either running elsewhere or has
                // not been scheduled yet.
                thread::yield_now();
            }
        }
    }

    /// Blocks until every job kicked so far has completed. Unlike `Drop`, the
    /// job system keeps running afterwards.
    pub fn wait_for_all(&self) {
        while self.is_busy() {
            if !self.try_pop_job() {
                thread::yield_now();
            }
        }
    }

    /// True while any job kicked through this system is still queued or running.
    pub fn is_busy(&self) -> bool {
        self.shared.in_flight.load(Ordering::Acquire) > 0
    }

    /// Total threads: workers + the main thread. This is the per-frame command
    /// list count.
    pub fn thread_count(&self) -> u32 {
        self.workers.len() as u32 + 1
    }

    /// Index of the calling thread in `0..thread_count()`.
    pub fn current_thread_index() -> u32 {
        THREAD_INDEX.with(Cell::get)
    }

    /// Pops ONE queued job and runs it on the calling thread. Returns false if
    /// the queue was empty. Used by workers and by any thread work-helping
    /// inside `wait`/`wait_for_all`/`TaskGraph`.
    pub fn try_pop_job(&self) -> bool {
        match pop(&self.shared) {
            Some(job) => {
                run_queued(job, &self.shared);
                true
            }
            None => false,
        }
    }
}

impl Default for Jobs {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Jobs {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Release);
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(index: u32, shared: Arc<Shared>) {
    THREAD_INDEX.with(|cell| cell.set(index));

    while shared.running.load(Ordering::Acquire) {
        match pop(&shared) {
            Some(job) => run_queued(job, &shared),
            None => {
                // Queue is empty: sleep until new work arrives or shutdown.
                let queue = shared.queue.lock().unwrap();
                let _unused = shared
                    .wake
                    .wait_while(queue, |queue| {
                        queue.is_empty() && shared.running.load(Ordering::Acquire)
                    })
                    .unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(order: &Arc<Mutex<Vec<u32>>>, value: u32) -> Job {
        let order = Arc::clone(order);
        Box::new(move || order.lock().unwrap().push(value))
    }

    #[test]
    fn the_queue_drains_high_before_normal_before_low() {
        let mut queue = Queue::default();
        let order = Arc::new(Mutex::new(Vec::new()));

        // Pushed in the opposite order to the one they must come out in.
        for (value, priority) in [
            (1, Priority::Low),
            (2, Priority::Normal),
            (3, Priority::High),
        ] {
            queue.push(
                QueuedJob {
                    job: marker(&order, value),
                    counter: None,
                    cancel: None,
                },
                priority,
            );
        }

        let shared = Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
            running: AtomicBool::new(true),
            in_flight: AtomicI32::new(0),
        };
        while let Some(job) = queue.pop() {
            run_queued(job, &shared);
        }
        assert_eq!(*order.lock().unwrap(), vec![3, 2, 1]);
    }

    #[test]
    fn a_cancelled_job_is_dropped_but_its_counter_settles() {
        let shared = Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
            running: AtomicBool::new(true),
            in_flight: AtomicI32::new(0),
        };
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        let token = CancelToken::new();
        token.cancel();

        let counter = Arc::new(JobCounter::new());
        counter.inc();
        run_queued(
            QueuedJob {
                job: Box::new(move || flag.store(true, Ordering::Release)),
                counter: Some(Arc::clone(&counter)),
                cancel: Some(token),
            },
            &shared,
        );

        assert!(!ran.load(Ordering::Acquire), "a cancelled job ran");
        assert!(
            counter.is_complete(),
            "a cancelled job left its counter armed"
        );
    }
}
