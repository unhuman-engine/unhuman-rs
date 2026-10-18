use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::Event;

type Handler = Box<dyn FnMut(&Event) + Send + 'static>;

struct Subscriber {
    alive: Arc<AtomicBool>,
    handler: Handler,
}

/// Events waiting for the frame, and the handlers watching them as they arrive.
///
/// A subscription decides for itself whether it cares, rather than a walk of the
/// layer stack until one sets `Handled`, so neither the flag nor a category
/// bitmask has a counterpart here.
///
/// Ordering contract: [`push`](Self::push) queues the event *then* hands it to
/// every live subscriber, in subscription order, so the queue always reads in
/// push order. A subscriber that calls [`Sender::send`] from its handler appends
/// behind the event it is handling instead of re-entering the loop; the new
/// event is delivered by the next [`drain`](Self::drain), never recursively.
#[derive(Default)]
pub struct EventBus {
    queue: Arc<Mutex<VecDeque<Event>>>,
    subscribers: Vec<Subscriber>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues an event and hands it to every live subscriber, in the order they
    /// subscribed.
    pub fn push(&mut self, event: Event) {
        // Queue first: a reentrant `Sender::send` from a handler then lands
        // behind this event, which is the order a later drain must preserve.
        self.queue
            .lock()
            .expect("event queue poisoned")
            .push_back(event.clone());
        self.subscribers.retain(|s| s.alive.load(Ordering::Acquire));
        for subscriber in &mut self.subscribers {
            (subscriber.handler)(&event);
        }
    }

    /// Hands the frame's events over, oldest first, and empties the queue.
    ///
    /// The frame has exactly one consumer, so a second call with nothing pushed
    /// in between returns empty rather than replaying.
    pub fn drain(&mut self) -> Vec<Event> {
        self.queue
            .lock()
            .expect("event queue poisoned")
            .drain(..)
            .collect()
    }

    /// A handle a handler can hold to queue an event later, without reaching for
    /// the bus it is running inside of.
    pub fn sender(&self) -> Sender {
        Sender {
            queue: Arc::clone(&self.queue),
        }
    }

    pub fn subscribe(&mut self, f: impl FnMut(&Event) + Send + 'static) -> Subscription {
        let alive = Arc::new(AtomicBool::new(true));
        self.subscribers.push(Subscriber {
            alive: Arc::clone(&alive),
            handler: Box::new(f),
        });
        Subscription { alive }
    }

    pub fn len(&self) -> usize {
        self.queue.lock().expect("event queue poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().expect("event queue poisoned").is_empty()
    }
}

/// Queues events from outside a push. Shared with the bus it came from.
#[derive(Clone)]
pub struct Sender {
    queue: Arc<Mutex<VecDeque<Event>>>,
}

impl Sender {
    /// Appends to the frame's queue. Does not run subscribers: an event sent
    /// from a handler is deferred to the single consumer, not re-broadcast.
    pub fn send(&self, event: Event) {
        self.queue
            .lock()
            .expect("event queue poisoned")
            .push_back(event);
    }
}

/// Stops its handler seeing events once dropped. The flag is set here and the
/// handler is dropped at the next push, so a subscription can be dropped from a
/// place that has no access to the bus.
pub struct Subscription {
    alive: Arc<AtomicBool>,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
    }
}
