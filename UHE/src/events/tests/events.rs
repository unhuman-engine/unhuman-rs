use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::Arc;

use pings::{Event, EventBus, KeyCode, MouseButton};

#[test]
fn drain_hands_the_frames_events_over_in_order() {
    let mut bus = EventBus::new();
    assert!(bus.is_empty());
    assert!(bus.drain().is_empty());

    bus.push(Event::FrameBegin);
    bus.push(Event::KeyPressed {
        key: KeyCode::W,
        repeat: false,
    });
    bus.push(Event::MouseMoved { x: 1.5, y: -2.0 });
    bus.push(Event::FrameEnd);

    assert_eq!(bus.len(), 4);
    assert_eq!(
        bus.drain(),
        vec![
            Event::FrameBegin,
            Event::KeyPressed {
                key: KeyCode::W,
                repeat: false
            },
            Event::MouseMoved { x: 1.5, y: -2.0 },
            Event::FrameEnd,
        ]
    );
    assert!(bus.is_empty());
    assert!(bus.drain().is_empty());
}

#[test]
fn subscribers_run_on_push_in_subscription_order() {
    let mut bus = EventBus::new();
    let (tx, rx) = channel::<(&'static str, Event)>();

    let first = tx.clone();
    let _first = bus.subscribe(move |event| {
        let _ = first.send(("first", event.clone()));
    });

    bus.push(Event::FrameBegin);

    let second = tx.clone();
    let _second = bus.subscribe(move |event| {
        let _ = second.send(("second", event.clone()));
    });

    bus.push(Event::MouseScrolled { dx: 0.0, dy: 1.0 });

    assert_eq!(
        rx.try_iter().collect::<Vec<_>>(),
        vec![
            ("first", Event::FrameBegin),
            ("first", Event::MouseScrolled { dx: 0.0, dy: 1.0 }),
            ("second", Event::MouseScrolled { dx: 0.0, dy: 1.0 }),
        ]
    );
}

#[test]
fn a_dropped_subscription_sees_nothing_more() {
    let mut bus = EventBus::new();
    let seen = Arc::new(AtomicUsize::new(0));

    let dropped = Arc::clone(&seen);
    let subscription = bus.subscribe(move |_| {
        let _ = dropped.fetch_add(1, Ordering::Relaxed);
    });

    let kept = Arc::clone(&seen);
    let _kept = bus.subscribe(move |event| {
        if matches!(event, Event::KeyReleased { .. }) {
            let _ = kept.fetch_add(1, Ordering::Relaxed);
        }
    });

    bus.push(Event::FrameBegin);
    assert_eq!(seen.load(Ordering::Relaxed), 1);

    drop(subscription);
    bus.push(Event::KeyReleased {
        key: KeyCode::Escape,
    });
    assert_eq!(seen.load(Ordering::Relaxed), 2);
}

#[test]
fn a_reentrant_send_is_queued_behind_the_event_in_hand_not_rebroadcast() {
    let mut bus = EventBus::new();
    let (tx, rx) = channel::<Event>();

    let reenter = bus.sender();
    let _sub = bus.subscribe(move |event| {
        let _ = tx.send(event.clone());
        if *event == Event::FrameBegin {
            reenter.send(Event::FrameEnd);
        }
    });

    bus.push(Event::FrameBegin);

    // The handler ran once for FrameBegin; the send did not re-enter it.
    assert_eq!(rx.try_iter().collect::<Vec<_>>(), vec![Event::FrameBegin]);
    // FrameEnd is behind FrameBegin in the queue, oldest first.
    assert_eq!(bus.drain(), vec![Event::FrameBegin, Event::FrameEnd]);
}

#[test]
fn the_frame_has_one_consumer_that_empties_the_queue() {
    let mut bus = EventBus::new();
    bus.push(Event::FrameBegin);
    bus.push(Event::FrameEnd);

    assert_eq!(bus.drain().len(), 2);
    assert!(bus.is_empty());
    assert!(bus.drain().is_empty());
}

#[test]
fn frame_begin_and_end_are_ordinary_events() {
    let mut bus = EventBus::new();
    let (tx, rx) = channel::<Event>();
    let _all = bus.subscribe(move |event| {
        let _ = tx.send(event.clone());
    });

    bus.push(Event::FrameBegin);
    bus.push(Event::MouseButtonPressed(MouseButton::Left));
    bus.push(Event::TextInput("a".to_string()));
    bus.push(Event::FrameEnd);

    assert_eq!(
        rx.try_iter().collect::<Vec<_>>(),
        vec![
            Event::FrameBegin,
            Event::MouseButtonPressed(MouseButton::Left),
            Event::TextInput("a".to_string()),
            Event::FrameEnd,
        ]
    );
    assert_eq!(bus.drain().len(), 4);
}
