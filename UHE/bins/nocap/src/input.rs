//! The input a capture is recorded from.
//!
//! A live recorder would take the window's events. Headless, the input is a
//! fixed function of the frame index instead, so `--record` is reproducible
//! with no window and no display. It still exercises the parts the digest
//! covers: mouse look, one shot per press, and a held key.

use pings::{Event, KeyCode, MouseButton};

/// Events delivered to frame `frame` (zero-based).
pub fn scripted(frame: u64) -> Vec<Event> {
    let index = frame as f32;
    let mut events = vec![Event::MouseMoved {
        x: 640.0 + index * 2.0,
        y: 360.0 + index * 0.5,
    }];

    // A press held for one frame, so the aim lab's edge detection sees it.
    match frame % 30 {
        0 => events.push(Event::MouseButtonPressed(MouseButton::Left)),
        1 => events.push(Event::MouseButtonReleased(MouseButton::Left)),
        _ => {}
    }

    // Held across frames, then released: exercises held-key state and its end.
    if frame == 5 {
        events.push(Event::KeyPressed {
            key: KeyCode::W,
            repeat: false,
        });
    }
    if frame == 25 {
        events.push(Event::KeyReleased { key: KeyCode::W });
    }

    events
}
