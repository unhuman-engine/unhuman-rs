//! The event codec: every variant, including the payload shapes that must not
//! lose bits.

#[path = "../src/event.rs"]
mod event;

use pings::{Event, KeyCode, MouseButton};

fn sample() -> Vec<Event> {
    vec![
        Event::WindowResized {
            width: 1280,
            height: 720,
        },
        Event::WindowCloseRequested,
        Event::WindowFocused(true),
        Event::WindowFocused(false),
        Event::KeyPressed {
            key: KeyCode::W,
            repeat: false,
        },
        Event::KeyPressed {
            key: KeyCode::Unknown(42),
            repeat: true,
        },
        Event::KeyReleased { key: KeyCode::F13 },
        Event::TextInput("hello world".to_string()),
        Event::MouseButtonPressed(MouseButton::Left),
        Event::MouseButtonReleased(MouseButton::Other(7)),
        Event::MouseMoved { x: 1.5, y: -2.25 },
        Event::MouseScrolled { dx: 0.0, dy: 3.0 },
        Event::FrameBegin,
        Event::FrameEnd,
    ]
}

#[test]
fn every_event_variant_round_trips() {
    for event in sample() {
        let token = event::encode(&event);
        assert!(
            !token.contains(char::is_whitespace),
            "{token:?} would split a frame line"
        );
        assert_eq!(event::decode(&token).unwrap(), event, "token {token:?}");
    }
}

/// Decimal values that are not exactly representable: the hex-bit encoding
/// must return the identical `f32`, not a nearest neighbour.
#[test]
fn floats_round_trip_bit_for_bit() {
    let event = Event::MouseMoved {
        x: 0.1,
        y: 1.0 / 3.0,
    };
    assert_eq!(event::decode(&event::encode(&event)).unwrap(), event);
}

#[test]
fn unknown_tags_and_payloads_are_errors() {
    assert!(event::decode("nope").is_err());
    assert!(event::decode("keyup:NotAKey").is_err());
    assert!(event::decode("mouse:nothex:0").is_err());
    assert!(event::decode("").is_err());
}
