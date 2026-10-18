//! The on-disk encoding of one `pings::Event`.
//!
//! An event is one `:`-separated token, so a whole frame of input fits on one
//! whitespace-free line. Floats are written as their IEEE-754 bit pattern in
//! hex: a cursor position then round-trips exactly, instead of through a
//! shortest-decimal print that can land a bit off on another build.

use pings::{Event, KeyCode, MouseButton};

/// Encodes an event as a single token, with no whitespace.
pub fn encode(event: &Event) -> String {
    match event {
        Event::WindowResized { width, height } => format!("resize:{width}:{height}"),
        Event::WindowCloseRequested => "close".to_string(),
        Event::WindowFocused(focused) => format!("focus:{}", *focused as u8),
        Event::KeyPressed { key, repeat } => {
            format!("keydown:{}:{}", key_name(*key), *repeat as u8)
        }
        Event::KeyReleased { key } => format!("keyup:{}", key_name(*key)),
        Event::TextInput(text) => format!("text:{}", hex(text.as_bytes())),
        Event::MouseButtonPressed(button) => format!("mbdown:{}", button_name(*button)),
        Event::MouseButtonReleased(button) => format!("mbup:{}", button_name(*button)),
        Event::MouseMoved { x, y } => format!("mouse:{}:{}", bits(*x), bits(*y)),
        Event::MouseScrolled { dx, dy } => format!("scroll:{}:{}", bits(*dx), bits(*dy)),
        Event::FrameBegin => "fbegin".to_string(),
        Event::FrameEnd => "fend".to_string(),
    }
}

/// The inverse of [`encode`]. Unknown tags are an error, not a dropped event.
pub fn decode(token: &str) -> Result<Event, String> {
    let fields: Vec<&str> = token.split(':').collect();
    match fields.first().copied() {
        Some("resize") => Ok(Event::WindowResized {
            width: number(fields.get(1))? as u32,
            height: number(fields.get(2))? as u32,
        }),
        Some("close") => Ok(Event::WindowCloseRequested),
        Some("focus") => Ok(Event::WindowFocused(number(fields.get(1))? != 0)),
        Some("keydown") => Ok(Event::KeyPressed {
            key: key_from(fields.get(1).copied().unwrap_or_default())?,
            repeat: number(fields.get(2))? != 0,
        }),
        Some("keyup") => Ok(Event::KeyReleased {
            key: key_from(fields.get(1).copied().unwrap_or_default())?,
        }),
        Some("text") => Ok(Event::TextInput(
            String::from_utf8_lossy(&unhex(fields.get(1).copied().unwrap_or_default())?)
                .into_owned(),
        )),
        Some("mbdown") => Ok(Event::MouseButtonPressed(button_from(
            fields.get(1).copied().unwrap_or_default(),
        )?)),
        Some("mbup") => Ok(Event::MouseButtonReleased(button_from(
            fields.get(1).copied().unwrap_or_default(),
        )?)),
        Some("mouse") => Ok(Event::MouseMoved {
            x: float(fields.get(1))?,
            y: float(fields.get(2))?,
        }),
        Some("scroll") => Ok(Event::MouseScrolled {
            dx: float(fields.get(1))?,
            dy: float(fields.get(2))?,
        }),
        Some("fbegin") => Ok(Event::FrameBegin),
        Some("fend") => Ok(Event::FrameEnd),
        Some(other) => Err(format!("unknown event tag {other:?}")),
        None => Err("empty event token".to_string()),
    }
}

fn number(field: Option<&&str>) -> Result<u64, String> {
    field
        .ok_or_else(|| "missing field".to_string())?
        .parse()
        .map_err(|_| format!("expected a number, got {field:?}"))
}

fn float(field: Option<&&str>) -> Result<f32, String> {
    u32::from_str_radix(field.ok_or_else(|| "missing field".to_string())?, 16)
        .map(f32::from_bits)
        .map_err(|_| format!("expected float bits, got {field:?}"))
}

fn bits(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err("odd-length hex".to_string());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(|_| "bad hex".to_string()))
        .collect()
}

fn key_name(key: KeyCode) -> String {
    match key {
        // A dash, not a colon: the event token is colon-separated, so
        // `Unknown:42` would split into an empty key name.
        KeyCode::Unknown(code) => format!("Unknown-{code}"),
        named => named_key_name(named).to_string(),
    }
}

fn key_from(name: &str) -> Result<KeyCode, String> {
    match name {
        _ if name.starts_with("Unknown-") => name["Unknown-".len()..]
            .parse()
            .map(KeyCode::Unknown)
            .map_err(|_| format!("bad unknown key {name:?}")),
        _ => KEY_NAMES
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, key)| *key)
            .ok_or_else(|| format!("unknown key {name:?}")),
    }
}

fn button_name(button: MouseButton) -> String {
    match button {
        MouseButton::Left => "Left".to_string(),
        MouseButton::Right => "Right".to_string(),
        MouseButton::Middle => "Middle".to_string(),
        MouseButton::Other(index) => format!("Other-{index}"),
    }
}

fn button_from(name: &str) -> Result<MouseButton, String> {
    match name {
        "Left" => Ok(MouseButton::Left),
        "Right" => Ok(MouseButton::Right),
        "Middle" => Ok(MouseButton::Middle),
        _ => name
            .strip_prefix("Other-")
            .and_then(|index| index.parse().ok())
            .map(MouseButton::Other)
            .ok_or_else(|| format!("unknown mouse button {name:?}")),
    }
}

// One table, so a variant added to `KeyCode` fails to compile here until it is
// named: a key that cannot be captured is a key that does not replay. The
// forward `match` is exhaustive over the named variants; `Unknown` is handled
// by the caller before it.
macro_rules! key_table {
    ($($key:ident),* $(,)?) => {
        const KEY_NAMES: &[(&str, KeyCode)] =
            &[$( (stringify!($key), KeyCode::$key) ),*];

        fn named_key_name(key: KeyCode) -> &'static str {
            match key {
                $( KeyCode::$key => stringify!($key), )*
                KeyCode::Unknown(_) => "Unknown",
            }
        }
    };
}

#[rustfmt::skip]
key_table!(
    Space, Apostrophe, Comma, Minus, Period, Slash, Num0, Num1, Num2, Num3, Num4, Num5, Num6,
    Num7, Num8, Num9, Semicolon, Equal, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S,
    T, U, V, W, X, Y, Z, LeftBracket, Backslash, RightBracket, GraveAccent, Escape, Enter, Tab,
    Backspace, Insert, Delete, Right, Left, Down, Up, PageUp, PageDown, Home, End, CapsLock,
    ScrollLock, NumLock, PrintScreen, Pause, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, Keypad0, Keypad1, Keypad2,
    Keypad3, Keypad4, Keypad5, Keypad6, Keypad7, Keypad8, Keypad9, KeypadDecimal, KeypadDivide,
    KeypadMultiply, KeypadSubtract, KeypadAdd, KeypadEnter, KeypadEqual, LeftShift, LeftControl,
    LeftAlt, LeftSuper, RightShift, RightControl, RightAlt, RightSuper, Menu,
);
