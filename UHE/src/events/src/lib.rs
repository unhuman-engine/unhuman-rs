//! Events: the bus, the key and mouse codes, the event enum, and ECS observers.
//!
//! Two delivery mechanisms, deliberately not collapsed:
//!
//! * [`EventBus`] broadcasts an [`Event`] to its subscribers the moment it is
//!   pushed, and keeps a copy in a queue for the frame's single consumer.
//! * [`observer`] hooks fire from the scene's structural operations and hand
//!   back the live entity, not a copy.

mod bus;
mod event;
mod key;
pub mod observer;

pub use bus::{EventBus, Sender, Subscription};
pub use event::Event;
pub use key::{KeyCode, MouseButton};
pub use observer::{ObservedScene, Observer, ObserverHandle, Observers};
