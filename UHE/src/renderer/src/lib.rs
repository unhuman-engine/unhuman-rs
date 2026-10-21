//! The renderer frontend. Passes, cameras, materials, meshes, text.
//!
//! Deliberately not in `Stormbreaker/`: a backend that knows how to draw is a
//! backend that cannot be swapped for one that draws differently. This crate
//! speaks `chassis` and never names a driver, so both backends run the same
//! renderer and the render graph stays the only owner of synchronization.
//!
//! Ownership: `Bifrost` holds every GPU resource it creates (meshes, materials,
//! pipelines, the font atlas) and drops them through the RHI. The plugin owns
//! `Bifrost`; `adam::Engine` never sees it, which keeps the engine free of a
//! dependency on a renderer.

pub mod animation;
pub mod batch2d;
pub mod camera;
pub mod font;
mod frame_ring;
pub mod ik;
pub mod light;
pub mod material;
pub mod mesh;
pub mod passes;
pub mod plugin;
pub mod renderer;
pub mod text;

pub use camera::EditorCamera;
pub use plugin::BifrostPlugin;
pub use renderer::{Bifrost, FrameImage, ViewSource};
