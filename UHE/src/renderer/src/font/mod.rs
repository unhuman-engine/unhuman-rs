//! Fonts.
//!
//! Split by what can fail and what can be tested without a device: [`atlas`] is
//! the file format, [`generate`] samples a face at startup, [`layout`] is pure
//! arithmetic over a loaded atlas, and [`shape`], [`edge_color`], [`distance`],
//! [`raster`] and [`reconstruct`] are the MSDF half that [`generate`] is built
//! from.

pub mod atlas;
pub mod distance;
pub mod edge_color;
pub mod generate;
pub mod layout;
pub mod raster;
pub mod reconstruct;
pub mod shape;

pub use atlas::{Atlas, AtlasError, Glyph, Metrics};
pub use generate::{GenerateError, GeneratorOptions};
pub use layout::{Align, Layout, LayoutOptions, Placement, TextQuad};

/// The face the engine ships, resolved through `vault` rather than the filesystem,
/// so it travels with the asset root like every other asset.
pub const DEFAULT_FONT_PATH: &str = "assets/fonts/Inter.ttf";
