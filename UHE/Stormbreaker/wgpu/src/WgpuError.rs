//! Error type.
//!
//! The twin of `stormbreaker`'s `VulkanError.rs`: `chassis::Error` says what a
//! frontend can recover from, this says what wgpu reported before it became that.
//!
//! ## What it covers
//!
//! Surface and adapter negotiation. Resource creation reports a bad argument
//! through the device's uncaptured-error handler, which panics with no scope
//! installed; `in_validation_scope` in `WgpuResources` turns those into
//! `chassis::Error::InvalidDescriptor`. A `WgpuError` therefore means a call
//! returned, never that wgpu was happy.

use std::fmt;

use chassis::Error as RhiError;

/// Errors raised by the backend itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuError {
    /// No adapter was found, or none that can present to the surface.
    ///
    /// One variant for both because the surface is requested *with* the adapter:
    /// an adapter that cannot present to it is not one this backend can use.
    NoAdapter,
    /// An adapter was found and `request_device` refused it.
    NoDevice,
    /// The surface offers no format for this adapter, so nothing can be configured.
    IncompatibleSurface,
    /// A caller asked for something the surface does not offer.
    ///
    /// Carries what was asked for and what was available: a surface that cannot
    /// give a format is nearly always a caller asking for one display's format on
    /// another, and the list is what says so.
    Unsupported { what: String, available: String },
    /// There is no surface to present to.
    NoSwapchain,
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgpuError::NoAdapter => write!(f, "no wgpu adapter can present to this surface"),
            WgpuError::NoDevice => write!(f, "wgpu refused to create a device on this adapter"),
            WgpuError::IncompatibleSurface => {
                write!(f, "the surface offers no format for this adapter")
            }
            WgpuError::Unsupported { what, available } => {
                write!(
                    f,
                    "the surface does not offer {what}; it offers {available}"
                )
            }
            WgpuError::NoSwapchain => write!(f, "no surface is attached to this gpu"),
        }
    }
}

impl std::error::Error for WgpuError {}

impl From<WgpuError> for RhiError {
    fn from(error: WgpuError) -> Self {
        // `Unsupported`, not a variant per case: the RHI's error set is small on
        // purpose, and "this device cannot do that" is the honest shape for all of
        // these. The same conversion the Vulkan backend makes, so a caller above
        // the RHI cannot tell the two backends apart by their errors either.
        RhiError::Unsupported(error.to_string())
    }
}

pub type Result<T, E = WgpuError> = std::result::Result<T, E>;
