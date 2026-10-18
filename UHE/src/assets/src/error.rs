use std::fmt;

/// Everything an asset operation can fail with.
///
/// Returning a null handle and logging would answer "which asset, and why" only
/// in a log line; a load failure here says what happened.
#[derive(Debug)]
pub enum Error {
    /// No mount root holds the path. Not the same as an I/O failure: the file
    /// may exist somewhere else entirely.
    NotFound(String),
    /// A virtual path with a `..` component that would leave its root,
    /// or an absolute/drive-qualified path. Refused, never resolved: the whole
    /// point of a mount root is that a path cannot name something outside it.
    EscapesRoot(String),
    /// The file is there but could not be read.
    Io(std::io::Error),
    /// A container or payload this build cannot decode. The string says why,
    /// because "unsupported" on its own is a bug report with no information.
    Unsupported(String),
    /// The container parsed and the payload did not.
    Decode(String),
    /// A path could not be watched. Distinct from [`Error::NotFound`] because
    /// watching is a different request: a watch that fails here relies on the
    /// path resolving now and forever, and a later disappearance is reported as
    /// a removal, not this error.
    Watch(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound(path) => write!(f, "asset not found: {path}"),
            Error::EscapesRoot(path) => write!(f, "path escapes its mount root: {path}"),
            Error::Io(error) => write!(f, "io: {error}"),
            Error::Unsupported(why) => write!(f, "unsupported: {why}"),
            Error::Decode(why) => write!(f, "decode failed: {why}"),
            Error::Watch(path) => write!(f, "cannot watch: {path}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
