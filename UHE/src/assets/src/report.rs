//! What one load read, skipped and substituted.
//!
//! A load that returns only its payload cannot answer "why does this model
//! look wrong". The report is that answer: counts for each kind of thing the
//! file declared, every extension and how it was treated, and a reason for
//! each substitution or warning.

/// How one declared glTF extension was treated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExtensionClass {
    /// Parsed into engine parameters and shaded with.
    Handled,
    /// Recognised, but this build cannot decode it.
    Refused,
    /// Not implemented and not needed for what the loader produces.
    Ignored,
}

/// One extension the file declared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionUse {
    pub name: String,
    pub classification: ExtensionClass,
    /// Why it was refused or ignored, and `None` when handled.
    pub reason: Option<String>,
}

/// Per-kind counts of what a model load read and what it dropped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadCounts {
    pub primitives: usize,
    pub meshes: usize,
    pub textures: usize,
    pub materials: usize,
}

/// One place the loader replaced something so the load could continue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Substitution {
    /// What was replaced, such as `texture (Base)` or `material (Tri)`.
    pub subject: String,
    pub reason: String,
}

/// The report attached to one loaded asset.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// The virtual path the load was asked for.
    pub source: String,
    /// Lower-case file extension, without the dot; empty when there is none.
    pub format: String,
    pub extensions: Vec<ExtensionUse>,
    pub loaded: LoadCounts,
    pub skipped: LoadCounts,
    pub substitutions: Vec<Substitution>,
    pub warnings: Vec<String>,
}

impl LoadReport {
    pub(crate) fn new(source: &str, format: String) -> Self {
        Self {
            source: source.to_owned(),
            format,
            ..Self::default()
        }
    }

    pub(crate) fn substitution(&mut self, subject: impl Into<String>, reason: impl Into<String>) {
        self.substitutions.push(Substitution {
            subject: subject.into(),
            reason: reason.into(),
        });
    }

    pub(crate) fn warning(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    /// True when nothing was skipped, substituted or warned about.
    pub fn is_clean(&self) -> bool {
        self.skipped.primitives == 0
            && self.skipped.meshes == 0
            && self.skipped.textures == 0
            && self.skipped.materials == 0
            && self.substitutions.is_empty()
            && self.warnings.is_empty()
    }

    /// One line per asset: a clean report is info, anything else is a warning.
    pub(crate) fn log(&self) {
        let summary = self.summary();
        if self.is_clean() {
            log::info!("{summary}");
        } else {
            log::warn!("{summary}");
        }
    }

    fn summary(&self) -> String {
        let (mut handled, mut refused, mut ignored) = (0, 0, 0);
        for extension in &self.extensions {
            match extension.classification {
                ExtensionClass::Handled => handled += 1,
                ExtensionClass::Refused => refused += 1,
                ExtensionClass::Ignored => ignored += 1,
            }
        }

        format!(
            "loaded {} [{}]: {}/{} primitives, {}/{} meshes, {}/{} textures, \
             {}/{} materials; {} substitution(s), {} warning(s); \
             extensions {handled} handled, {refused} refused, {ignored} ignored",
            self.source,
            self.format,
            self.loaded.primitives,
            self.loaded.primitives + self.skipped.primitives,
            self.loaded.meshes,
            self.loaded.meshes + self.skipped.meshes,
            self.loaded.textures,
            self.loaded.textures + self.skipped.textures,
            self.loaded.materials,
            self.loaded.materials + self.skipped.materials,
            self.substitutions.len(),
            self.warnings.len(),
        )
    }
}
