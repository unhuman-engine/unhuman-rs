//! What this loader does with each glTF extension name.
//!
//! Declaring an extension is not honouring it. A name in the code could mean
//! parsed-into-parameters, recognised-and-refused, or read-and-dropped, and
//! nothing said which. This module is that statement: every name the loader
//! knows is [`Classification::Handled`] or [`Classification::Refused`], and
//! every other name is [`Classification::Ignored`], logged so it is not silent.
//!
//! Draco and meshopt are compression. A file that uses either cannot be read by
//! reading its buffers, so recognising the name and producing empty geometry is
//! the bug this refuses instead.

use crate::error::{Error, Result};
use crate::report::{ExtensionClass, ExtensionUse};

/// How the loader treats one extension name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Classification {
    /// The values reach [`crate::MaterialData`] or the model and are shaded
    /// with.
    Handled,
    /// Needs a decoder this build does not carry. A file that applies or
    /// requires it is refused, naming the extension.
    Refused(&'static str),
    /// Not implemented and not needed for correctness of what this loader
    /// produces; logged so a file using it is not silent.
    Ignored,
}

/// Extensions whose values land on the loaded material or mesh.
const HANDLED: &[&str] = &[
    "KHR_texture_transform",
    "KHR_materials_pbrSpecularGlossiness",
    "KHR_materials_unlit",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_clearcoat",
    "KHR_materials_specular",
    "KHR_materials_sheen",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_iridescence",
    "KHR_materials_anisotropy",
    "KHR_materials_diffuse_transmission",
    "KHR_lights_punctual",
    "KHR_texture_basisu",
];

/// Compression this build has no decoder for, with the reason to report.
const REFUSED: &[(&str, &str)] = &[
    (
        "KHR_draco_mesh_compression",
        "the geometry is Draco-compressed and this crate has no decoder, so the \
         compressed bytes would decode as garbage",
    ),
    (
        "EXT_meshopt_compression",
        "the buffer views are meshopt-compressed and this crate has no decoder, \
         so the compressed bytes would decode as garbage",
    ),
];

/// The classification of one name; anything not listed is [`Ignored`].
pub(crate) fn classify(name: &str) -> Classification {
    if HANDLED.contains(&name) {
        return Classification::Handled;
    }
    if let Some((_, reason)) = REFUSED.iter().find(|(known, _)| *known == name) {
        return Classification::Refused(reason);
    }
    Classification::Ignored
}

/// Every name this loader classifies, for the coverage test.
#[cfg(test)]
pub(crate) fn known_names() -> Vec<&'static str> {
    HANDLED
        .iter()
        .copied()
        .chain(REFUSED.iter().map(|(name, _)| *name))
        .collect()
}

/// Validates the declared extensions and returns how each one is treated.
///
/// An `extensionsRequired` name that is not handled is an error naming it: a
/// file that cannot be correct without an extension this loader does not
/// implement must not load as if it were. The caller owns the reporting; this
/// function only classifies.
pub(crate) fn check(document: &gltf::Document) -> Result<Vec<ExtensionUse>> {
    for name in document.extensions_required() {
        match classify(name) {
            Classification::Handled => {}
            Classification::Refused(reason) => {
                return Err(Error::Unsupported(format!(
                    "required extension '{name}': {reason}"
                )))
            }
            Classification::Ignored => {
                return Err(Error::Unsupported(format!(
                    "required extension '{name}' is not implemented by this loader"
                )))
            }
        }
    }

    Ok(document
        .extensions_used()
        .map(|name| {
            let classification = classify(name);
            ExtensionUse {
                name: name.to_owned(),
                classification: match classification {
                    Classification::Handled => ExtensionClass::Handled,
                    Classification::Refused(_) => ExtensionClass::Refused,
                    Classification::Ignored => ExtensionClass::Ignored,
                },
                reason: match classification {
                    Classification::Refused(reason) => Some(reason.to_owned()),
                    Classification::Ignored => Some("not implemented by this loader".to_owned()),
                    Classification::Handled => None,
                },
            }
        })
        .collect())
}

/// The compression extension actually applied to geometry or a buffer view, if
/// any.
///
/// Declaration alone is not usage: a file may list an extension it does not
/// apply, and refusing that would reject valid geometry. This looks for the
/// extension where it would change how the bytes are read.
pub(crate) fn compression_in_use(document: &gltf::Document) -> Option<&'static str> {
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if primitive
                .extension_value("KHR_draco_mesh_compression")
                .is_some()
            {
                return Some("KHR_draco_mesh_compression");
            }
        }
    }
    if document
        .views()
        .any(|view| view.extension_value("EXT_meshopt_compression").is_some())
    {
        return Some("EXT_meshopt_compression");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_is_refused_and_materials_are_handled() {
        assert!(matches!(
            classify("KHR_draco_mesh_compression"),
            Classification::Refused(_)
        ));
        assert!(matches!(
            classify("EXT_meshopt_compression"),
            Classification::Refused(_)
        ));
        assert_eq!(classify("KHR_materials_clearcoat"), Classification::Handled);
        assert_eq!(classify("KHR_texture_basisu"), Classification::Handled);
        assert_eq!(classify("KHR_materials_variants"), Classification::Ignored);
    }

    #[test]
    fn every_named_extension_is_classified() {
        for name in known_names() {
            assert_ne!(
                classify(name),
                Classification::Ignored,
                "{name} is listed but classifies as ignored"
            );
        }
    }

    /// The sixteen names the loader is expected to know: fourteen that reach
    /// the material, two compression schemes it refuses.
    #[test]
    fn every_listed_extension_is_handled_or_refused() {
        let listed = [
            "KHR_texture_transform",
            "KHR_materials_specular",
            "KHR_materials_emissive_strength",
            "KHR_materials_iridescence",
            "KHR_materials_ior",
            "KHR_materials_sheen",
            "KHR_materials_unlit",
            "KHR_materials_transmission",
            "KHR_materials_clearcoat",
            "KHR_materials_volume",
            "KHR_materials_anisotropy",
            "KHR_materials_diffuse_transmission",
            "KHR_lights_punctual",
            "KHR_texture_basisu",
            "KHR_draco_mesh_compression",
            "EXT_meshopt_compression",
        ];
        for name in listed {
            assert_ne!(
                classify(name),
                Classification::Ignored,
                "{name} is neither handled nor refused"
            );
        }
    }
}
