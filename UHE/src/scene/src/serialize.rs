//! The `.unhuman` serializer: the scene document, read and written.
//!
//! # One format, and what it is
//!
//! A **save is** `.unhuman` YAML: the authoritative, diffable, mergeable scene
//! document, and the only form this crate writes.
//!
//! An **asset is** a content-addressed payload in `vault` (a mesh, an image, a
//! clip). A `.unhuman` file is also an asset to `vault`, as `AssetKind::Text`;
//! loading it as a scene means handing its bytes to [`from_yaml`].
//!
//! The binary `postcard` document with a schema hash and its TOML export
//! described in `docs/TRANSFER.md` is **not implemented**. If it is built, it
//! is a derived import/cook path for shipped scenes, never a second save
//! format: it must be produced from this document and its loader must call
//! [`from_yaml`]/[`to_yaml`], so there is one schema and one migration path.
//! That is the seam; it needs no `assets` change until such a cooked kind
//! exists.
//!
//! Field names are the established ones, so the scenes in
//! `assets/scenes/*.unhuman` load unchanged and a file written here keeps that
//! layout. The one addition is a `*Path` string beside every asset handle,
//! because a `vault::AssetId` cannot be rebuilt from a file without the
//! registry that issued it.

mod read;
mod read_physics;
mod value;
mod write;

use std::path::Path;

use yaml_serde::{Mapping, Value};

use crate::entity::Entity;
use crate::scene::Scene;

use self::value::key;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The version this build writes and the newest it can migrate to.
pub const FORMAT_VERSION: u32 = 1;

/// The layout written before a `FormatVersion` key existed. Its keys are the
/// v1 keys, so the 0 -> 1 migration changes no data.
const LEGACY_VERSION: u32 = 0;

/// The document marker written beside the version.
const FORMAT_NAME: &str = "unhuman";

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The bytes are not YAML. `line` is the line the parser stopped on, when it
    /// knew one.
    Yaml {
        line: Option<usize>,
        message: String,
    },
    /// The bytes are YAML but not a scene.
    Format {
        line: Option<usize>,
        message: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(err) => write!(f, "{err}"),
            Error::Yaml { line, message } | Error::Format { line, message } => match line {
                Some(line) => write!(f, "line {line}: {message}"),
                None => write!(f, "{message}"),
            },
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

pub fn save(scene: &Scene, path: &Path) -> Result<()> {
    std::fs::write(path, to_yaml(scene)?)?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Scene> {
    from_yaml(&std::fs::read_to_string(path)?)
}

pub fn to_yaml(scene: &Scene) -> Result<String> {
    let mut root = Mapping::new();
    root.insert(key("Scene"), Value::String(scene.name().to_string()));
    root.insert(key("Format"), Value::String(FORMAT_NAME.to_string()));
    root.insert(key("FormatVersion"), Value::Number(FORMAT_VERSION.into()));
    root.insert(key("Entities"), Value::Sequence(write::tree(scene)));
    yaml_serde::to_string(&Value::Mapping(root)).map_err(yaml_error)
}

pub fn from_yaml(text: &str) -> Result<Scene> {
    let value: Value = yaml_serde::from_str(text).map_err(yaml_error)?;
    let Value::Mapping(mut root) = value else {
        return Err(Error::Format {
            line: None,
            message: "the file is not a mapping".to_string(),
        });
    };

    if !root.contains_key("Scene") {
        return Err(Error::Format {
            line: None,
            message: "no `Scene` key: this is not an .unhuman scene".to_string(),
        });
    }

    if let Some(format) = root.get("Format") {
        if format.as_str() != Some(FORMAT_NAME) {
            return Err(Error::Format {
                line: None,
                message: format!("not an .unhuman document (Format is {format:?})"),
            });
        }
    }

    let version = document_version(&root)?;
    migrate(&mut root, version)?;

    let mut scene = Scene::new();
    scene.set_name(
        root.get("Scene")
            .and_then(Value::as_str)
            .unwrap_or("Untitled"),
    );

    let Some(entities) = root.get("Entities") else {
        return Ok(scene);
    };
    let entities = entities.as_sequence().ok_or_else(|| Error::Format {
        line: None,
        message: "`Entities` is not a sequence".to_string(),
    })?;

    // Pass 1: create everything, restoring serialized ids, so a child can find
    // a parent that appears later in the file.
    let mut created: Vec<Entity> = Vec::with_capacity(entities.len());
    for node in entities {
        let tag = node
            .get("TagComponent")
            .and_then(|tag| tag.get("Tag"))
            .and_then(Value::as_str)
            .unwrap_or("Entity");
        let entity = scene.spawn(tag);
        if let Some(id) = node.get("Entity").and_then(Value::as_u64) {
            if id != 0 {
                if scene.id_is_taken(id) {
                    // Shipped scenes repeat one id on every entity; refusing
                    // the file would make them unloadable. Hierarchy references
                    // still resolve to the first entity with that id.
                    log::warn!("meats: duplicate Entity id {id}; a fresh id was assigned");
                } else {
                    scene.set_id(entity, id);
                }
            }
        }
        created.push(entity);
    }

    // Pass 2: components, then the parent links. `set_parent` refuses a cycle
    // and ignores a dangling parent.
    for (node, entity) in entities.iter().zip(&created) {
        read::components(&mut scene, *entity, node);
        read::note_dropped(&mut scene, *entity, node);
        if let Some(parent) = node
            .get("RelationshipComponent")
            .and_then(|relationship| relationship.get("Parent"))
            .and_then(Value::as_u64)
        {
            if parent != 0 {
                match scene.entity_of_id(parent) {
                    Some(parent) => scene.set_parent(*entity, Some(parent)),
                    None => log::warn!(
                        "meats: {entity:?} names missing parent id {parent}; leaving it a root"
                    ),
                }
            }
        }
    }

    Ok(scene)
}

fn document_version(root: &Mapping) -> Result<u32> {
    match root.get("FormatVersion") {
        None => Ok(LEGACY_VERSION),
        Some(value) => value
            .as_u64()
            .and_then(|version| u32::try_from(version).ok())
            .ok_or_else(|| Error::Format {
                line: None,
                message: format!("FormatVersion is not a version number: {value:?}"),
            }),
    }
}

/// Upgrades `root` to [`FORMAT_VERSION`], one version at a time.
///
/// A document from a newer build is refused rather than half-read, because the
/// fields it added would be silently dropped on the next save.
fn migrate(root: &mut Mapping, from: u32) -> Result<()> {
    let mut version = from;
    while version < FORMAT_VERSION {
        match version {
            // The pre-version layout already uses the v1 key names.
            LEGACY_VERSION => {}
            other => {
                return Err(Error::Format {
                    line: None,
                    message: format!(
                        "no migration from scene format version {other} to {FORMAT_VERSION}"
                    ),
                })
            }
        }
        version += 1;
    }
    if version > FORMAT_VERSION {
        return Err(Error::Format {
            line: None,
            message: format!(
                "scene format version {version} is newer than this build supports \
                 ({FORMAT_VERSION}); open it with a newer editor"
            ),
        });
    }
    root.insert(key("FormatVersion"), Value::Number(FORMAT_VERSION.into()));
    Ok(())
}

fn yaml_error(err: yaml_serde::Error) -> Error {
    Error::Yaml {
        line: err.location().map(|location| location.line()),
        message: err.to_string(),
    }
}
