//! The content browser's model: which directory is open, what is in it, and
//! what a drag out of it carries. The grid, the icons and the drag preview are
//! the toolkit's; this is the state and the filesystem walk behind them.

use std::path::{Component, Path, PathBuf};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Entry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub extension: String,
}

impl Entry {
    pub fn is_image(&self) -> bool {
        matches!(self.extension.as_str(), "png" | "jpg" | "jpeg")
    }

    pub fn is_model(&self) -> bool {
        matches!(self.extension.as_str(), "gltf" | "glb" | "fbx" | "obj")
    }
}

/// What a drag out of the browser carries: the asset's virtual path, which is
/// what `vault::Assets::load` takes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DragPayload {
    pub path: String,
}

pub struct ContentBrowser {
    /// The primary mount, kept for callers that only ever have one.
    pub root: PathBuf,
    pub current: PathBuf,
    pub selected: Option<String>,
    roots: Vec<PathBuf>,
    /// Extension filter; empty shows every file.
    filter: String,
    /// The last directory listing. Kept, because a panel redraws every frame and
    /// a `read_dir` per frame is filesystem work the user did not ask for.
    /// [`ContentBrowser::refresh`] rebuilds it.
    entries: Vec<Entry>,
}

impl Default for ContentBrowser {
    fn default() -> Self {
        Self::new(PathBuf::from("assets"))
    }
}

impl ContentBrowser {
    pub fn new(root: PathBuf) -> Self {
        let mut browser = Self {
            current: root.clone(),
            roots: vec![root.clone()],
            root,
            filter: String::new(),
            selected: None,
            entries: Vec::new(),
        };
        browser.refresh();
        browser
    }

    /// Adds a directory to the search path. An already-mounted directory is not
    /// mounted twice, and a path that is not a directory is ignored.
    pub fn mount(&mut self, dir: PathBuf) {
        if dir.is_dir() && !self.roots.contains(&dir) {
            self.roots.push(dir);
        }
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    pub fn filter(&self) -> &str {
        &self.filter
    }

    /// Sets the extension filter and rebuilds the listing.
    pub fn set_filter(&mut self, filter: impl Into<String>) {
        self.filter = filter.into();
        self.refresh();
    }

    /// The cached contents of the current directory: directories first, then
    /// files, both by name. A caller that changed the filesystem calls
    /// [`ContentBrowser::refresh`].
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Re-reads the current directory into the cache.
    ///
    /// An unreadable directory is empty rather than fatal.
    pub fn refresh(&mut self) {
        let Ok(read) = std::fs::read_dir(&self.current) else {
            self.entries.clear();
            return;
        };
        let mut entries: Vec<Entry> = read
            .filter_map(|item| item.ok())
            .filter_map(|item| {
                let path = item.path();
                let is_dir = item.file_type().map(|t| t.is_dir()).unwrap_or(false);
                let name = item.file_name().to_string_lossy().into_owned();
                let extension = if is_dir {
                    String::new()
                } else {
                    path.extension()
                        .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
                        .unwrap_or_default()
                };
                if !is_dir && !self.filter.is_empty() && extension != self.filter {
                    return None;
                }
                Some(Entry {
                    name,
                    path: path.to_string_lossy().into_owned(),
                    is_dir,
                    extension,
                })
            })
            .collect();
        entries.sort_by(|a, b| {
            b.is_dir.cmp(&a.is_dir).then_with(|| {
                a.name
                    .to_ascii_lowercase()
                    .cmp(&b.name.to_ascii_lowercase())
            })
        });
        self.entries = entries;
    }

    /// From the mounted root down to the current directory, inclusive.
    pub fn breadcrumbs(&self) -> Vec<(String, PathBuf)> {
        let mut crumbs = Vec::new();
        let root = self.containing_root();
        let mut path = self.current.as_path();
        loop {
            let name = match path.file_name() {
                Some(name) => name.to_string_lossy().into_owned(),
                None => path.to_string_lossy().into_owned(),
            };
            crumbs.push((name, path.to_path_buf()));
            if path == root {
                break;
            }
            match path.parent() {
                Some(parent) => path = parent,
                None => break,
            }
        }
        crumbs.reverse();
        crumbs
    }

    /// Only descends into a mounted root, never anywhere the process can read.
    pub fn navigate(&mut self, dir: &Path) -> bool {
        if !dir.is_dir() || !self.roots.iter().any(|root| dir.starts_with(root)) {
            return false;
        }
        self.current = dir.to_path_buf();
        self.refresh();
        true
    }

    fn containing_root(&self) -> &Path {
        self.roots
            .iter()
            .filter(|root| self.current.starts_with(root))
            .max_by_key(|root| root.components().count())
            .map(|root| root.as_path())
            .unwrap_or(self.current.as_path())
    }

    pub fn up(&mut self) -> bool {
        let root = self.containing_root().to_path_buf();
        if self.current == root {
            return false;
        }
        match self.current.parent() {
            Some(parent) if parent.starts_with(&root) => {
                self.current = parent.to_path_buf();
                self.refresh();
                true
            }
            _ => false,
        }
    }

    /// The virtual path a loader wants: relative to the mounted root, with
    /// forward slashes.
    pub fn vault_path(&self, path: &Path) -> String {
        let root = self
            .roots
            .iter()
            .filter(|root| path.starts_with(root))
            .max_by_key(|root| root.components().count());
        let relative = match root {
            Some(root) => path.strip_prefix(root).unwrap_or(path),
            None => path,
        };
        relative
            .components()
            .map(|part| match part {
                // The root component *is* the leading separator, so joining it
                // as text would double the first slash.
                Component::RootDir => String::new(),
                other => other.as_os_str().to_string_lossy().into_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    pub fn path_string(&self, entry: &Entry) -> String {
        self.vault_path(Path::new(&entry.path))
    }

    pub fn payload(&self, entry: &Entry) -> DragPayload {
        DragPayload {
            path: self.path_string(entry),
        }
    }

    /// Accepts a drag by name. The payload is a root-relative virtual path, so
    /// it resolves against the mounted root, not the open directory; an empty
    /// payload is refused.
    pub fn accept(&mut self, payload: &DragPayload) -> Option<PathBuf> {
        if payload.path.is_empty() {
            return None;
        }
        self.selected = Some(payload.path.clone());
        Some(self.root.join(Path::new(&payload.path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_tree(name: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("mirror_content_browser_{name}"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("textures")).unwrap();
        std::fs::create_dir_all(base.join("scenes")).unwrap();
        std::fs::write(base.join("textures/rock.png"), b"png").unwrap();
        std::fs::write(base.join("scenes/level.unhuman"), b"yaml").unwrap();
        std::fs::write(base.join("readme.txt"), b"txt").unwrap();
        base
    }

    #[test]
    fn lists_directories_before_files() {
        let base = temp_tree("list");
        let browser = ContentBrowser::new(base.clone());
        let entries = browser.entries();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["scenes", "textures", "readme.txt"]);

        assert!(entries[0].is_dir);
        let readme = entries.iter().find(|e| e.name == "readme.txt").unwrap();
        assert_eq!(readme.extension, "txt");
        assert!(!readme.is_dir);
    }

    #[test]
    fn the_listing_is_cached_until_refreshed() {
        let base = temp_tree("cache");
        let mut browser = ContentBrowser::new(base.clone());
        let before = browser.entries().len();
        std::fs::write(base.join("new.txt"), b"new").unwrap();

        // Reading again does not walk the disk, so the new file is not there.
        assert_eq!(
            browser.entries().len(),
            before,
            "a read must not hit the disk"
        );
        browser.refresh();
        assert_eq!(browser.entries().len(), before + 1);
        assert!(browser
            .entries()
            .iter()
            .any(|entry| entry.name == "new.txt"));
    }

    #[test]
    fn filters_files_by_extension_and_keeps_directories() {
        let base = temp_tree("filter");
        let mut browser = ContentBrowser::new(base.clone());
        browser.set_filter("unhuman");
        let names: Vec<String> = browser.entries().iter().map(|e| e.name.clone()).collect();
        // Directories are kept whatever the filter; `readme.txt` is filtered out,
        // and the only `.unhuman` file lives under `scenes/`, seen on navigate.
        assert_eq!(names, vec!["scenes", "textures"]);

        browser.navigate(&base.join("scenes"));
        assert_eq!(browser.entries().len(), 1);
    }

    #[test]
    fn breadcrumbs_run_from_the_root_to_the_current_directory() {
        let base = temp_tree("crumbs");
        let mut browser = ContentBrowser::new(base.clone());
        assert!(browser.navigate(&base.join("textures")));
        let crumbs = browser.breadcrumbs();
        let names: Vec<&str> = crumbs.iter().map(|(name, _)| name.as_str()).collect();
        let expected_root = base.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(names, vec![expected_root.as_str(), "textures"]);
        assert_eq!(crumbs.last().unwrap().1, base.join("textures"));
    }

    #[test]
    fn navigate_refuses_anything_outside_a_root() {
        let base = temp_tree("escape");
        let mut browser = ContentBrowser::new(base.clone());
        assert!(!browser.navigate(base.parent().unwrap()));
        assert_eq!(browser.current, base);
    }

    #[test]
    fn up_stops_at_the_root() {
        let base = temp_tree("up");
        let mut browser = ContentBrowser::new(base.clone());
        assert!(browser.navigate(&base.join("textures")));
        assert!(browser.up());
        assert_eq!(browser.current, base);
        assert!(!browser.up());
        assert_eq!(browser.current, base);
    }

    #[test]
    fn payloads_carry_the_virtual_path() {
        let base = temp_tree("payload");
        let mut browser = ContentBrowser::new(base.clone());
        assert!(browser.navigate(&base.join("textures")));
        let entry = browser
            .entries()
            .iter()
            .find(|e| e.name == "rock.png")
            .unwrap()
            .clone();
        assert_eq!(browser.path_string(&entry), "textures/rock.png");
        assert_eq!(browser.payload(&entry).path, "textures/rock.png");
        assert!(entry.is_image());
        assert_eq!(
            browser.accept(&browser.payload(&entry)),
            Some(base.join("textures/rock.png"))
        );
        assert_eq!(browser.selected.as_deref(), Some("textures/rock.png"));
    }

    #[test]
    fn a_second_mount_is_visible_in_roots() {
        let base = temp_tree("mount");
        let other = std::env::temp_dir().join("mirror_content_browser_mount_other");
        std::fs::create_dir_all(&other).unwrap();
        let mut browser = ContentBrowser::new(base.clone());
        browser.mount(other.clone());
        browser.mount(other.clone());
        assert_eq!(browser.roots().len(), 2);
        assert_eq!(browser.roots()[1], other);
        assert!(browser.navigate(&other));
        assert_eq!(
            browser.vault_path(Path::new("/nowhere/at/all.txt")),
            "/nowhere/at/all.txt"
        );
    }

    #[test]
    fn a_missing_directory_is_empty() {
        let browser = ContentBrowser::new(PathBuf::from("/nonexistent/mirror/dir"));
        assert!(browser.entries().is_empty());
        assert_eq!(browser.breadcrumbs().len(), 1);
    }
}
