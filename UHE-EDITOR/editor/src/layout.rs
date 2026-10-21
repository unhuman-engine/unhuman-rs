//! The editor's dock layout and its `imgui.ini` equivalent.
//!
//! Pure data: which panel sits in which dock, which tab of a dock is in front,
//! the splitter sizes, and a drag in progress. No gpui type appears here, so the
//! layout can be parsed, rearranged and saved without a window, and the file it
//! writes (see [`ini`]) is readable and hand-editable.

/// A panel the user can open, close and dock.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PanelId {
    Hierarchy,
    Inspector,
    ContentBrowser,
    Console,
    Profiler,
    Shortcuts,
}

impl PanelId {
    pub const ALL: [PanelId; 6] = [
        PanelId::Hierarchy,
        PanelId::Inspector,
        PanelId::ContentBrowser,
        PanelId::Console,
        PanelId::Profiler,
        PanelId::Shortcuts,
    ];

    /// The panel's title, on its tab and in the View menu.
    pub fn title(self) -> &'static str {
        match self {
            PanelId::Hierarchy => "Hierarchy",
            PanelId::Inspector => "Inspector",
            PanelId::ContentBrowser => "Content Browser",
            PanelId::Console => "Console",
            PanelId::Profiler => "Profiler",
            PanelId::Shortcuts => "Shortcuts",
        }
    }

    /// The stable id written to disk, independent of the title's wording.
    fn key(self) -> &'static str {
        match self {
            PanelId::Hierarchy => "hierarchy",
            PanelId::Inspector => "inspector",
            PanelId::ContentBrowser => "content_browser",
            PanelId::Console => "console",
            PanelId::Profiler => "profiler",
            PanelId::Shortcuts => "shortcuts",
        }
    }

    fn from_key(key: &str) -> Option<Self> {
        PanelId::ALL.into_iter().find(|panel| panel.key() == key)
    }

    /// Where the panel lands when it is opened or its dock is forgotten.
    pub fn default_dock(self) -> Dock {
        match self {
            PanelId::Hierarchy => Dock::Left,
            PanelId::ContentBrowser | PanelId::Console => Dock::Bottom,
            PanelId::Inspector | PanelId::Profiler | PanelId::Shortcuts => Dock::Right,
        }
    }
}

/// One edge of the window a panel can be docked to. The center is the viewport,
/// which never moves; the bottom dock lines up with the content browser.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dock {
    Left,
    Right,
    Bottom,
}

impl Dock {
    pub const ALL: [Dock; 3] = [Dock::Left, Dock::Right, Dock::Bottom];

    pub fn title(self) -> &'static str {
        match self {
            Dock::Left => "Left",
            Dock::Right => "Right",
            Dock::Bottom => "Bottom",
        }
    }

    fn index(self) -> usize {
        match self {
            Dock::Left => 0,
            Dock::Right => 1,
            Dock::Bottom => 2,
        }
    }

    fn key(self) -> &'static str {
        match self {
            Dock::Left => "left",
            Dock::Right => "right",
            Dock::Bottom => "bottom",
        }
    }

    fn from_key(key: &str) -> Option<Self> {
        Dock::ALL.into_iter().find(|dock| dock.key() == key)
    }
}

pub const MIN_HIERARCHY: f32 = 150.0;
pub const MAX_HIERARCHY: f32 = 520.0;
pub const MIN_INSPECTOR: f32 = 190.0;
pub const MAX_INSPECTOR: f32 = 620.0;
pub const MIN_CONTENT: f32 = 70.0;
pub const MAX_CONTENT: f32 = 460.0;

/// A splitter drag in progress, so a move between two frames follows one origin.
#[derive(Clone, Copy)]
struct Drag {
    splitter: Splitter,
    origin: f32,
    start: f32,
}

/// The three draggable splitters. The viewport is what is left over.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Splitter {
    Hierarchy,
    Inspector,
    Content,
}

/// Which coordinate a splitter follows: a width edge moves on x, a height edge
/// on y.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitterAxis {
    Width,
    Height,
}

impl Splitter {
    pub fn id(self) -> &'static str {
        match self {
            Splitter::Hierarchy => "splitter-hierarchy",
            Splitter::Inspector => "splitter-inspector",
            Splitter::Content => "splitter-content",
        }
    }

    pub fn axis(self) -> SplitterAxis {
        match self {
            Splitter::Hierarchy | Splitter::Inspector => SplitterAxis::Width,
            Splitter::Content => SplitterAxis::Height,
        }
    }
}

pub struct Layout {
    docks: [Vec<PanelId>; 3],
    active: [Option<PanelId>; 3],
    pub left_width: f32,
    pub right_width: f32,
    pub bottom_height: f32,
    drag: Option<Drag>,
    dirty: bool,
}

impl Default for Layout {
    fn default() -> Self {
        let mut layout = Self {
            docks: [Vec::new(), Vec::new(), Vec::new()],
            active: [None, None, None],
            left_width: 240.0,
            right_width: 300.0,
            bottom_height: 160.0,
            drag: None,
            dirty: false,
        };
        for panel in [
            PanelId::Hierarchy,
            PanelId::Inspector,
            PanelId::ContentBrowser,
        ] {
            layout.place(panel, panel.default_dock());
        }
        layout.dirty = false;
        layout
    }
}

impl Layout {
    /// The panels in a dock, in tab order.
    pub fn panels(&self, dock: Dock) -> &[PanelId] {
        &self.docks[dock.index()]
    }

    pub fn is_empty(&self, dock: Dock) -> bool {
        self.docks[dock.index()].is_empty()
    }

    /// The tab in front in a dock: the active one, or the first if it left.
    pub fn active(&self, dock: Dock) -> Option<PanelId> {
        let panels = &self.docks[dock.index()];
        self.active[dock.index()]
            .filter(|panel| panels.contains(panel))
            .or_else(|| panels.first().copied())
    }

    pub fn is_open(&self, panel: PanelId) -> bool {
        Dock::ALL
            .into_iter()
            .any(|dock| self.docks[dock.index()].contains(&panel))
    }

    pub fn dock_of(&self, panel: PanelId) -> Option<Dock> {
        Dock::ALL
            .into_iter()
            .find(|dock| self.docks[dock.index()].contains(&panel))
    }

    /// Brings a panel's tab to the front of the dock it already lives in.
    pub fn activate(&mut self, panel: PanelId) {
        if let Some(dock) = self.dock_of(panel) {
            self.set_active(dock, panel);
        }
    }

    /// Moves a panel to a dock, tabbed behind whatever is already there.
    pub fn place(&mut self, panel: PanelId, dock: Dock) {
        self.detach(panel);
        let panels = &mut self.docks[dock.index()];
        if !panels.contains(&panel) {
            panels.push(panel);
        }
        self.set_active(dock, panel);
        self.dirty = true;
    }

    pub fn close(&mut self, panel: PanelId) {
        if self.detach(panel) {
            self.dirty = true;
        }
    }

    /// Opens a panel in its default dock, or closes it if it is already open.
    pub fn toggle(&mut self, panel: PanelId) {
        match self.dock_of(panel) {
            Some(_) => self.close(panel),
            None => self.place(panel, panel.default_dock()),
        }
    }

    fn detach(&mut self, panel: PanelId) -> bool {
        for dock in Dock::ALL {
            let panels = &mut self.docks[dock.index()];
            if let Some(index) = panels.iter().position(|entry| *entry == panel) {
                panels.remove(index);
                if self.active[dock.index()] == Some(panel) {
                    self.active[dock.index()] = panels.first().copied();
                }
                return true;
            }
        }
        false
    }

    fn set_active(&mut self, dock: Dock, panel: PanelId) {
        self.active[dock.index()] = Some(panel);
    }

    pub fn begin_drag(&mut self, splitter: Splitter, position: f32) -> bool {
        if self.drag.is_some() {
            return false;
        }
        self.drag = Some(Drag {
            splitter,
            origin: position,
            start: self.value(splitter),
        });
        true
    }

    /// Moves the dragged edge to follow the pointer. Returns whether the layout
    /// actually changed, so a frame that moves nothing does not redraw.
    pub fn drag_to(&mut self, position: f32) -> bool {
        let Some(drag) = self.drag else {
            return false;
        };

        // The inspector's edge moves left as its width grows, and the content
        // browser's moves up as its height grows, so theirs are inverted.
        let delta = position - drag.origin;
        let moved = match drag.splitter {
            Splitter::Hierarchy => (drag.start + delta).clamp(MIN_HIERARCHY, MAX_HIERARCHY),
            Splitter::Inspector => (drag.start - delta).clamp(MIN_INSPECTOR, MAX_INSPECTOR),
            Splitter::Content => (drag.start - delta).clamp(MIN_CONTENT, MAX_CONTENT),
        };

        let slot = match drag.splitter {
            Splitter::Hierarchy => &mut self.left_width,
            Splitter::Inspector => &mut self.right_width,
            Splitter::Content => &mut self.bottom_height,
        };
        if *slot == moved {
            return false;
        }
        *slot = moved;
        self.dirty = true;
        true
    }

    pub fn end_drag(&mut self) -> bool {
        self.drag.take().is_some()
    }

    pub fn is_dragging(&self, splitter: Splitter) -> bool {
        self.drag.is_some_and(|drag| drag.splitter == splitter)
    }

    fn value(&self, splitter: Splitter) -> f32 {
        match splitter {
            Splitter::Hierarchy => self.left_width,
            Splitter::Inspector => self.right_width,
            Splitter::Content => self.bottom_height,
        }
    }

    /// Whether the layout changed since the last save.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }
}

mod ini;
pub(crate) use ini::config_path;

#[cfg(test)]
mod tests;
