//! The editor pass: the grid, then the gizmo lines, in one pass.
//!
//! Both draw into the scene colour with blending and against the scene depth the
//! pass loads, so they share one attachment list. Two passes writing the same
//! colour target would be a write-after-write the graph does not order: edges come
//! from declared *reads* (`compiler::topological_order`), and a `LoadOp::Load` is
//! not a read in that model, so two loading writers of one texture have no edge
//! between them at all.

use chassis::{CommandList, PassScope};

use super::lines::{GridPass, LinePass};

pub struct EditorPass {
    pub grid: Option<GridPass>,
    pub lines: Option<LinePass>,
    /// The physics debug view; drawn before the gizmo so the editor's handles
    /// stay on top of the simulation's wireframes.
    pub debug: Option<LinePass>,
}

impl EditorPass {
    pub fn new(grid: Option<GridPass>, lines: Option<LinePass>, debug: Option<LinePass>) -> Self {
        Self { grid, lines, debug }
    }

    pub fn is_empty(&self) -> bool {
        self.grid.is_none() && self.lines.is_none() && self.debug.is_none()
    }

    pub fn record(
        &self,
        scope: &PassScope<'_>,
        list: &mut dyn CommandList,
    ) -> Result<(), chassis::Error> {
        if let Some(grid) = &self.grid {
            grid.record(scope, list)?;
        }
        if let Some(debug) = &self.debug {
            debug.record(scope, list)?;
        }
        if let Some(lines) = &self.lines {
            lines.record(scope, list)?;
        }
        Ok(())
    }
}
