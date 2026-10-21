//! The layout's tests, kept apart from the data it arranges.

use super::*;

#[test]
fn a_default_layout_opens_the_three_working_panels() {
    let layout = Layout::default();
    assert_eq!(layout.panels(Dock::Left), [PanelId::Hierarchy]);
    assert_eq!(layout.panels(Dock::Right), [PanelId::Inspector]);
    assert_eq!(layout.panels(Dock::Bottom), [PanelId::ContentBrowser]);
    assert_eq!(layout.active(Dock::Left), Some(PanelId::Hierarchy));
    assert!(!layout.is_open(PanelId::Profiler));
}

#[test]
fn moving_a_panel_tabs_it_in_the_target_dock() {
    let mut layout = Layout::default();
    layout.place(PanelId::ContentBrowser, Dock::Right);
    assert_eq!(
        layout.panels(Dock::Right),
        [PanelId::Inspector, PanelId::ContentBrowser]
    );
    assert!(layout.panels(Dock::Bottom).is_empty());
    // The moved panel is the tab in front.
    assert_eq!(layout.active(Dock::Right), Some(PanelId::ContentBrowser));
}

#[test]
fn closing_and_reopening_a_panel_restores_it() {
    let mut layout = Layout::default();
    layout.close(PanelId::Inspector);
    assert!(!layout.is_open(PanelId::Inspector));
    assert_eq!(layout.active(Dock::Right), None);

    layout.place(PanelId::Inspector, PanelId::Inspector.default_dock());
    assert_eq!(layout.dock_of(PanelId::Inspector), Some(Dock::Right));
}

#[test]
fn the_active_tab_falls_to_the_first_when_it_closes() {
    let mut layout = Layout::default();
    layout.place(PanelId::Profiler, Dock::Right);
    assert_eq!(layout.active(Dock::Right), Some(PanelId::Profiler));
    layout.close(PanelId::Profiler);
    assert_eq!(layout.active(Dock::Right), Some(PanelId::Inspector));
}

#[test]
fn the_layout_survives_a_round_trip() {
    let mut layout = Layout::default();
    layout.place(PanelId::Profiler, Dock::Bottom);
    layout.left_width = 333.0;
    layout.close(PanelId::ContentBrowser);

    let back = Layout::from_ini(&layout.to_ini());
    assert_eq!(back.panels(Dock::Left), layout.panels(Dock::Left));
    assert_eq!(back.panels(Dock::Right), layout.panels(Dock::Right));
    assert_eq!(back.panels(Dock::Bottom), [PanelId::Profiler]);
    assert_eq!(back.left_width, 333.0);
    assert!(!back.is_open(PanelId::ContentBrowser));
}

#[test]
fn garbage_lines_are_ignored() {
    let layout =
        Layout::from_ini("left= hierarchy, nonsense\n# comment\n= \nleft_width= not a number\n");
    assert_eq!(layout.panels(Dock::Left), [PanelId::Hierarchy]);
    assert_eq!(layout.left_width, 240.0);
}

#[test]
fn a_splitter_drag_stops_at_its_limits() {
    let mut layout = Layout::default();
    assert!(layout.begin_drag(Splitter::Hierarchy, 0.0));
    assert!(layout.drag_to(-10_000.0));
    assert_eq!(layout.left_width, MIN_HIERARCHY);
    assert!(layout.drag_to(10_000.0));
    assert_eq!(layout.left_width, MAX_HIERARCHY);
    assert!(layout.end_drag());
    assert!(!layout.drag_to(500.0));
}

#[test]
fn edits_mark_the_layout_dirty_until_saved() {
    let mut layout = Layout::default();
    assert!(!layout.take_dirty());
    layout.place(PanelId::Profiler, Dock::Right);
    assert!(layout.take_dirty());
    assert!(!layout.take_dirty());
}
