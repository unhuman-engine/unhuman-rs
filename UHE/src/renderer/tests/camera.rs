//! The editor camera: its two matrices, and its three controls.
//!
//! Every expected number is computed by hand and written out here, not recorded
//! from the code. The projection test asserts through `project_point3`, so it
//! checks the properties that matter (near plane at depth 0, far plane at depth
//! 1, field of view, aspect ratio) instead of a matrix's eleven entries, whose
//! sign convention is the library's business.

use bifrost::camera::{Camera, CameraInput};
use bifrost::EditorCamera;
use glam::{Mat4, Vec3};

/// The default viewport is 1280x720 and a default camera takes 45 degrees.
fn camera() -> EditorCamera {
    EditorCamera::new(45.0, 16.0 / 9.0, 0.1, 1000.0)
}

fn assert_near(actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).length() < 1e-5,
        "expected {expected:?}, got {actual:?}"
    );
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}

/// Nothing has moved it: it orbits from 10 units up +Z looking down -Z, which is
/// what distance 10 and a zero yaw/pitch mean.
#[test]
fn a_fresh_camera_sits_on_plus_z_at_the_default_distance() {
    let camera = camera();

    assert_eq!(camera.distance(), 10.0);
    assert_near(camera.position(), Vec3::new(0.0, 0.0, 10.0));
    assert_near(camera.focal_point(), Vec3::ZERO);
    assert_near(camera.forward_direction(), Vec3::new(0.0, 0.0, -1.0));
    assert_near(camera.up_direction(), Vec3::Y);
    assert_near(camera.right_direction(), Vec3::X);

    // The view matrix is `inverse(translate(position) * toMat4(orientation))`,
    // so with an identity orientation its translation is `-position`.
    assert_near(
        camera.view_matrix().w_axis.truncate(),
        Vec3::new(0.0, 0.0, -10.0),
    );
}

/// The projection is Vulkan's: right-handed, depth 0..1.
#[test]
fn the_projection_is_a_vulkan_perspective() {
    let camera = camera();
    let projection = camera.projection();

    // 1/tan(45/2) = 2.414213562373095, and f/aspect with aspect 1280/720.
    assert_close(projection.col(0).x, 1.357_995_2);
    assert_close(projection.col(1).y, 2.414_213_7);

    // The near plane lands on depth 0 and the far plane on depth 1. This is the
    // property a GL-style projection would fail, and it is what every depth
    // comparison in the renderer assumes.
    let near = projection.project_point3(Vec3::new(0.0, 0.0, -0.1));
    let far = projection.project_point3(Vec3::new(0.0, 0.0, -1000.0));
    assert_close(near.z, 0.0);
    assert_close(far.z, 1.0);

    // The vertical half-angle: a point at `near * tan(fov/2)` is exactly the top
    // edge, and `aspect` times that is exactly the right edge.
    let half = 0.1 * (45.0f32 / 2.0).to_radians().tan();
    assert_close(projection.project_point3(Vec3::new(0.0, half, -0.1)).y, 1.0);
    assert_close(
        projection
            .project_point3(Vec3::new(half * (1280.0 / 720.0), 0.0, -0.1))
            .x,
        1.0,
    );
}

/// `set_viewport_size` recalculates the aspect ratio and therefore the
/// projection: a square viewport halves neither term.
#[test]
fn a_square_viewport_gives_a_square_projection() {
    let mut camera = camera();
    camera.set_viewport_size(1000.0, 1000.0);

    assert_close(camera.aspect_ratio(), 1.0);
    assert_close(camera.projection().col(0).x, 2.414_213_7);
    assert_close(camera.projection().col(1).y, 2.414_213_7);
}

/// Alt + left-drag orbits. `delta = (cursor - initial) * 0.003`, the two speeds
/// are `RotationSpeed() = 0.8`, so a 100x50 pixel drag adds 0.24 rad of yaw and
/// 0.12 rad of pitch.
#[test]
fn an_alt_left_drag_orbits_around_the_focal_point() {
    let mut camera = camera();
    let mut input = CameraInput {
        alt: true,
        left: true,
        ..CameraInput::default()
    };

    // The first frame only records where the drag started.
    camera.update(&input);
    assert_near(camera.position(), Vec3::new(0.0, 0.0, 10.0));

    input.cursor = (100.0, 50.0);
    camera.update(&input);

    assert_close(camera.yaw(), 0.24);
    assert_close(camera.pitch(), 0.12);

    // orientation = Ry(-yaw) * Rx(-pitch), so forward is
    // (0.2359932, -0.1197122, -0.9643527) and the camera is 10 units behind it.
    assert_near(
        camera.position(),
        Vec3::new(-2.3599322, 1.1971221, 9.643_527),
    );
    assert_near(camera.focal_point(), Vec3::ZERO);
}

/// Alt + middle-drag pans the focal point in the camera's own axes. The pan
/// speed for a 1280x720 viewport is (0.13448144, 0.19305744), and a 10x-20 pixel
/// drag is `delta = (0.03, -0.06)`.
#[test]
fn an_alt_middle_drag_pans_the_focal_point() {
    let mut camera = camera();
    let mut input = CameraInput {
        alt: true,
        middle: true,
        ..CameraInput::default()
    };
    camera.update(&input);

    input.cursor = (10.0, -20.0);
    camera.update(&input);

    // focal -= right * 0.03 * 0.13448144 * 10; focal += up * -0.06 * 0.19305744 * 10
    let expected = Vec3::new(-0.040344432, -0.115_834_47, 0.0);
    assert_near(camera.focal_point(), expected);
    // The camera keeps its 10-unit standoff along -Z.
    assert_near(camera.position(), expected + Vec3::new(0.0, 0.0, 10.0));
}

/// Alt + right-drag dollies: `ZoomSpeed()` is `min((distance * 0.2)^2, 100)`, so
/// at distance 10 it is 4 and a tenth of a scroll unit moves 0.4.
#[test]
fn scrolling_dollies_the_camera() {
    let mut camera = camera();
    camera.on_scroll(1.0);

    assert_close(camera.distance(), 9.6);
    assert_near(camera.position(), Vec3::new(0.0, 0.0, 9.6));
}

/// The dolly clamp: below distance 1 the camera steps one unit towards the focal
/// point instead of passing through it.
#[test]
fn the_dolly_stops_at_distance_one() {
    let mut camera = camera();
    camera.set_distance(1.0);
    camera.on_scroll(1.0);

    assert_eq!(camera.distance(), 1.0);
    // focal += forward = (0, 0, -1), and position = focal - forward * distance.
    assert_near(camera.focal_point(), Vec3::new(0.0, 0.0, -1.0));
    assert_near(camera.position(), Vec3::ZERO);
}

/// The base `Camera` is a projection matrix; a view-projection is that matrix
/// times the inverse of the transform.
#[test]
fn a_scene_camera_uses_the_inverse_of_its_transform() {
    let projection = glam::camera::rh::proj::directx::perspective(1.0, 1.0, 0.1, 100.0);
    let camera = Camera::from_projection(projection);
    let transform = Mat4::from_translation(Vec3::new(0.0, 0.0, 5.0));

    assert_eq!(camera.projection(), projection);
    assert_eq!(
        camera.view_projection(&transform),
        projection * transform.inverse()
    );
}

/// An editor camera is a `Camera`: `From<EditorCamera>` is how the frame's
/// `ViewSource::Editor` path reaches the same type the scene path uses.
#[test]
fn an_editor_camera_converts_to_a_camera() {
    let editor = camera();
    let camera = Camera::from(editor);

    assert_eq!(camera.projection(), editor.projection());
}

/// The plugin's default view is the ground from above.
///
/// The grid lies at y = 0. A camera below it draws the grid's underside, which
/// is the same picture mirrored: the axes still cross at the centre, so the only
/// way to tell is which side of the plane the eye is on, and which way it looks.
#[test]
fn the_plugins_default_camera_looks_down_at_the_grid_from_above() {
    let plugin = bifrost::BifrostPlugin::new();
    let camera = plugin.camera();

    assert!(
        camera.position().y > 0.0,
        "the eye is below the grid at y = {}",
        camera.position().y
    );
    assert!(
        camera.forward_direction().y < 0.0,
        "the camera looks up at the grid, forward.y = {}",
        camera.forward_direction().y
    );
}
