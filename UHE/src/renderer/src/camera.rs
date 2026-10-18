//! Cameras: the base scene camera, and the editor's orbit camera.
//!
//! The controls take a [`CameraInput`] snapshot per update rather than reading a
//! global input state: the same numbers, with the ambient access removed.

use glam::{Mat4, Quat, Vec2, Vec3};

/// A projection matrix, and nothing else.
///
/// Lets a scene camera be passed by projection without knowing how it was
/// configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    projection: Mat4,
}

impl Camera {
    pub fn from_projection(projection: Mat4) -> Self {
        Self { projection }
    }

    pub fn projection(&self) -> Mat4 {
        self.projection
    }

    pub fn view_projection(&self, transform: &Mat4) -> Mat4 {
        self.projection * transform.inverse()
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            projection: Mat4::IDENTITY,
        }
    }
}

/// The input the orbit controls read, as a value.
///
/// Four held flags, the cursor, and the frame's scroll. `eyes::Input` is the
/// source (`is_key_held`, `is_mouse_held`, `cursor`, `scroll`), copied in by the
/// binary each frame: the camera must not know how input is collected, and the
/// controls must be drivable from a test with no window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraInput {
    /// Held Alt gates every other control.
    pub alt: bool,
    pub left: bool,
    pub middle: bool,
    pub right: bool,
    /// Absolute cursor position in window coordinates.
    pub cursor: (f32, f32),
    /// Scroll for this frame, `(dx, dy)`.
    pub scroll: (f32, f32),
}

/// Orbit camera: pan, rotate and dolly about a focal point.
///
/// Tuned with the two polynomials in [`EditorCamera::pan_speed`] and a dolly
/// clamp at distance 1.
#[derive(Clone, Copy, Debug)]
pub struct EditorCamera {
    fov: f32,
    aspect_ratio: f32,
    near_clip: f32,
    far_clip: f32,

    projection: Mat4,
    view_matrix: Mat4,
    position: Vec3,
    focal_point: Vec3,
    initial_mouse: Vec2,

    distance: f32,
    pitch: f32,
    yaw: f32,

    viewport_width: f32,
    viewport_height: f32,
}

impl EditorCamera {
    /// Recalculates the projection and the view.
    pub fn new(fov: f32, aspect_ratio: f32, near_clip: f32, far_clip: f32) -> Self {
        let mut camera = Self {
            fov,
            aspect_ratio,
            near_clip,
            far_clip,
            ..Self::default()
        };
        camera.update_projection();
        camera.update_view();
        camera
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance;
    }

    pub fn set_viewport_size(&mut self, width: f32, height: f32) {
        self.viewport_width = width;
        self.viewport_height = height;
        self.update_projection();
    }

    pub fn viewport_size(&self) -> (f32, f32) {
        (self.viewport_width, self.viewport_height)
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.view_matrix
    }

    pub fn projection(&self) -> Mat4 {
        self.projection
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection * self.view_matrix
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn fov(&self) -> f32 {
        self.fov
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }

    pub fn clip_planes(&self) -> (f32, f32) {
        (self.near_clip, self.far_clip)
    }

    pub fn up_direction(&self) -> Vec3 {
        self.orientation() * Vec3::Y
    }

    pub fn right_direction(&self) -> Vec3 {
        self.orientation() * Vec3::X
    }

    pub fn forward_direction(&self) -> Vec3 {
        self.orientation() * Vec3::NEG_Z
    }

    pub fn orientation(&self) -> Quat {
        // Spelled as the yaw-times-pitch product, with `(-pitch, -yaw, 0)`,
        // rather than through an Euler order, so the convention is not left to a
        // library default.
        Quat::from_rotation_y(-self.yaw) * Quat::from_rotation_x(-self.pitch)
    }

    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Sets the elevation, in radians, and refreshes the view.
    ///
    /// The view is derived from pitch, yaw and distance, so a setter that leaves
    /// it alone renders the previous orientation: the eye is `focal - forward *
    /// distance`, and `forward` is the orientation's.
    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch;
        self.update_view();
    }

    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = yaw;
        self.update_view();
    }

    /// The focal point the camera orbits.
    pub fn focal_point(&self) -> Vec3 {
        self.focal_point
    }

    pub fn set_focal_point(&mut self, focal_point: Vec3) {
        self.focal_point = focal_point;
    }

    pub fn update(&mut self, input: &CameraInput) {
        if input.alt {
            let mouse = Vec2::new(input.cursor.0, input.cursor.1);
            let delta = (mouse - self.initial_mouse) * 0.003;
            self.initial_mouse = mouse;

            if input.middle {
                self.mouse_pan(delta);
            } else if input.left {
                self.mouse_rotate(delta);
            } else if input.right {
                self.mouse_zoom(delta.y);
            }
        }

        self.update_view();
    }

    /// Zooms by a tenth of the scroll offset.
    pub fn on_scroll(&mut self, dy: f32) {
        self.mouse_zoom(dy * 0.1);
        self.update_view();
    }

    pub fn mouse_pan(&mut self, delta: Vec2) {
        let (x_speed, y_speed) = self.pan_speed();
        self.focal_point += -self.right_direction() * delta.x * x_speed * self.distance;
        self.focal_point += self.up_direction() * delta.y * y_speed * self.distance;
    }

    pub fn mouse_rotate(&mut self, delta: Vec2) {
        // Flipped when the camera is upside down, so a drag does not reverse.
        let yaw_sign = if self.up_direction().y < 0.0 {
            -1.0
        } else {
            1.0
        };
        self.yaw += yaw_sign * delta.x * self.rotation_speed();
        self.pitch += delta.y * self.rotation_speed();
    }

    pub fn mouse_zoom(&mut self, delta: f32) {
        self.distance -= delta * self.zoom_speed();
        if self.distance < 1.0 {
            self.focal_point += self.forward_direction();
            self.distance = 1.0;
        }
    }

    fn calculate_position(&self) -> Vec3 {
        self.focal_point - self.forward_direction() * self.distance
    }

    fn update_projection(&mut self) {
        self.aspect_ratio = self.viewport_width / self.viewport_height;
        // Right-handed, depth 0..1.
        self.projection = glam::camera::rh::proj::directx::perspective(
            self.fov.to_radians(),
            self.aspect_ratio,
            self.near_clip,
            self.far_clip,
        );
    }

    fn update_view(&mut self) {
        self.position = self.calculate_position();
        self.view_matrix =
            Mat4::from_rotation_translation(self.orientation(), self.position).inverse();
    }

    /// Pan speed per axis, from the viewport size.
    ///
    /// Pan has to slow down in a small viewport and saturate in a large one.
    fn pan_speed(&self) -> (f32, f32) {
        let x = (self.viewport_width / 1000.0).min(2.4);
        let x_factor = 0.0366 * (x * x) - 0.1778 * x + 0.3021;

        let y = (self.viewport_height / 1000.0).min(2.4);
        let y_factor = 0.0366 * (y * y) - 0.1778 * y + 0.3021;

        (x_factor, y_factor)
    }

    fn rotation_speed(&self) -> f32 {
        0.8
    }

    fn zoom_speed(&self) -> f32 {
        let distance = (self.distance * 0.2).max(0.0);
        (distance * distance).min(100.0)
    }
}

impl Default for EditorCamera {
    /// No projection, no view; `new` calculates them. A default camera is one
    /// nothing has looked through yet, not one with a made-up matrix.
    fn default() -> Self {
        Self {
            fov: 45.0,
            aspect_ratio: 1.778,
            near_clip: 0.1,
            far_clip: 1000.0,
            projection: Mat4::IDENTITY,
            view_matrix: Mat4::IDENTITY,
            position: Vec3::ZERO,
            focal_point: Vec3::ZERO,
            initial_mouse: Vec2::ZERO,
            distance: 10.0,
            pitch: 0.0,
            yaw: 0.0,
            viewport_width: 1280.0,
            viewport_height: 720.0,
        }
    }
}

impl From<EditorCamera> for Camera {
    fn from(editor: EditorCamera) -> Self {
        Self::from_projection(editor.projection())
    }
}
