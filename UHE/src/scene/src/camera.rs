//! Scene camera: projection type, clipping planes and viewport aspect.

use glam::Mat4;

/// The discriminants are `Perspective` = 0, because the serializer writes the
/// integer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum ProjectionType {
    Perspective,
    /// The default projection.
    #[default]
    Orthographic,
}

/// A camera's projection. The matrix is rebuilt on demand, which cannot go
/// stale and costs a handful of multiplies.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SceneCamera {
    projection_type: ProjectionType,
    perspective_fov: f32,
    perspective_near: f32,
    perspective_far: f32,
    orthographic_size: f32,
    orthographic_near: f32,
    orthographic_far: f32,
    aspect_ratio: f32,
}

impl Default for SceneCamera {
    fn default() -> Self {
        Self {
            projection_type: ProjectionType::Orthographic,
            perspective_fov: 45.0f32.to_radians(),
            perspective_near: 0.01,
            perspective_far: 1000.0,
            orthographic_size: 10.0,
            orthographic_near: -1.0,
            orthographic_far: 1.0,
            aspect_ratio: 0.0,
        }
    }
}

impl SceneCamera {
    pub fn new_perspective(fov_radians: f32, near: f32, far: f32) -> Self {
        let mut camera = Self::default();
        camera.set_perspective(fov_radians, near, far);
        camera.projection_type = ProjectionType::Perspective;
        camera
    }

    pub fn new_orthographic(size: f32, near: f32, far: f32) -> Self {
        let mut camera = Self::default();
        camera.set_orthographic(size, near, far);
        camera
    }

    /// Right-handed with a `[0, 1]` depth range, matching the renderer's editor
    /// camera and the Vulkan/wgpu backends; glam spells that convention
    /// `camera::rh::proj::directx`. An OpenGL `[-1, 1]` range here would put the
    /// near half of the frustum behind the near plane and clip it.
    pub fn projection(&self) -> Mat4 {
        let aspect = self.effective_aspect_ratio();
        match self.projection_type {
            ProjectionType::Perspective => glam::camera::rh::proj::directx::perspective(
                self.perspective_fov,
                aspect,
                self.perspective_near,
                self.perspective_far,
            ),
            ProjectionType::Orthographic => {
                let half_height = self.orthographic_size * 0.5;
                let half_width = half_height * aspect;
                glam::camera::rh::proj::directx::orthographic(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.orthographic_near,
                    self.orthographic_far,
                )
            }
        }
    }

    pub fn projection_type(&self) -> ProjectionType {
        self.projection_type
    }

    pub fn set_projection_type(&mut self, kind: ProjectionType) {
        self.projection_type = kind;
    }

    pub fn set_viewport_size(&mut self, width: f32, height: f32) {
        self.aspect_ratio = aspect_ratio_of(width, height);
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }

    pub fn fov(&self) -> f32 {
        self.perspective_fov
    }

    /// The active projection's near plane. `SetPerspective` and
    /// `SetOrthographic` write separate fields, so this picks the one in use.
    pub fn near(&self) -> f32 {
        match self.projection_type {
            ProjectionType::Perspective => self.perspective_near,
            ProjectionType::Orthographic => self.orthographic_near,
        }
    }

    /// The active projection's far plane. See [`near`](Self::near).
    pub fn far(&self) -> f32 {
        match self.projection_type {
            ProjectionType::Perspective => self.perspective_far,
            ProjectionType::Orthographic => self.orthographic_far,
        }
    }

    pub fn set_perspective(&mut self, fov_radians: f32, near: f32, far: f32) {
        self.perspective_fov = fov_radians;
        self.perspective_near = near;
        self.perspective_far = far;
    }

    pub fn set_fov(&mut self, fov_radians: f32) {
        self.perspective_fov = fov_radians;
    }

    pub fn perspective_near(&self) -> f32 {
        self.perspective_near
    }

    pub fn perspective_far(&self) -> f32 {
        self.perspective_far
    }

    pub fn set_perspective_near(&mut self, near: f32) {
        self.perspective_near = near;
    }

    pub fn set_perspective_far(&mut self, far: f32) {
        self.perspective_far = far;
    }

    pub fn orthographic_size(&self) -> f32 {
        self.orthographic_size
    }

    pub fn set_orthographic(&mut self, size: f32, near: f32, far: f32) {
        self.orthographic_size = size;
        self.orthographic_near = near;
        self.orthographic_far = far;
        self.projection_type = ProjectionType::Orthographic;
    }

    pub fn set_orthographic_size(&mut self, size: f32) {
        self.orthographic_size = size;
    }

    pub fn orthographic_near(&self) -> f32 {
        self.orthographic_near
    }

    pub fn orthographic_far(&self) -> f32 {
        self.orthographic_far
    }

    pub fn set_orthographic_near(&mut self, near: f32) {
        self.orthographic_near = near;
    }

    pub fn set_orthographic_far(&mut self, far: f32) {
        self.orthographic_far = far;
    }

    /// A zero-height viewport would make the projection non-finite, so a camera
    /// that has never been given a viewport (aspect 0) falls back to square
    /// rather than NaN.
    fn effective_aspect_ratio(&self) -> f32 {
        if self.aspect_ratio.is_finite() && self.aspect_ratio > 0.0 {
            self.aspect_ratio
        } else {
            1.0
        }
    }
}

fn aspect_ratio_of(width: f32, height: f32) -> f32 {
    if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 {
        width / height
    } else {
        1.0
    }
}

/// A render camera: its projection, whether it is primary, and whether the
/// aspect ratio is fixed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Camera {
    pub camera: SceneCamera,
    pub primary: bool,
    pub fixed_aspect_ratio: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            camera: SceneCamera::default(),
            primary: true,
            fixed_aspect_ratio: false,
        }
    }
}

impl Camera {
    pub fn projection(&self) -> Mat4 {
        self.camera.projection()
    }
}
