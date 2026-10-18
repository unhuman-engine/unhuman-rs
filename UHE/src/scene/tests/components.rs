//! Component defaults and the camera projections.

use glam::{Mat4, Vec3, Vec4};
use meats::{
    euler_degrees, Camera, Collider, ColliderShape, Light, LightKind, ProjectionType, RigidBody,
    RigidBodyKind, Scene, SceneCamera, Transform,
};

fn assert_matrices_close(actual: Mat4, expected: Mat4) {
    let columns = [
        (actual.x_axis, expected.x_axis),
        (actual.y_axis, expected.y_axis),
        (actual.z_axis, expected.z_axis),
        (actual.w_axis, expected.w_axis),
    ];
    for (actual, expected) in columns {
        assert!(
            (actual - expected).length() < 1e-5,
            "column {actual:?} != {expected:?}"
        );
    }
}

#[test]
fn perspective_projection_at_a_known_aspect() {
    let mut camera = SceneCamera::new_perspective(std::f32::consts::FRAC_PI_2, 0.1, 100.0);
    camera.set_viewport_size(16.0, 9.0);
    assert_eq!(camera.projection_type(), ProjectionType::Perspective);

    // f = 1 / tan(fov / 2) = 1 at fov = pi/2, so the standard right-handed
    // perspective matrix with a `[0, 1]` depth range: 1/aspect across, 1 up,
    // `-far/(far-near)` and `-near*far/(far-near)` down the z column.
    let aspect = 16.0 / 9.0;
    let near = 0.1;
    let far = 100.0;
    let expected = Mat4::from_cols(
        Vec4::new(1.0 / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -far / (far - near), -1.0),
        Vec4::new(0.0, 0.0, -near * far / (far - near), 0.0),
    );
    assert_matrices_close(camera.projection(), expected);
}

#[test]
fn orthographic_projection_at_a_known_aspect() {
    let mut camera = SceneCamera::new_orthographic(10.0, -1.0, 1.0);
    camera.set_viewport_size(2.0, 1.0);
    assert_eq!(camera.projection_type(), ProjectionType::Orthographic);

    // half_height = size / 2 = 5, half_width = half_height * aspect = 10, so the
    // projection spans [-10, 10] x [-5, 5]. With a right-handed `[0, 1]` depth
    // range over near/far of -1 and 1, z maps through -1/2 with a 1/2 shift.
    let expected = Mat4::from_cols(
        Vec4::new(1.0 / 10.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / 5.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, -0.5, 0.0),
        Vec4::new(0.0, 0.0, 0.5, 1.0),
    );
    assert_matrices_close(camera.projection(), expected);
}

#[test]
fn a_camera_with_no_viewport_is_square_not_nan() {
    // A zero-height viewport would make the projection non-finite; this one
    // clamps instead.
    let camera = SceneCamera::new_perspective(std::f32::consts::FRAC_PI_4, 0.1, 10.0);
    assert_eq!(camera.aspect_ratio(), 0.0);
    assert!(camera.projection().is_finite());
}

#[test]
fn the_perspective_and_orthographic_planes_are_separate() {
    let mut camera = SceneCamera::new_perspective(0.5, 0.1, 500.0);
    assert_eq!(camera.near(), 0.1);
    assert_eq!(camera.far(), 500.0);
    assert_eq!(camera.perspective_far(), 500.0);

    camera.set_orthographic(4.0, -1.0, 1.0);
    assert_eq!(camera.projection_type(), ProjectionType::Orthographic);
    assert_eq!(camera.near(), -1.0);
    assert_eq!(camera.far(), 1.0);
    // Switching projection does not disturb the perspective planes.
    assert_eq!(camera.perspective_far(), 500.0);
    assert_eq!(camera.perspective_near(), 0.1);
}

#[test]
fn matrix_matches_a_hand_computed_ninety_degree_rotation() {
    let transform = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: Vec3::new(0.0, 0.0, 90.0),
        scale: Vec3::splat(2.0),
    };

    // R_z(90) maps +X to +Y and +Y to -X; scaled by 2 and translated by
    // (1, 2, 3) that is T * R * S.
    let expected = Mat4::from_cols(
        Vec4::new(0.0, 2.0, 0.0, 0.0),
        Vec4::new(-2.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 2.0, 0.0),
        Vec4::new(1.0, 2.0, 3.0, 1.0),
    );
    assert_matrices_close(transform.matrix(), expected);
}

#[test]
fn the_rotation_is_degrees_and_the_default_transform_is_identity() {
    assert_eq!(Transform::default().matrix(), Mat4::IDENTITY);

    let quarter_turn = Transform {
        rotation: Vec3::new(90.0, 0.0, 0.0),
        ..Transform::default()
    };
    let rotated = quarter_turn.matrix().transform_vector3(Vec3::Y);
    assert!((rotated - Vec3::Z).length() < 1e-5, "{rotated:?}");
}

#[test]
fn view_matrix_inverts_the_model_matrix() {
    let transform = Transform {
        translation: Vec3::new(4.0, -2.0, 7.0),
        rotation: Vec3::new(10.0, 20.0, 30.0),
        scale: Vec3::ONE,
    };
    assert_matrices_close(transform.view_matrix() * transform.matrix(), Mat4::IDENTITY);
}

#[test]
fn euler_degrees_round_trips_a_compound_rotation() {
    let transform = Transform {
        rotation: Vec3::new(30.0, 20.0, 10.0),
        ..Transform::default()
    };
    let back = euler_degrees(transform.rotation_quat());
    assert!((back.x - 30.0).abs() < 0.1, "{back:?}");
    assert!((back.y - 20.0).abs() < 0.1, "{back:?}");
    assert!((back.z - 10.0).abs() < 0.1, "{back:?}");
}

#[test]
fn euler_degrees_and_rotation_quat_are_inverses() {
    // `euler_degrees` is not glam's `to_euler`, so the pair can only be checked
    // against itself. Rotations are compared, not quaternions: `q` and `-q` are
    // the same rotation and both are reachable.
    let rotations = [
        Vec3::new(30.0, 20.0, 10.0),
        Vec3::new(-45.0, 60.0, -15.0),
        Vec3::new(10.0, -80.0, 170.0),
        Vec3::new(120.0, 5.0, -35.0),
        Vec3::ZERO,
    ];
    for rotation in rotations {
        let quat = Transform {
            rotation,
            ..Transform::default()
        }
        .rotation_quat();
        let back = Transform {
            rotation: euler_degrees(quat),
            ..Transform::default()
        }
        .rotation_quat();
        assert_matrices_close(Mat4::from_quat(back), Mat4::from_quat(quat));
    }
}

#[test]
fn light_defaults_match_the_cpp_headers() {
    let light = Light::default();
    assert_eq!(light.kind, LightKind::Directional);
    assert_eq!(light.color, Vec3::ONE);
    assert_eq!(light.intensity, 1.0);
    assert_eq!(light.radius, 10.0);
    // 0.2617994 rad / 0.5235988 rad, expressed in degrees here.
    assert!((light.inner_cone_angle - 15.0).abs() < 1e-4);
    assert!((light.outer_cone_angle - 30.0).abs() < 1e-4);
    assert!(light.enabled);
}

#[test]
fn rigid_body_defaults_match_the_cpp_headers() {
    let body = RigidBody::default();
    assert_eq!(body.kind, RigidBodyKind::Static);
    assert_eq!(body.mass, 1.0);
    assert_eq!(body.velocity, Vec3::ZERO);
    assert_eq!(body.damping, 0.05);
    assert_eq!(body.gravity_scale, 1.0);
}

#[test]
fn collider_defaults_match_the_cpp_headers() {
    let collider = Collider::default();
    assert_eq!(
        collider.shape,
        ColliderShape::Box {
            half_extent: Vec3::splat(0.5)
        }
    );
    assert_eq!(collider.offset, Vec3::ZERO);
    assert_eq!(collider.friction, 0.2);
    assert_eq!(collider.restitution, 0.0);
}

#[test]
fn camera_defaults_match_the_cpp_header() {
    let camera = Camera::default();
    assert!(camera.primary);
    assert!(!camera.fixed_aspect_ratio);
    // `SceneCamera`'s default projection is orthographic.
    assert_eq!(
        camera.camera.projection_type(),
        ProjectionType::Orthographic
    );
    assert_eq!(camera.camera.orthographic_size(), 10.0);
    assert!((camera.camera.fov() - 45.0f32.to_radians()).abs() < 1e-6);
}

#[test]
fn a_light_rotated_ninety_degrees_about_y_points_along_negative_x() {
    let mut scene = Scene::new();
    let entity = scene.spawn("Sun");
    scene.insert(
        entity,
        Light {
            kind: LightKind::Directional,
            ..Light::default()
        },
    );
    scene.insert(
        entity,
        Transform {
            rotation: Vec3::new(0.0, 90.0, 0.0),
            ..Transform::default()
        },
    );

    let lights = scene.lights();
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].entity, entity);
    let expected = Vec3::new(-1.0, 0.0, 0.0);
    assert!(
        (lights[0].direction - expected).length() < 1e-5,
        "{:?}",
        lights[0].direction
    );
    assert_eq!(lights[0].position, Vec3::ZERO);
}

#[test]
fn disabled_lights_and_lights_without_a_transform_are_skipped() {
    let mut scene = Scene::new();
    let off = scene.spawn("Off");
    scene.insert(
        off,
        Light {
            enabled: false,
            ..Light::default()
        },
    );
    assert!(scene.lights().is_empty());

    scene.insert(
        off,
        Light {
            enabled: true,
            ..Light::default()
        },
    );
    assert_eq!(scene.lights().len(), 1);

    let untransformed = scene.spawn("NoTransform");
    scene.insert(
        untransformed,
        Light {
            enabled: true,
            ..Light::default()
        },
    );
    let _ = scene.remove::<Transform>(untransformed);
    assert_eq!(scene.lights().len(), 1);
}

#[test]
fn meshes_skips_invisible_and_unresolved_renderers() {
    use meats::MeshRenderer;

    let mut scene = Scene::new();
    let visible = scene.spawn("Visible");
    scene.insert(
        visible,
        MeshRenderer {
            mesh_path: "models/cube.glb".to_string(),
            ..MeshRenderer::default()
        },
    );
    // No handle resolved yet, so nothing is submitted.
    assert!(scene.meshes().is_empty());

    let hidden = scene.spawn("Hidden");
    scene.insert(
        hidden,
        MeshRenderer {
            mesh_path: "models/cube.glb".to_string(),
            visible: false,
            ..MeshRenderer::default()
        },
    );
    assert!(scene.meshes().is_empty());
}
