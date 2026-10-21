//! How long a step costs, so the async question is answered with a number.

mod common;
use common::*;

use adam::PhysicsBackend;
use femur::Physics3D;
use glam::Vec3;
use meats::{RigidBodyKind, Scene};
use std::time::Instant;

/// 500 falling boxes on a floor: enough contacts to be a real solver load.
#[test]
#[ignore = "benchmark"]
fn step_cost() {
    let bodies = 500;
    let mut scene = Scene::new();
    add_body(
        &mut scene,
        "floor",
        Vec3::new(0.0, -1.0, 0.0),
        RigidBodyKind::Static,
        box_shape(Vec3::new(50.0, 1.0, 50.0)),
    );
    for i in 0..bodies {
        let x = (i % 25) as f32 * 1.1 - 13.0;
        let z = (i / 25) as f32 * 1.1 - 11.0;
        add_body(
            &mut scene,
            &format!("box{i}"),
            Vec3::new(x, 0.5 + (i / 100) as f32 * 1.05, z),
            RigidBodyKind::Dynamic,
            box_shape(Vec3::splat(0.5)),
        );
    }

    let mut physics = Physics3D::new();
    for _ in 0..30 {
        physics.step(&mut scene, DT);
    }
    let runs = 300;
    let start = Instant::now();
    for _ in 0..runs {
        physics.step(&mut scene, DT);
    }
    let per_step = start.elapsed().as_secs_f64() * 1.0e3 / runs as f64;
    println!("{bodies} boxes: {per_step:.3} ms/step");
}
