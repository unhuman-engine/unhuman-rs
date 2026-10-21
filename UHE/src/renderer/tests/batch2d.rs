//! The sprite batch: when it flushes, what it hands over, and that growing it
//! does not lose anything.
//!
//! No device. The batch's job is vertex and index assembly; uploading the buffers
//! and drawing are the pass's, so everything checked here is pure CPU.

use std::mem::size_of;

use bifrost::batch2d::{quad_transform, Batch2D, SpriteVertex};
use bifrost::text::TEXT_WGSL;
use chassis::{Registry, Texture, TextureTag};
use glam::{Mat4, Vec2, Vec3, Vec4};

/// The 2D shaders have to survive naga, which is the same validator wgpu runs at
/// shader-module creation. No device: this is the one check that catches a WGSL
/// mistake without a GPU, and both files are exercised by a pipeline this crate
/// builds.
#[test]
fn the_2d_shaders_validate() {
    forge::wgsl::check(bifrost::batch2d::SPRITE_WGSL, "sprite.wgsl")
        .expect("sprite.wgsl validates");
    forge::wgsl::check(TEXT_WGSL, "text.wgsl").expect("text.wgsl validates");

    for (label, source) in [
        ("sprite.wgsl", bifrost::batch2d::SPRITE_WGSL),
        ("text.wgsl", TEXT_WGSL),
    ] {
        let spirv = forge::wgsl::to_spirv(source, label).expect("the source compiles");
        assert_eq!(spirv.len() % 4, 0, "{label}: SPIR-V is a word stream");
        assert!(!spirv.is_empty(), "{label}: compiled nothing");
    }
}

/// Two distinct texture handles. Minted through `Registry` because a handle's
/// constructor is private to `chassis` and two `Texture::null()`s compare equal.
fn textures() -> (Texture, Texture) {
    let mut registry: Registry<(), TextureTag> = Registry::new();
    (registry.create(()), registry.create(()))
}

fn identity() -> Mat4 {
    Mat4::IDENTITY
}

#[test]
fn a_second_texture_starts_a_second_draw() {
    let (first, second) = textures();
    let mut batch = Batch2D::new();
    batch.begin();

    batch.push_sprite(Some(first), identity(), Vec4::ONE, 1.0);
    batch.push_sprite(Some(first), identity(), Vec4::ONE, 1.0);
    // Complete without a flush: a pass asks what to declare and draw before it
    // has any reason to call one, and a batch it cannot see is a batch it
    // silently does not draw.
    assert_eq!(batch.batches().len(), 1);
    assert_eq!(batch.batches()[0].texture, Some(first));
    assert_eq!(batch.batches()[0].index_offset, 0);
    assert_eq!(batch.batches()[0].index_count, 12);

    batch.push_sprite(Some(second), identity(), Vec4::ONE, 1.0);
    assert_eq!(batch.batches().len(), 2);
    assert_eq!(batch.batches()[1].texture, Some(second));
    assert_eq!(batch.batches()[1].index_offset, 12);
    assert_eq!(batch.batches()[1].index_count, 6);
    assert_eq!(batch.stats().draw_calls, 2);

    // A flush closes the draw in progress, so the same texture starts another.
    batch.flush();
    assert_eq!(batch.batches().len(), 2);
    batch.push_sprite(Some(second), identity(), Vec4::ONE, 1.0);
    assert_eq!(batch.batches().len(), 3);
    assert_eq!(batch.batches()[2].texture, Some(second));
    assert_eq!(batch.batches()[2].index_offset, 18);
    assert_eq!(batch.batches()[2].index_count, 6);
    assert_eq!(batch.stats().draw_calls, 3);

    // Nothing left to close.
    batch.flush();
    assert_eq!(batch.batches().len(), 3);
    assert_eq!(batch.stats().draw_calls, 3);
}

#[test]
fn an_untextured_quad_is_its_own_batch() {
    let (texture, _) = textures();
    let mut batch = Batch2D::new();

    batch.push_quad(identity(), Vec4::ONE);
    batch.push_sprite(Some(texture), identity(), Vec4::ONE, 1.0);
    batch.flush();

    assert_eq!(batch.batches().len(), 2);
    assert_eq!(batch.batches()[0].texture, None);
    assert_eq!(batch.batches()[1].texture, Some(texture));
}

#[test]
fn a_quad_is_four_vertices_and_six_indices() {
    let (texture, _) = textures();
    let mut batch = Batch2D::new();
    for _ in 0..7 {
        batch.push_sprite(Some(texture), identity(), Vec4::ONE, 1.0);
    }
    batch.flush();

    assert_eq!(batch.vertex_count(), 28);
    assert_eq!(batch.index_count(), 42);
    assert_eq!(batch.vertices().len(), 7 * 4);
    assert_eq!(batch.indices().len(), 7 * 6);
    assert_eq!(batch.stats().quad_count, 7);
    assert_eq!(batch.stats().total_vertex_count(), 28);
    assert_eq!(batch.stats().total_index_count(), 42);
    assert_eq!(batch.batches().len(), 1);
    assert_eq!(size_of::<SpriteVertex>(), 40);

    // The per-quad index pattern: 0,1,2, 2,3,0.
    assert_eq!(&batch.indices()[..6], &[0, 1, 2, 2, 3, 0]);
    assert_eq!(&batch.indices()[6..12], &[4, 5, 6, 6, 7, 4]);
    // Untextured quads carry the white-texel UVs and no tiling.
    assert_eq!(batch.vertices()[0].uv, [0.0, 0.0]);
    assert_eq!(batch.vertices()[2].uv, [1.0, 1.0]);
    assert_eq!(batch.vertices()[0].tiling_factor, 1.0);
}

#[test]
fn the_buffers_grow_without_dropping_data() {
    let (texture, _) = textures();
    let quads = 5000u32;
    let mut batch = Batch2D::new();

    for index in 0..quads {
        let transform = quad_transform(Vec3::new(index as f32, 0.0, 0.0), Vec2::splat(1.0));
        batch.push_sprite(Some(texture), transform, Vec4::ONE, 1.0);
    }
    batch.flush();

    assert_eq!(batch.vertex_count(), quads * 4);
    assert_eq!(batch.index_count(), quads * 6);
    assert_eq!(batch.stats().quad_count, quads);
    assert_eq!(batch.stats().draw_calls, 1);

    // A reallocation that dropped the tail would leave the count intact and the
    // contents wrong, so both ends are checked.
    assert_eq!(batch.vertices()[0].position, [-0.5, -0.5, 0.0]);
    let last = batch.vertices()[quads as usize * 4 - 1].position;
    assert_eq!(last, [quads as f32 - 1.5, 0.5, 0.0]);

    batch.begin();
    assert_eq!(batch.vertex_count(), 0);
    assert_eq!(batch.index_count(), 0);
    assert!(batch.batches().is_empty());
}

#[test]
fn quad_transform_scales_the_unit_quad_about_the_position() {
    let transform = quad_transform(Vec3::new(2.0, 3.0, 0.0), Vec2::new(4.0, 6.0));
    let corner = transform * Vec4::new(-0.5, -0.5, 0.0, 1.0);
    assert_eq!([corner.x, corner.y, corner.z], [0.0, 0.0, 0.0]);

    let (texture, _) = textures();
    let mut batch = Batch2D::new();
    batch.push_sprite(Some(texture), transform, Vec4::ONE, 1.0);
    assert_eq!(batch.vertices()[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(batch.vertices()[2].position, [4.0, 6.0, 0.0]);
}

#[test]
fn tint_and_tiling_travel_with_the_vertex() {
    let (texture, _) = textures();
    let tint = Vec4::new(1.0, 0.5, 0.25, 0.75);
    let mut batch = Batch2D::new();
    batch.push_sprite(Some(texture), identity(), tint, 3.0);

    for vertex in batch.vertices() {
        assert_eq!(vertex.color, tint.to_array());
        assert_eq!(vertex.tiling_factor, 3.0);
    }
}

#[test]
fn a_sub_sprite_carries_its_own_atlas_corners() {
    let (texture, _) = textures();
    let uv = [[0.25, 0.0], [0.5, 0.0], [0.5, 0.25], [0.25, 0.25]];
    let mut batch = Batch2D::new();
    batch.push_sprite_uv(Some(texture), identity(), Vec4::ONE, 1.0, uv);
    batch.flush();

    assert_eq!(batch.vertex_count(), 4);
    assert_eq!(batch.batches().len(), 1);
    for (corner, vertex) in batch.vertices().iter().enumerate() {
        assert_eq!(vertex.uv, uv[corner]);
    }
}

#[test]
fn statistics_are_reset_separately_from_the_batch() {
    let (texture, _) = textures();
    let mut batch = Batch2D::new();
    batch.push_sprite(Some(texture), identity(), Vec4::ONE, 1.0);
    batch.reset_stats();
    batch.begin();
    assert_eq!(batch.stats().quad_count, 0);
    assert_eq!(batch.stats().draw_calls, 0);
}
