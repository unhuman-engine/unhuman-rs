//! Screen-space ambient occlusion: the hemisphere kernel and the pass that runs it.
//!
//! Samples lie in the unit hemisphere about +Z, scaled so later ones cluster near
//! the origin, which stops a large radius from reading as a uniform grey wash. The
//! kernel is generated once at pass creation and never changes; each pixel rotates
//! it about +Z so the reduced sample count reads as noise rather than bands.
//!
//! Occlusion is rendered at half resolution and upsampled with a depth test, so
//! the quarter of the pixels that dominate the cost disappear without a full-rate
//! blur. The graph refuses a pass that reads and writes one resource
//! (`ErrorKind::PassStateConflict`, `compiler.rs:154-171`), so an AO pass cannot
//! modulate the colour target it just read. The occlusion texture is multiplied in
//! during composition instead, which also avoids a second full-resolution colour
//! target.

use bytemuck::{Pod, Zeroable};
use chassis::{BindGroup, CommandList, PassScope, Pipeline};

/// Samples per pixel. Halved from the full-rate kernel because the rotation and
/// the depth-aware upsample carry the noise the extra samples used to hide.
pub const KERNEL_SIZE: usize = 32;

/// Bind group number within the SSAO shader.
pub const SSAO_SET: u32 = 0;

/// The kernel and the matrices that turn a texel back into a view-space point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SsaoUniforms {
    pub samples: [[f32; 4]; KERNEL_SIZE],
    pub projection: [[f32; 4]; 4],
    pub inverse_projection: [[f32; 4]; 4],
    /// radius, bias, power, unused.
    pub params: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SsaoUniforms>().is_multiple_of(16));

/// Sensible defaults: a 0.5 m radius, a small depth bias, and a contrast exponent
/// that turns the raw fraction into something that reads as shading.
pub const DEFAULT_RADIUS: f32 = 0.5;
pub const DEFAULT_BIAS: f32 = 0.025;
pub const DEFAULT_POWER: f32 = 2.0;

impl SsaoUniforms {
    pub fn new(
        kernel: &[[f32; 4]; KERNEL_SIZE],
        projection: [[f32; 4]; 4],
        inverse_projection: [[f32; 4]; 4],
        radius: f32,
        bias: f32,
        power: f32,
    ) -> Self {
        Self {
            samples: *kernel,
            projection,
            inverse_projection,
            params: [radius, bias, power, 0.0],
        }
    }
}

/// A hemisphere kernel, deterministic for a given seed.
///
/// Determinism lets a frame be compared against a reference image; no `rand`
/// dependency for six lines of xorshift.
pub fn kernel(size: usize, seed: u64) -> Vec<[f32; 4]> {
    let mut state = seed | 1;
    let mut random = move || {
        // xorshift64*, Marsaglia.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f32 / (1u64 << 53) as f32
    };

    (0..size)
        .map(|index| {
            let mut sample = glam::Vec3::new(
                random() * 2.0 - 1.0,
                random() * 2.0 - 1.0,
                random() * 2.0 - 1.0,
            );
            if sample.length_squared() < 1e-6 {
                sample = glam::Vec3::Z;
            }
            // Into the hemisphere around +Z, unit length, random magnitude.
            let mut sample = sample.normalize() * random();
            // Ramp: scatter outwards for the first samples and cluster the rest,
            // which is what keeps a wide radius from reading as a uniform wash.
            let t = index as f32 / size.max(1) as f32;
            sample *= 0.1 + 0.9 * t * t;
            [sample.x, sample.y, sample.z, 0.0]
        })
        .collect()
}

/// The kernel, fixed length so it can live in a uniform block.
pub fn fixed_kernel(seed: u64) -> [[f32; 4]; KERNEL_SIZE] {
    let mut out = [[0.0f32; 4]; KERNEL_SIZE];
    for (slot, sample) in out.iter_mut().zip(kernel(KERNEL_SIZE, seed)) {
        *slot = sample;
    }
    out
}

pub struct SsaoPass {
    pipeline: Pipeline,
    group: BindGroup,
}

impl SsaoPass {
    pub fn new(pipeline: Pipeline, group: BindGroup) -> Self {
        Self { pipeline, group }
    }

    /// One full-screen triangle: no vertex buffer, the triangle comes from
    /// `@builtin(vertex_index)`.
    pub fn record(
        &self,
        _scope: &PassScope<'_>,
        list: &mut dyn CommandList,
    ) -> Result<(), chassis::Error> {
        list.bind_pipeline(&self.pipeline);
        list.bind_group(SSAO_SET, &self.group);
        list.draw(3, 1);
        Ok(())
    }
}
