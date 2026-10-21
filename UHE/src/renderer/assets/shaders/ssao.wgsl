// Screen-space ambient occlusion, from the depth buffer alone.
//
// The standard formulation: reconstruct the view-space position of the pixel,
// offset it by each kernel sample, project the sample back and compare its depth
// against what the depth buffer holds there.
//
// The projection is right-handed with depth 0..1, and NDC +Y maps to the *top*
// of the framebuffer, which is why the y term of every reprojection below is
// negated.

const TAU: f32 = 6.28318530718;

struct Ssao {
    // Hemisphere samples, in view space, length <= 1.
    samples: array<vec4<f32>, 32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    // radius, bias, power, unused.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> ssao: Ssao;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var depth_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // One triangle covering the viewport: (-1,-1), (3,-1), (-1,3).
    var out: VertexOutput;
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn reconstruct(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view = ssao.inverse_projection * ndc;
    return view.xyz / view.w;
}

// Interleaved gradient noise: no trig and no texture fetch, and deterministic.
fn rotation(pixel: vec2<f32>) -> f32 {
    let s = fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715)));
    return fract(52.9829189 * s) * TAU;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // A depth texture samples with an integer level, unlike a colour one.
    let depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0);
    let origin = reconstruct(in.uv, depth);
    // One projection of the origin, then a w=0 offset per sample: projecting
    // `origin + offset` every iteration would re-project the origin too.
    let origin_clip = ssao.projection * vec4<f32>(origin, 1.0);

    let angle = rotation(in.clip.xy);
    let c = cos(angle);
    let s = sin(angle);

    // The occlusion fraction: how many samples land behind what the depth buffer
    // holds at their own screen position.
    var occluded = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let offset = ssao.samples[i].xyz;
        let rotated = vec3<f32>(
            (offset.x * c - offset.y * s) * ssao.params.x,
            (offset.x * s + offset.y * c) * ssao.params.x,
            offset.z * ssao.params.x,
        );
        let projected = origin_clip + ssao.projection * vec4<f32>(rotated, 0.0);
        let sample_uv = vec2<f32>(
            projected.x / projected.w * 0.5 + 0.5,
            0.5 - projected.y / projected.w * 0.5,
        );

        // Outside the viewport there is no depth to compare against, so the
        // sample counts as unoccluded rather than as fully occluded.
        if (sample_uv.x > 0.0 && sample_uv.x < 1.0 && sample_uv.y > 0.0 && sample_uv.y < 1.0) {
            let stored = textureSampleLevel(depth_texture, depth_sampler, sample_uv, 0);
            let reference = projected.z / projected.w - ssao.params.y;
            if (stored < reference) {
                occluded = occluded + 1.0;
            }
        }
    }

    let ao = pow(clamp(1.0 - occluded / 32.0, 0.0, 1.0), ssao.params.z);
    return vec4<f32>(ao, ao, ao, 1.0);
}
