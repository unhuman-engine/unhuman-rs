// Depth-aware upsample of the half-resolution ambient occlusion.
//
// A plain bilinear fetch bleeds occlusion across a depth edge, which shows up as
// a halo around silhouettes. Of the four half-resolution taps under a full-rate
// pixel, this keeps the one whose view-space depth matches the pixel, breaking a
// tie by proximity. That preserves the half-rate occlusion instead of averaging
// it away, which a silent blur would do.

struct Ssao {
    samples: array<vec4<f32>, 32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    // radius, bias, power, unused.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> ssao: Ssao;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var depth_sampler: sampler;
@group(0) @binding(3) var occlusion: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// Only view-space z is needed here, so skip the full position reconstruction.
fn view_depth(uv: vec2<f32>, depth: f32) -> f32 {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view = ssao.inverse_projection * ndc;
    return view.z / view.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let full = vec2<i32>(textureDimensions(depth_texture));
    let half = vec2<f32>(textureDimensions(occlusion));
    let half_max = vec2<i32>(half) - vec2<i32>(1);

    let centre_depth = textureLoad(depth_texture, vec2<i32>(in.clip.xy), 0);
    let centre_z = view_depth(in.uv, centre_depth);

    let texel = in.uv * half - 0.5;
    let base = floor(texel);
    let frac = texel - base;
    let base_i = vec2<i32>(base);

    // Pick the tap with the closest view-space depth; ties go to the tap
    // closest to the pixel, so a flat surface keeps a sharp half-rate sample
    // while a silhouette takes the surface it actually belongs to.
    var best_ao = 1.0;
    var best_dz = 1e9;
    var best_bilinear = -1.0;
    for (var y = 0; y < 2; y = y + 1) {
        for (var x = 0; x < 2; x = x + 1) {
            let coord = clamp(base_i + vec2<i32>(x, y), vec2<i32>(0), half_max);
            let uv = (vec2<f32>(coord) + 0.5) / half;
            let stored = textureLoad(depth_texture, vec2<i32>(uv * vec2<f32>(full)), 0);
            let z = view_depth(uv, stored);
            let dz = abs(centre_z - z);
            let bilinear = mix(1.0 - frac.x, frac.x, f32(x))
                * mix(1.0 - frac.y, frac.y, f32(y));
            if (dz < best_dz - 1e-5 || (abs(dz - best_dz) <= 1e-5 && bilinear > best_bilinear)) {
                best_ao = textureLoad(occlusion, coord, 0).r;
                best_dz = dz;
                best_bilinear = bilinear;
            }
        }
    }

    let ao = clamp(best_ao, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
