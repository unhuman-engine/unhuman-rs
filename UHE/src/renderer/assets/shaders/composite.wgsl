// Composition: exposure and the display transform of the linear HDR scene into
// the sRGB backbuffer.
//
// Both textures are the same size as the target, so this is a straight fetch at
// the pixel's own uv with no filtering to speak of, and the triangle covers the
// viewport. The target is sRGB, so the hardware encodes the linear result.

@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var ambient_occlusion: texture_2d<f32>;
@group(0) @binding(3) var composite_sampler: sampler;

struct CompositeParams {
    exposure: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(0) var<uniform> params: CompositeParams;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // (-1,-1), (3,-1), (-1,3): one triangle over the viewport.
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;

    var out: VertexOutput;
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    out.clip = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// ACES fitted curve. It rolls highlights off instead of clipping them at 1.0,
// which is the point of carrying HDR until composition.
fn tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(scene_color, composite_sampler, in.uv, 0.0).rgb;
    let ao = textureSampleLevel(ambient_occlusion, composite_sampler, in.uv, 0.0).r;
    return vec4<f32>(tonemap(color * ao * params.exposure), 1.0);
}
