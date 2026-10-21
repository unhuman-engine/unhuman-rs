// The depth prepass and the shadow map: one vertex stage, no colour output.
//
// `ShaderSet::fragment_for` returns `None` for the depth variant, so this module
// has a vertex entry point and nothing else. A prepass is cheaper than writing
// depth from the model shader, and is what the shadow map needs.

struct Shadow {
    light_view_projection: mat4x4<f32>,
};

struct Draw {
    model: mat4x4<f32>,
    tint: vec4<f32>,
    // x = entity id, y = use vertex colour,
    // z = first palette matrix, w = palette matrix count.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> shadow: Shadow;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(2) @binding(0) var<storage, read> palette: array<mat4x4<f32>>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let first = u32(draw.params.z);
    let count = u32(draw.params.w);
    var skin = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    if (count != 0u && dot(weights, vec4<f32>(1.0)) > 0.0) {
        skin = weights.x * palette[first + joints.x]
            + weights.y * palette[first + joints.y]
            + weights.z * palette[first + joints.z]
            + weights.w * palette[first + joints.w];
    }
    return shadow.light_view_projection * draw.model * skin * vec4<f32>(position, 1.0);
}
