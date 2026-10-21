// The editor grid.
//
// Six vertices from `@builtin(vertex_index)` cover the viewport; the shader
// unprojects the near and far planes, intersects that ray with the y = 0 plane,
// and draws a fading grid with a screen-space-constant line width. The axis
// lines are coloured: x red, z blue.
//
// Two constraints:
//   * the two matrices are a uniform block, because the RHI cannot declare push
//     constants;
//   * the grid pass loads the scene depth (see `PipelineKey::grid`), so the
//     fragment writes `@builtin(frag_depth)` from the grid plane's own position:
//     that is what lets geometry occlude the grid. The pipeline has a depth
//     attachment with the test enabled and the write disabled.

struct Grid {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    axis_x: vec4<f32>,
    axis_z: vec4<f32>,
};

@group(0) @binding(0) var<uniform> grid: Grid;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) near_point: vec3<f32>,
    @location(1) far_point: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    let pos = positions[index];

    // Depth 0 is the near plane and 1 the far one: the projection is 0..1.
    let near_clip = vec4<f32>(pos.x, pos.y, 0.0, 1.0);
    let far_clip = vec4<f32>(pos.x, pos.y, 1.0, 1.0);

    let unprojected_near = grid.inverse_view_projection * near_clip;
    let unprojected_far = grid.inverse_view_projection * far_clip;

    var out: VertexOutput;
    out.near_point = unprojected_near.xyz / unprojected_near.w;
    out.far_point = unprojected_far.xyz / unprojected_far.w;
    out.clip = vec4<f32>(pos, 0.9999, 1.0);
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    // Where the ray crosses y = 0.
    let t = -in.near_point.y / (in.far_point.y - in.near_point.y);
    if (t < 0.0) {
        discard;
    }

    let frag_pos = in.near_point + t * (in.far_point - in.near_point);

    let coord = frag_pos.xz;
    let derivative = fwidth(coord);
    let cell = abs(fract(coord - 0.5) - 0.5) / derivative;
    let line = min(cell.x, cell.y);
    let minimum_z = min(derivative.y, 1.0);
    let minimum_x = min(derivative.x, 1.0);

    var color = vec4<f32>(0.4, 0.4, 0.4, 1.0 - min(line, 1.0));

    // The primary axes. Their colours come from the uniform so they can stay
    // dimmer than the gizmo's handles, which sit on the same two lines.
    if (frag_pos.x > -0.5 * minimum_x && frag_pos.x < 0.5 * minimum_x) {
        color = vec4<f32>(grid.axis_x.rgb, color.a);
    }
    if (frag_pos.z > -0.5 * minimum_z && frag_pos.z < 0.5 * minimum_z) {
        color = vec4<f32>(grid.axis_z.rgb, color.a);
    }

    // Fade with distance, cubically so the horizon does not alias into noise.
    let linear_depth = length(frag_pos - in.near_point);
    var fade = max(0.0, 1.0 - linear_depth / 50.0);
    fade = fade * fade * fade;
    color.a = color.a * fade * 0.5;

    if (color.a < 0.01) {
        discard;
    }

    // The grid plane's own depth, so geometry the opaque pass wrote occludes it.
    // Clamped because the depth output is defined over [0, 1].
    let clip = grid.view_projection * vec4<f32>(frag_pos, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    out.color = color;
    return out;
}
