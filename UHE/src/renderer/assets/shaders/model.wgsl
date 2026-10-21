// The model pass: PBR-ish metallic-roughness shading with directional, point
// and spot lights, one shadow lookup, and the five maps the material system
// binds.
//
// `chassis::Resources::create_bind_group` has no descriptor arrays, so the
// Tier 2/3 glTF extensions that need extra maps (clearcoat, sheen, iridescence,
// anisotropy, transmission, volume) have no way to reach a shader here.
// `params.y` (`unlit`) and `emissive_factor_strength` are the two extension
// values the maps below do use.

struct Frame {
    view_projection: mat4x4<f32>,
    shadow_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    // x = light count, y = lighting enabled, z/w unused.
    params: vec4<f32>,
};

struct Light {
    // x = type (0 directional, 1 point, 2 spot), y = radius,
    // z/w = cos(inner)/cos(outer) cone angles.
    type_radius_cones: vec4<f32>,
    position_or_direction: vec4<f32>,
    direction_pad: vec4<f32>,
    color_intensity: vec4<f32>,
};

struct MaterialParams {
    base_color_factor: vec4<f32>,
    emissive_factor_strength: vec4<f32>,
    // metallic, roughness, normal scale, occlusion strength.
    factors: vec4<f32>,
    // alpha cutoff, alpha mode, unlit, pad.
    params: vec4<f32>,
    // Per slot: (cos, sin, scale x, scale y), in MaterialTextureSlot order.
    uv_transform: array<vec4<f32>, 5>,
    uv_offset: array<vec4<f32>, 5>,
};

struct Draw {
    model: mat4x4<f32>,
    tint: vec4<f32>,
    // x = entity id, y = use vertex colour,
    // z = first palette matrix, w = palette matrix count.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler;

@group(1) @binding(0) var<uniform> material: MaterialParams;
@group(1) @binding(1) var material_sampler: sampler;
@group(1) @binding(2) var albedo_texture: texture_2d<f32>;
@group(1) @binding(3) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(4) var normal_texture: texture_2d<f32>;
@group(1) @binding(5) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(6) var emissive_texture: texture_2d<f32>;

@group(2) @binding(0) var<uniform> draw: Draw;
@group(3) @binding(0) var<storage, read> palette: array<mat4x4<f32>>;

const PI: f32 = 3.14159265359;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // xyz tangent, w handedness.
    @location(3) tangent: vec4<f32>,
    // Four indices into the frame's palette.
    @location(4) joints: vec4<u32>,
    // Weights parallel to `joints`; a zero sum leaves the vertex in bind pose.
    @location(5) weights: vec4<f32>,
    // COLOR_0, at location 6.
    @location(6) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) color: vec4<f32>,
};

fn skin_matrix(in: VertexInput) -> mat4x4<f32> {
    let first = u32(draw.params.z);
    let count = u32(draw.params.w);
    if (count == 0u || dot(in.weights, vec4<f32>(1.0)) <= 0.0) {
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
    }
    let j = in.joints;
    return in.weights.x * palette[first + j.x]
        + in.weights.y * palette[first + j.y]
        + in.weights.z * palette[first + j.z]
        + in.weights.w * palette[first + j.w];
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let skin = skin_matrix(in);
    let local = skin * vec4<f32>(in.position, 1.0);
    let world = draw.model * local;
    out.world_position = world.xyz;

    // The model matrix is a TRS with no shear, so transforming by the matrix is
    // correct and an inverse-transpose is not needed. The skin matrix is applied
    // to the local normal first, in the same space as the position.
    let normal = normalize((draw.model * (skin * vec4<f32>(in.normal, 0.0))).xyz);
    let tangent = normalize((draw.model * (skin * vec4<f32>(in.tangent.xyz, 0.0))).xyz);
    out.normal = normal;
    out.tangent = tangent;
    out.bitangent = cross(normal, tangent) * in.tangent.w;
    out.uv = in.uv;
    out.color = in.color;
    out.clip = frame.view_projection * world;
    return out;
}

// uv' = R(rot) * (uv * scale) + offset, in the engine's UV space: the loader
// already flipped V, so the transform is applied here rather than in the decoder.
fn slot_uv(slot: u32, uv: vec2<f32>) -> vec2<f32> {
    let t = material.uv_transform[slot];
    let o = material.uv_offset[slot];
    let scaled = uv * vec2<f32>(t.z, t.w);
    let rotated = vec2<f32>(
        scaled.x * t.x - scaled.y * t.y,
        scaled.x * t.y + scaled.y * t.x,
    );
    return rotated + vec2<f32>(o.x, o.y);
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-7);
    let gl = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-7);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn shade(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, radiance: vec3<f32>,
         albedo: vec3<f32>, metallic: f32, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }

    let h = normalize(view_dir + light_dir);
    let n_dot_v = max(dot(normal, view_dir), 1e-4);
    let n_dot_h = max(dot(normal, h), 0.0);

    let ndf = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(max(dot(h, view_dir), 0.0), f0);

    let specular = (ndf * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-7);
    // Metals have no diffuse lobe; the same Fresnel term removes it for them.
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / PI;

    return (diffuse + specular) * radiance * n_dot_l;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Every sample happens here, unconditionally: `textureSample` needs uniform
    // control flow, and a branch on a material or light value is not uniform.
    let albedo_texel = textureSample(albedo_texture, material_sampler, slot_uv(0u, in.uv));
    let mr = textureSample(metallic_roughness_texture, material_sampler, slot_uv(1u, in.uv));
    let normal_texel = textureSample(normal_texture, material_sampler, slot_uv(2u, in.uv));
    let occlusion = textureSample(occlusion_texture, material_sampler, slot_uv(3u, in.uv)).r;
    let emissive_texel = textureSample(emissive_texture, material_sampler, slot_uv(4u, in.uv));

    // Multiplied unconditionally: an absent COLOR_0 is white on the CPU side
    // (`mesh::interleave`), so this is a no-op for a mesh that has none.
    let base = albedo_texel.rgb * material.base_color_factor.rgb * draw.tint.rgb * in.color.rgb;
    let alpha = albedo_texel.a * material.base_color_factor.a * draw.tint.a;

    // glTF alphaMode: 0 opaque, 1 mask, 2 blend.
    if (material.params.y == 1.0 && alpha < material.params.x) {
        discard;
    }

    let metallic = clamp(mr.b * material.factors.x, 0.0, 1.0);
    let roughness = clamp(mr.g * material.factors.y, 0.04, 1.0);

    // Tangent-space normal, assembled from the interpolated frame. The neutral
    // texel (0.5, 0.5, 1) leaves the geometric normal alone.
    let n = normalize(in.normal);
    let t = normalize(in.tangent);
    let b = normalize(in.bitangent);
    let tangent_normal = vec3<f32>(
        (normal_texel.r * 2.0 - 1.0) * material.factors.z,
        (normal_texel.g * 2.0 - 1.0) * material.factors.z,
        normal_texel.b * 2.0 - 1.0,
    );
    let normal = normalize(t * tangent_normal.x + b * tangent_normal.y + n * tangent_normal.z);

    let emissive = emissive_texel.rgb * material.emissive_factor_strength.rgb
        * material.emissive_factor_strength.a;

    if (material.params.z == 1.0) {
        // KHR_materials_unlit: base colour, straight out.
        return vec4<f32>(base + emissive, alpha);
    }

    let camera = frame.camera_position.xyz;
    let view_dir = normalize(camera - in.world_position);
    // 0.04 is glTF's fixed dielectric F0; Metals use their base colour.
    let f0 = mix(vec3<f32>(0.04), base, metallic);

    // One shadow lookup, for the first directional light. The RHI has no compare
    // sampler, so the depth comparison is done by hand.
    let shadow_clip = frame.shadow_view_projection * vec4<f32>(in.world_position, 1.0);
    let shadow_uv = vec2<f32>(
        shadow_clip.x / shadow_clip.w * 0.5 + 0.5,
        0.5 - shadow_clip.y / shadow_clip.w * 0.5,
    );
    var shadow = 1.0;
    if (shadow_uv.x > 0.0 && shadow_uv.x < 1.0 && shadow_uv.y > 0.0 && shadow_uv.y < 1.0) {
        // A depth texture samples with an integer level, unlike a colour one.
        let stored = textureSampleLevel(shadow_map, shadow_sampler, shadow_uv, 0);
        let reference = shadow_clip.z / shadow_clip.w - 0.0015;
        shadow = select(1.0, 0.0, stored < reference);
    }

    // Occlusion scales the direct-light sum only; ambient and emissive are not
    // cavity-shaded. The flat 0.08 ambient and the `1 - exp(-x)` exposure tone
    // map match the original's shading curve so highlights roll off instead of
    // clipping at 1.0.
    let occlusion_amount = 1.0 - (1.0 - occlusion) * material.factors.w;
    let ambient = base * 0.08;

    var lo = vec3<f32>(0.0);
    let light_count = u32(frame.params.x);
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light = lights[i];
        let kind = light.type_radius_cones.x;
        let radiance = light.color_intensity.rgb * light.color_intensity.a * PI;

        if (kind < 0.5) {
            // Directional: the direction is the axis the light travels along.
            let light_dir = normalize(-light.position_or_direction.xyz);
            lo = lo + shade(normal, view_dir, light_dir, radiance * shadow, base, metallic,
                            roughness, f0);
        } else {
            let to_light = light.position_or_direction.xyz - in.world_position;
            let distance = max(length(to_light), 1e-4);
            let light_dir = to_light / distance;

            // Window falloff only: the original does not divide by distance
            // squared, so a point light keeps its intensity out to its radius.
            let radius = light.type_radius_cones.y;
            let falloff = clamp(1.0 - (distance * distance) / max(radius * radius, 1e-4), 0.0, 1.0);
            var attenuation = falloff * falloff;

            if (kind > 1.5) {
                // Spot: cosines, so the cone test is two dots.
                let cos_angle = dot(normalize(-light.direction_pad.xyz), light_dir);
                let inner = light.type_radius_cones.z;
                let outer = light.type_radius_cones.w;
                attenuation = attenuation
                    * clamp((cos_angle - outer) / max(inner - outer, 1e-4), 0.0, 1.0);
            }

            lo = lo + shade(normal, view_dir, light_dir, radiance * attenuation, base,
                            metallic, roughness, f0);
        }
    }

    let lit = ambient + lo * occlusion_amount + emissive;
    return vec4<f32>(vec3<f32>(1.0) - exp(-lit), alpha);
}
