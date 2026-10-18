use std::sync::Arc;

use crate::image::{Image, MipLevel};

/// How a keyframe's interpolation is meant (`glTF` sampler interpolation).
///
/// `CubicSpline` carries the Hermite tangents from the glTF output layout
/// `[inTangent_i, value_i, outTangent_i]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Interpolation {
    Step,
    #[default]
    Linear,
    CubicSpline,
}

/// One translation or scale curve, targeting a single node.
///
/// `times`/`values` are parallel vectors rather than keyframes, because that is
/// what a keep-mask wants to index anyway. `target` is the glTF node index.
///
/// `in_tangents`/`out_tangents` are parallel to `values` and empty for
/// `Step`/`Linear` tracks, so those stay compact.
#[derive(Clone, Debug, Default)]
pub struct VectorTrack {
    pub target: u32,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    pub values: Vec<[f32; 3]>,
    pub in_tangents: Vec<[f32; 3]>,
    pub out_tangents: Vec<[f32; 3]>,
}

/// One rotation curve, targeting a single node.
///
/// Values are unit quaternions in `glTF` order `[x, y, z, w]`; `CubicSpline`
/// tangents are derivatives and are deliberately *not* normalised (glTF 2.0
/// animation section).
#[derive(Clone, Debug, Default)]
pub struct QuaternionTrack {
    pub target: u32,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    pub values: Vec<[f32; 4]>,
    pub in_tangents: Vec<[f32; 4]>,
    pub out_tangents: Vec<[f32; 4]>,
}

/// One animation clip: three lists of curves, a name and a duration.
///
/// `duration` is the largest input time across every channel of the clip, which
/// is what playback wraps on and what the compression tolerances are stated
/// against.
#[derive(Clone, Debug, Default)]
pub struct AnimationData {
    pub name: String,
    pub duration: f32,
    pub position_tracks: Vec<VectorTrack>,
    pub rotation_tracks: Vec<QuaternionTrack>,
    pub scale_tracks: Vec<VectorTrack>,
}

/// One bone of a skin's joint hierarchy.
///
/// `id` is the index into the owning [`SkeletonData`]'s `bones`, and equals the
/// glTF node index it came from; `parent` is -1 at a root.
#[derive(Clone, Debug)]
pub struct BoneData {
    pub name: String,
    pub id: i32,
    pub parent: i32,
    pub inverse_bind: [[f32; 4]; 4],
    pub local: [[f32; 4]; 4],
}

/// A joint hierarchy with the inverse bind pose that puts a mesh in it.
///
/// One hierarchy is built across all nodes, and each skin binds a subset of it.
#[derive(Clone, Debug, Default)]
pub struct SkeletonData {
    pub bones: Vec<BoneData>,
    /// Joint node indices of the first skin, so call sites do not need a skin
    /// lookup.
    pub joints: Vec<i32>,
    pub root: i32,
}

/// One skin: the joints it binds and their inverse bind matrices.
///
/// `inverse_bind` is parallel to `joints`.
#[derive(Clone, Debug, Default)]
pub struct SkinData {
    pub name: String,
    pub joints: Vec<i32>,
    pub inverse_bind: Vec<[[f32; 4]; 4]>,
}

/// How a material's alpha is interpreted (`glTF` `alphaMode`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AlphaMode {
    /// Alpha ignored.
    #[default]
    Opaque,
    /// Fragment discarded below `alpha_cutoff`.
    Mask,
    /// Alpha blended. Needs a depth-sorted pass, which does not exist yet, so
    /// the renderer can warn instead of drawing it in submission order.
    Blend,
}

/// `glTF` sampler filter. The mip-map variant folds into these two, because
/// mip selection is the sampler's other axis, not a third filter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Filter {
    Nearest,
    Linear,
}

/// `glTF` sampler wrap mode, per axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Wrap {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

/// The sampler a texture declares, with the `glTF` spec defaults.
///
/// Mag LINEAR, min LINEAR_MIPMAP_LINEAR, REPEAT wrap. A file with no sampler
/// object gets exactly these.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerData {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub wrap_s: Wrap,
    pub wrap_t: Wrap,
}

impl Default for SamplerData {
    fn default() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            wrap_s: Wrap::Repeat,
            wrap_t: Wrap::Repeat,
        }
    }
}

/// A decoded material texture with the sampler it was declared with.
///
/// Device-free: the pixels live on the CPU, so the material list exists without
/// a device. The pixels are shared (`Arc`), because one glTF image is routinely
/// bound by several materials and slots. `mips` is the chain the file shipped,
/// empty when it shipped none: recomputing an authored chain would discard its
/// filter.
#[derive(Clone, Debug)]
pub struct TextureData {
    pub image: Arc<Image>,
    pub mips: Arc<[MipLevel]>,
    pub sampler: SamplerData,
}

/// How many per-material texture slots a `KHR_texture_transform` can ride on.
///
/// Mirrors the renderer's per-material GPU slot order.
pub const MATERIAL_TEXTURE_SLOT_COUNT: usize = 19;

/// Indices into [`MaterialData::uv_transforms`], in GPU slot order.
pub mod texture_slot {
    pub const ALBEDO: usize = 0;
    pub const METALLIC_ROUGHNESS: usize = 1;
    pub const NORMAL: usize = 2;
    pub const OCCLUSION: usize = 3;
    pub const EMISSIVE: usize = 4;
    pub const CLEARCOAT: usize = 5;
    pub const CLEARCOAT_ROUGHNESS: usize = 6;
    pub const CLEARCOAT_NORMAL: usize = 7;
    pub const SPECULAR: usize = 8;
    pub const SPECULAR_COLOR: usize = 9;
    pub const SHEEN_COLOR: usize = 10;
    pub const SHEEN_ROUGHNESS: usize = 11;
    pub const TRANSMISSION: usize = 12;
    pub const THICKNESS: usize = 13;
    pub const IRIDESCENCE: usize = 14;
    pub const IRIDESCENCE_THICKNESS: usize = 15;
    pub const ANISOTROPY: usize = 16;
    pub const DIFFUSE_TRANSMISSION: usize = 17;
    pub const DIFFUSE_TRANSMISSION_COLOR: usize = 18;
}

/// `KHR_texture_transform`, converted to the engine's V-flipped UV convention.
///
/// [`crate::load_model_geometry`] flips V while importing TEXCOORD_0, so the raw
/// glTF offset cannot be applied on top of the imported coordinates. The stored
/// parameters are engine-space: the shader computes
/// `uv' = R(rotation) * (uv * scale) + offset` in that space, with `cos`/`sin`
/// of the *original* glTF angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UVTransform {
    pub cos_rotation: f32,
    pub sin_rotation: f32,
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    /// The `texCoordIndex` the map asked for. Only TEXCOORD_0 is extracted, so a
    /// non-zero value cannot be honoured and is warned about at load time.
    pub tex_coord_set: u32,
    pub has_transform: bool,
}

impl Default for UVTransform {
    fn default() -> Self {
        Self {
            cos_rotation: 1.0,
            sin_rotation: 0.0,
            scale: [1.0, 1.0],
            offset: [0.0, 0.0],
            tex_coord_set: 0,
            has_transform: false,
        }
    }
}

/// How a `KHR_materials_transmission` surface is composited.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BlendApproach {
    Opaque,
    /// Drawn in the opaque pass, ignoring transmission. Wrong but stable for a
    /// thin surface with a small factor.
    #[default]
    OpaqueWithTransmission,
    /// Drawn after opaques, blended. Needs depth sorting to avoid artefacts.
    Blend,
}

/// The optional PBR extension values of one material.
///
/// Kept in one aggregate so "which extensions did this material use" is a
/// single question, and a material declaring none renders byte-identically to
/// one loaded before this existed. Defaults are the *spec* defaults, not zero:
/// a zeroed attenuation distance or sheen roughness silently disables the
/// feature it names.
#[derive(Clone, Debug)]
pub struct MaterialExtensions {
    /// `KHR_materials_emissive_strength`, spec default 1.0.
    pub emissive_strength: f32,
    /// `KHR_materials_ior`, spec default 1.5 (dielectric F0).
    pub ior: f32,
    /// `KHR_materials_unlit`.
    pub unlit: bool,

    /// `KHR_materials_clearcoat`.
    pub has_clearcoat: bool,
    pub clearcoat_factor: f32,
    pub clearcoat_roughness_factor: f32,
    pub clearcoat_normal_scale: f32,
    pub clearcoat_texture: Option<Arc<TextureData>>,
    pub clearcoat_roughness_texture: Option<Arc<TextureData>>,
    pub clearcoat_normal_texture: Option<Arc<TextureData>>,

    /// `KHR_materials_specular`. A factor of 1.0 reproduces the fixed 0.04 F0.
    pub has_specular: bool,
    pub specular_factor: f32,
    pub specular_color_factor: [f32; 3],
    pub specular_texture: Option<Arc<TextureData>>,
    pub specular_color_texture: Option<Arc<TextureData>>,

    /// `KHR_materials_sheen`.
    pub has_sheen: bool,
    pub sheen_color_factor: [f32; 3],
    pub sheen_roughness_factor: f32,
    pub sheen_color_texture: Option<Arc<TextureData>>,
    pub sheen_roughness_texture: Option<Arc<TextureData>>,

    /// `KHR_materials_transmission` and `KHR_materials_volume`. Attenuation
    /// distance defaults to infinity: light travels without attenuation.
    pub has_transmission: bool,
    pub transmission_factor: f32,
    pub thickness_factor: f32,
    pub attenuation_distance: f32,
    pub attenuation_color: [f32; 3],
    pub transmission_texture: Option<Arc<TextureData>>,
    pub thickness_texture: Option<Arc<TextureData>>,
    /// How transmission is drawn, chosen once at load time.
    pub transmission_blend: BlendApproach,

    /// `KHR_materials_iridescence`.
    pub has_iridescence: bool,
    pub iridescence_factor: f32,
    pub iridescence_ior: f32,
    pub iridescence_thickness_minimum: f32,
    pub iridescence_thickness_maximum: f32,
    pub iridescence_texture: Option<Arc<TextureData>>,
    pub iridescence_thickness_texture: Option<Arc<TextureData>>,

    /// `KHR_materials_anisotropy`.
    pub has_anisotropy: bool,
    pub anisotropy_strength: f32,
    pub anisotropy_rotation: f32,
    pub anisotropy_texture: Option<Arc<TextureData>>,

    /// `KHR_materials_diffuse_transmission`: thin-surface translucency.
    pub has_diffuse_transmission: bool,
    pub diffuse_transmission_factor: f32,
    pub diffuse_transmission_color: [f32; 3],
    pub diffuse_transmission_texture: Option<Arc<TextureData>>,
    pub diffuse_transmission_color_texture: Option<Arc<TextureData>>,

    /// `KHR_materials_pbrSpecularGlossiness`. The workload this replaces uses
    /// these in place of the metallic-roughness factors when present.
    pub has_specular_glossiness: bool,
    pub sg_diffuse_factor: [f32; 4],
    pub sg_diffuse_texture: Option<Arc<TextureData>>,
    pub sg_specular_factor: [f32; 3],
    pub sg_glossiness_factor: f32,
    pub sg_specular_glossiness_texture: Option<Arc<TextureData>>,
}

impl Default for MaterialExtensions {
    fn default() -> Self {
        Self {
            emissive_strength: 1.0,
            ior: 1.5,
            unlit: false,

            has_clearcoat: false,
            clearcoat_factor: 0.0,
            clearcoat_roughness_factor: 0.0,
            clearcoat_normal_scale: 1.0,
            clearcoat_texture: None,
            clearcoat_roughness_texture: None,
            clearcoat_normal_texture: None,

            has_specular: false,
            specular_factor: 1.0,
            specular_color_factor: [1.0, 1.0, 1.0],
            specular_texture: None,
            specular_color_texture: None,

            has_sheen: false,
            sheen_color_factor: [0.0, 0.0, 0.0],
            sheen_roughness_factor: 0.0,
            sheen_color_texture: None,
            sheen_roughness_texture: None,

            has_transmission: false,
            transmission_factor: 0.0,
            thickness_factor: 0.0,
            attenuation_distance: f32::INFINITY,
            attenuation_color: [1.0, 1.0, 1.0],
            transmission_texture: None,
            thickness_texture: None,
            transmission_blend: BlendApproach::OpaqueWithTransmission,

            has_iridescence: false,
            iridescence_factor: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness_minimum: 100.0,
            iridescence_thickness_maximum: 400.0,
            iridescence_texture: None,
            iridescence_thickness_texture: None,

            has_anisotropy: false,
            anisotropy_strength: 0.0,
            anisotropy_rotation: 0.0,
            anisotropy_texture: None,

            has_diffuse_transmission: false,
            diffuse_transmission_factor: 0.0,
            diffuse_transmission_color: [1.0, 1.0, 1.0],
            diffuse_transmission_texture: None,
            diffuse_transmission_color_texture: None,

            has_specular_glossiness: false,
            sg_diffuse_factor: [1.0, 1.0, 1.0, 1.0],
            sg_diffuse_texture: None,
            sg_specular_factor: [1.0, 1.0, 1.0],
            sg_glossiness_factor: 1.0,
            sg_specular_glossiness_texture: None,
        }
    }
}

/// A material's PBR factors, the five core texture slots, and the optional
/// `KHR_materials_*` values that reach shading.
#[derive(Clone, Debug)]
pub struct MaterialData {
    pub name: String,
    /// `baseColorFactor`, spec default opaque white.
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// `normalTexture.scale`, spec default 1.0; a 0.0 default silently
    /// disables every normal map in the scene.
    pub normal_scale: f32,
    /// `occlusionTexture.strength`, spec default 1.0.
    pub occlusion_strength: f32,
    /// `emissiveFactor`, spec default **black**: defaulting to white makes
    /// every unlit surface glow.
    pub emissive: [f32; 3],
    pub alpha: AlphaMode,
    /// `alphaCutoff`, spec default 0.5. Only meaningful for [`AlphaMode::Mask`].
    pub alpha_cutoff: f32,
    /// Selects the cull-mode pipeline variant.
    pub double_sided: bool,
    pub albedo: Option<Arc<TextureData>>,
    pub metallic_roughness: Option<Arc<TextureData>>,
    pub normal: Option<Arc<TextureData>>,
    pub occlusion: Option<Arc<TextureData>>,
    pub emissive_texture: Option<Arc<TextureData>>,
    /// Per-slot `KHR_texture_transform`, identity where the file declared none.
    /// Indexed by [`texture_slot`].
    pub uv_transforms: [UVTransform; MATERIAL_TEXTURE_SLOT_COUNT],
    /// The optional PBR extension values.
    pub extensions: MaterialExtensions,
}

impl Default for MaterialData {
    /// The material a primitive gets when the file names none, with the spec's
    /// defaults rather than `f32::default()`. A zeroed `roughness` and a
    /// `normal_scale` of 0 would both change how every unlidded surface renders.
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            albedo: None,
            metallic_roughness: None,
            normal: None,
            occlusion: None,
            emissive_texture: None,
            uv_transforms: [UVTransform::default(); MATERIAL_TEXTURE_SLOT_COUNT],
            extensions: MaterialExtensions::default(),
        }
    }
}

/// CPU-side geometry for one `glTF` mesh primitive, plus the material and the
/// model-level animation that goes with it.
///
/// One `MeshData` per primitive, not per node: a `glTF` mesh is extracted once
/// and every node that references it points at the same entry (see
/// [`ModelNode::primitives`]), which is what keeps repeated props and foliage
/// from being duplicated per instance.
///
/// The four vertex arrays are `[f32; ...]`, so a consumer never has to know
/// about a math library to read geometry. `joints`/`weights`/`colors` are
/// additional, because a skinned mesh cannot be posed without them.
///
/// `animations` and `skeleton` are model-level facts the contract nevertheless
/// asks for on `MeshData`. They are filled on the *first* mesh of a model and
/// left empty on the rest, because cloning every clip once per primitive would
/// cost more than the geometry does; [`crate::Assets::meshes`] returns the list
/// and the first entry is the one that carries them.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    /// The `glTF` mesh name, `"Unnamed_Mesh"` when the file gave none.
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub colors: Vec<[f32; 4]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    /// True when this primitive carried `COLOR_0`. One fact about the whole
    /// primitive, not per vertex: the shader multiplies vertex colour in only
    /// when it is set, and `COLOR_0` is linear data that must not be
    /// gamma-decoded.
    pub has_vertex_color: bool,
    pub indices: Vec<u32>,
    /// Axis-aligned bounds in the mesh's own space, node transforms not applied.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub material: Option<MaterialData>,
    pub animations: Vec<AnimationData>,
    pub skeleton: Option<SkeletonData>,
}

/// The kind of a `KHR_lights_punctual` light.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightType {
    Directional,
    Spot,
    #[default]
    Point,
}

/// One `KHR_lights_punctual` light definition.
///
/// Scene placement lives on the node that references it ([`ModelNode::light`]);
/// this is only the light's own definition.
#[derive(Clone, Debug)]
pub struct PunctualLight {
    pub light_type: LightType,
    /// Linear-space colour.
    pub color: [f32; 3],
    pub intensity: f32,
    /// Point/spot only; infinite when the file omitted it.
    pub range: f32,
    /// Spot only, radians. Spec defaults 0 and PI/4.
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    pub name: String,
}

impl Default for PunctualLight {
    fn default() -> Self {
        Self {
            light_type: LightType::Point,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: f32::INFINITY,
            inner_cone_angle: 0.0,
            outer_cone_angle: std::f32::consts::FRAC_PI_4,
            name: String::new(),
        }
    }
}

/// One node of the `glTF` scene graph, flattened.
///
/// The editor turns each node into a controllable child entity instead of one
/// opaque model.
#[derive(Clone, Debug)]
pub struct ModelNode {
    pub name: String,
    /// Index into [`ModelData::nodes`], -1 at a root.
    pub parent: i32,
    pub children: Vec<u32>,
    /// The `glTF` mesh index this node draws, -1 when it draws nothing.
    pub mesh: i32,
    /// The entries of the model's mesh list this node instantiates, in
    /// primitive order.
    pub primitives: Vec<u32>,
    pub translation: [f32; 3],
    /// `[x, y, z, w]`, `glTF` order.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    /// Every ancestor's transform composed down to and including this one.
    ///
    /// Deliberately not baked into the vertices: a skinning bone matrix has to
    /// apply on top of it, and rewriting every vertex per instance would defeat
    /// the geometry sharing.
    pub world_transform: [[f32; 4]; 4],
    /// Index into [`ModelData::punctual_lights`] this node places, -1 when none.
    pub light: i32,
}

/// Everything about a loaded model that is not per-primitive geometry.
///
/// The pieces the contract does not put on [`MeshData`] live here so a caller
/// can still reach the node graph, the skins and the list of extensions the
/// loader did not implement.
#[derive(Clone, Debug, Default)]
pub struct ModelData {
    pub nodes: Vec<ModelNode>,
    /// Root node indices, in scene order.
    pub roots: Vec<u32>,
    pub skins: Vec<SkinData>,
    /// `KHR_lights_punctual` definitions; nodes reference them by index.
    pub punctual_lights: Vec<PunctualLight>,
    /// Extensions the file declares that this loader does not honour, by name.
    ///
    /// Reported at load time because a parsed-but-ignored extension makes an
    /// asset render with subtly wrong shading and no error anywhere.
    pub unsupported_extensions: Vec<String>,
}
