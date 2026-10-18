//! Materials: the CPU description, the GPU layout, and the caches.
//!
//! Two caches sit here: textures keyed by asset, and material bind groups keyed
//! by a [`MaterialKey`].
//!
//! ## What is not carried, and why
//!
//! A descriptor *array* is not expressible in this RHI: `create_bind_group` takes
//! images and samplers keyed by binding number with no array index. So this pass
//! binds the five maps it actually shades (albedo, metallic/roughness, normal,
//! occlusion, emissive) as explicit bindings, and carries the two extension
//! values those maps' shading depends on (`emissiveStrength`, `unlit`). The
//! remaining Tier 2/3 factors (clearcoat, sheen, ior, transmission, iridescence,
//! anisotropy, diffuse transmission, volume) have no shader to read them here.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use chassis::{
    BindGroup, Buffer, BufferDesc, BufferUsage, DescriptorSetLayoutDesc, Extent, ImageView,
    Resources, Sampler, SamplerDesc, Texture, TextureDesc, TextureFormat, TextureUsage,
};
use vault::AssetId;

/// The material texture slots this pass samples. The numbers are the shader's
/// slot indices, kept from a wider glTF slot table even though only five are used.
pub const SLOT_ALBEDO: usize = 0;
pub const SLOT_METALLIC_ROUGHNESS: usize = 1;
pub const SLOT_NORMAL: usize = 2;
pub const SLOT_OCCLUSION: usize = 3;
pub const SLOT_EMISSIVE: usize = 4;
pub const SLOT_COUNT: usize = 5;

/// glTF `alphaMode`: the discriminator the shader reads out of `params.y`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

impl AlphaMode {
    pub fn as_f32(self) -> f32 {
        match self {
            AlphaMode::Opaque => 0.0,
            AlphaMode::Mask => 1.0,
            AlphaMode::Blend => 2.0,
        }
    }
}

/// `KHR_texture_transform`, already in the engine's UV space.
///
/// The loader flips V on import, so the shader cannot apply glTF's raw
/// offset/rotation on top; the values stored here are engine-space, and the shader
/// computes `uv' = R(rot) * (uv * scale) + offset`. Identity is `(1, 0, 1, 1, 0, 0)`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct UVTransform {
    pub cos_rotation: f32,
    pub sin_rotation: f32,
    pub scale: [f32; 2],
    pub offset: [f32; 2],
}

impl UVTransform {
    pub const IDENTITY: Self = Self {
        cos_rotation: 1.0,
        sin_rotation: 0.0,
        scale: [1.0, 1.0],
        offset: [0.0, 0.0],
    };
}

impl Default for UVTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A material, as the loader hands it over.
///
/// Defaults are the glTF spec's, not zero: a zero `normalScale` silently disables
/// every normal map, and a zero `occlusionStrength` every occlusion map.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Material {
    pub albedo: Option<AssetId>,
    pub metallic_roughness: Option<AssetId>,
    pub normal: Option<AssetId>,
    pub occlusion: Option<AssetId>,
    pub emissive: Option<AssetId>,

    /// `baseColorFactor`: opaque white when absent.
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    /// `normalTexture.scale`. 1.0, not 0.0.
    pub normal_scale: f32,
    /// `occlusionTexture.strength`. 1.0.
    pub occlusion_strength: f32,
    /// BLACK when absent: defaulting to white makes every surface glow.
    pub emissive_factor: [f32; 3],
    /// `KHR_materials_emissive_strength`. 1.0, the spec default.
    pub emissive_strength: f32,

    pub alpha: AlphaMode,
    /// 0.5, the spec default.
    pub alpha_cutoff: f32,
    /// Selects the cull variant; cull mode is pipeline state.
    pub double_sided: bool,
    /// `KHR_materials_unlit`: output base colour and shade nothing.
    pub unlit: bool,

    pub uv_transforms: [UVTransform; SLOT_COUNT],
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: None,
            metallic_roughness: None,
            normal: None,
            occlusion: None,
            emissive: None,
            base_color_factor: [1.0; 4],
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            emissive_factor: [0.0; 3],
            emissive_strength: 1.0,
            alpha: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
            uv_transforms: [UVTransform::IDENTITY; SLOT_COUNT],
        }
    }
}

impl Material {
    /// The five texture slots, in slot order.
    pub fn texture(&self, slot: usize) -> Option<AssetId> {
        match slot {
            SLOT_ALBEDO => self.albedo,
            SLOT_METALLIC_ROUGHNESS => self.metallic_roughness,
            SLOT_NORMAL => self.normal,
            SLOT_OCCLUSION => self.occlusion,
            SLOT_EMISSIVE => self.emissive,
            _ => None,
        }
    }
}

/// The material parameters, as the shader's uniform block reads them.
///
/// Field order is the layout contract with `model.wgsl`; every member is a `vec4`
/// so std140 and the shader's `uniform` layout cannot disagree. `size_of` is
/// asserted below rather than trusted.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct MaterialGpu {
    pub base_color_factor: [f32; 4],
    /// `rgb` emissive, `a` emissive strength.
    pub emissive_factor_strength: [f32; 4],
    /// metallic, roughness, normal scale, occlusion strength.
    pub factors: [f32; 4],
    /// alpha cutoff, alpha mode, unlit, pad.
    pub params: [f32; 4],
    /// Per slot: `(cos, sin, scale x, scale y)`.
    pub uv_transform: [[f32; 4]; SLOT_COUNT],
    /// Per slot: `(offset x, offset y, -, -)`.
    pub uv_offset: [[f32; 4]; SLOT_COUNT],
}

const _: () = assert!(std::mem::size_of::<MaterialGpu>().is_multiple_of(16));

/// Packs a material into the GPU layout, for the slots and factors this pass
/// shades.
pub fn fill_gpu(material: &Material) -> MaterialGpu {
    let mut gpu = MaterialGpu::zeroed();

    gpu.base_color_factor = material.base_color_factor;
    gpu.emissive_factor_strength = [
        material.emissive_factor[0],
        material.emissive_factor[1],
        material.emissive_factor[2],
        material.emissive_strength,
    ];
    gpu.factors = [
        material.metallic_factor,
        material.roughness_factor,
        material.normal_scale,
        material.occlusion_strength,
    ];
    gpu.params = [
        material.alpha_cutoff,
        material.alpha.as_f32(),
        if material.unlit { 1.0 } else { 0.0 },
        0.0,
    ];

    for slot in 0..SLOT_COUNT {
        let transform = material.uv_transforms[slot];
        // Each slot carries its own transform, so KHR_texture_transform applies
        // per map rather than per material.
        gpu.uv_transform[slot] = [
            transform.cos_rotation,
            transform.sin_rotation,
            transform.scale[0],
            transform.scale[1],
        ];
        gpu.uv_offset[slot] = [transform.offset[0], transform.offset[1], 0.0, 0.0];
    }

    gpu
}

/// Which shader variant draws a material.
///
/// The pipeline is where cull mode and blending live, so they are part of the key
/// rather than state a draw can change.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum MaterialVariant {
    /// Opaque, back-face culled.
    Opaque,
    /// `doubleSided`: identical but for culling.
    OpaqueDoubleSided,
    /// `alphaMode: Blend`, drawn after the opaques.
    Transparent,
    /// Blended and double-sided.
    TransparentDoubleSided,
}

impl MaterialVariant {
    pub fn of(material: &Material, transparent: bool) -> Self {
        match (transparent, material.double_sided) {
            (false, false) => Self::Opaque,
            (false, true) => Self::OpaqueDoubleSided,
            (true, false) => Self::Transparent,
            (true, true) => Self::TransparentDoubleSided,
        }
    }

    pub fn blends(self) -> bool {
        matches!(self, Self::Transparent | Self::TransparentDoubleSided)
    }

    pub fn double_sided(self) -> bool {
        matches!(self, Self::OpaqueDoubleSided | Self::TransparentDoubleSided)
    }
}

/// The five maps a material samples, resolved to GPU handles.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MaterialTextures {
    pub albedo: Option<Texture>,
    pub metallic_roughness: Option<Texture>,
    pub normal: Option<Texture>,
    pub occlusion: Option<Texture>,
    pub emissive: Option<Texture>,
}

impl MaterialTextures {
    pub fn get(&self, slot: usize) -> Option<Texture> {
        match slot {
            SLOT_ALBEDO => self.albedo,
            SLOT_METALLIC_ROUGHNESS => self.metallic_roughness,
            SLOT_NORMAL => self.normal,
            SLOT_OCCLUSION => self.occlusion,
            SLOT_EMISSIVE => self.emissive,
            _ => None,
        }
    }
}

/// What a material is cached under: the textures it binds and the values the
/// shader reads.
///
/// Floats are keyed by bit pattern, so two materials that differ in the last
/// mantissa bit are two entries. That is the correct rule for a cache; an epsilon
/// comparison would make the key depend on the caller's tolerance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MaterialKey {
    pub variant: MaterialVariant,
    pub textures: MaterialTextures,
    pub base_color_factor: [u32; 4],
    pub emissive_factor_strength: [u32; 4],
    pub factors: [u32; 4],
    pub params: [u32; 4],
}

impl MaterialKey {
    pub fn of(material: &Material, textures: MaterialTextures, variant: MaterialVariant) -> Self {
        let gpu = fill_gpu(material);
        Self {
            variant,
            textures,
            base_color_factor: gpu.base_color_factor.map(f32::to_bits),
            emissive_factor_strength: gpu.emissive_factor_strength.map(f32::to_bits),
            factors: gpu.factors.map(f32::to_bits),
            params: gpu.params.map(f32::to_bits),
        }
    }
}

/// A material resident on the GPU: where its parameters live and the set that
/// binds them.
#[derive(Clone, Copy, Debug)]
pub struct MaterialBinding {
    pub slot: u32,
    pub group: BindGroup,
}

struct MaterialEntry {
    binding: MaterialBinding,
    params: Buffer,
    textures: MaterialTextures,
}

/// Bind groups and parameter buffers, keyed by [`MaterialKey`].
pub struct MaterialCache {
    entries: HashMap<MaterialKey, MaterialEntry>,
    next_slot: u32,
}

impl MaterialCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            next_slot: 0,
        }
    }

    pub fn get(&self, key: &MaterialKey) -> Option<MaterialBinding> {
        self.entries.get(key).map(|entry| entry.binding)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The binding for this key, creating its parameter buffer and bind group
    /// the first time it is asked for.
    ///
    /// `images` are the resolved views to sample, `(binding, view)`, and must be
    /// the views behind `key.textures`: they are what the entry is invalidated
    /// by when a texture is destroyed. `samplers` is the one sampler every map
    /// in the set is read with.
    pub fn get_or_insert(
        &mut self,
        resources: &mut dyn Resources,
        key: MaterialKey,
        material: &Material,
        images: &[(u32, ImageView)],
        samplers: &[(u32, Sampler)],
        layout: &DescriptorSetLayoutDesc,
    ) -> Result<MaterialBinding, chassis::Error> {
        if let Some(entry) = self.entries.get(&key) {
            return Ok(entry.binding);
        }

        let gpu = fill_gpu(material);
        let params = resources.create_buffer(
            BufferDesc::new(
                std::mem::size_of::<MaterialGpu>() as u64,
                BufferUsage::UNIFORM,
            ),
            Some(bytemuck::bytes_of(&gpu)),
        )?;

        let group = resources.create_bind_group(layout, &[(0, params, 0)], images, samplers)?;

        let binding = MaterialBinding {
            slot: self.next_slot,
            group,
        };
        self.next_slot += 1;
        self.entries.insert(
            key,
            MaterialEntry {
                binding,
                params,
                textures: key.textures,
            },
        );
        Ok(binding)
    }

    /// Drops every material that binds `texture`.
    ///
    /// The RHI destroys a texture, so a bind group still naming it is a
    /// descriptor pointing at freed memory. The entry has to go with the
    /// texture, and returning how many went lets a caller log a surprising
    /// number rather than guess.
    pub fn invalidate_texture(&mut self, resources: &mut dyn Resources, texture: Texture) -> usize {
        let doomed: Vec<MaterialKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                (0..SLOT_COUNT).any(|slot| entry.textures.get(slot) == Some(texture))
            })
            .map(|(key, _)| *key)
            .collect();

        for key in &doomed {
            self.destroy(resources, key);
        }
        doomed.len()
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources, key: &MaterialKey) -> bool {
        let Some(entry) = self.entries.remove(key) else {
            return false;
        };
        resources.destroy_bind_group(entry.binding.group);
        resources.destroy_buffer(entry.params);
        true
    }

    pub fn clear(&mut self, resources: &mut dyn Resources) {
        let keys: Vec<MaterialKey> = self.entries.keys().copied().collect();
        for key in &keys {
            self.destroy(resources, key);
        }
    }
}

impl Default for MaterialCache {
    fn default() -> Self {
        Self::new()
    }
}

/// A texture resident on the GPU.
#[derive(Clone, Copy, Debug)]
pub struct CachedTexture {
    pub texture: Texture,
    pub view: ImageView,
    pub sampler: Sampler,
    pub width: u32,
    pub height: u32,
}

/// Textures, keyed by asset.
///
/// One shared sampler per filter mode, because the RHI's samplers are separate
/// objects and every material map wants the same one.
pub struct TextureCache {
    entries: HashMap<AssetId, CachedTexture>,
    linear: Option<Sampler>,
    white: Option<CachedTexture>,
    neutral: Option<CachedTexture>,
}

impl TextureCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            linear: None,
            white: None,
            neutral: None,
        }
    }

    pub fn get(&self, id: AssetId) -> Option<CachedTexture> {
        self.entries.get(&id).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn sampler(&mut self, resources: &mut dyn Resources) -> Result<Sampler, chassis::Error> {
        if let Some(sampler) = self.linear {
            return Ok(sampler);
        }
        // Trilinear with repeat: glTF's default wrap is REPEAT.
        let sampler = resources.create_sampler(&SamplerDesc::trilinear())?;
        self.linear = Some(sampler);
        Ok(sampler)
    }

    /// The 1x1 texture a material with no map of a given kind samples: white.
    ///
    /// Multiplying by white is the correct neutral for albedo, metallic/roughness,
    /// occlusion and emissive; glTF says an absent map leaves the factor in force,
    /// which is what sampling 1.0 does.
    pub fn white(
        &mut self,
        resources: &mut dyn Resources,
    ) -> Result<CachedTexture, chassis::Error> {
        if let Some(white) = self.white {
            return Ok(white);
        }
        let image = vault::Image {
            width: 1,
            height: 1,
            rgba8: vec![255, 255, 255, 255],
            srgb: true,
        };
        let cached = self.upload(resources, &image)?;
        self.white = Some(cached);
        Ok(cached)
    }

    /// The 1x1 texture an absent *normal* map samples: `(0.5, 0.5, 1)`.
    ///
    /// Not white: `(1, 1, 1)` decodes to a tangent-space normal tilted 45 degrees
    /// in both axes, so binding white for a missing normal map tilts every normal
    /// on that surface. Linear, because a normal map is data and not colour; an
    /// sRGB view would decode `0.5` to `0.21`.
    pub fn neutral_normal(
        &mut self,
        resources: &mut dyn Resources,
    ) -> Result<CachedTexture, chassis::Error> {
        if let Some(neutral) = self.neutral {
            return Ok(neutral);
        }
        let image = vault::Image {
            width: 1,
            height: 1,
            rgba8: vec![128, 128, 255, 255],
            srgb: false,
        };
        let cached = self.upload(resources, &image)?;
        self.neutral = Some(cached);
        Ok(cached)
    }

    /// Uploads an image. `MeshSample`'s material ids resolve through
    /// [`TextureCache::get`]; this is the call that puts one there.
    ///
    /// The bytes go through [`Resources::upload_texture`] rather than
    /// `TextureDesc::initial_data`, because `stormglass` ignores that field: a
    /// texture created with it would be empty on the default backend, and every
    /// material would sample black on the backend that works.
    pub fn upload(
        &mut self,
        resources: &mut dyn Resources,
        image: &vault::Image,
    ) -> Result<CachedTexture, chassis::Error> {
        let format = if image.srgb {
            TextureFormat::Rgba8Srgb
        } else {
            TextureFormat::Rgba8Unorm
        };
        let desc = TextureDesc::new_2d(
            Extent::new(image.width, image.height),
            format,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        )
        .with_mips(1);

        let texture = resources.create_texture(desc)?;
        // `upload_texture`, not `TextureDesc::initial_data`: `stormglass` never
        // reads that field, so a texture created with it holds nothing on the
        // default backend. This is the path both backends implement.
        //
        // Queued, not finished: destroying the texture before the copy drains is
        // illegal, so teardown waits at frame end.
        resources.upload_texture(&texture, &image.rgba8, image.width, image.height)?;
        let view = resources.create_image_view(&texture)?;
        let sampler = self.sampler(resources)?;
        Ok(CachedTexture {
            texture,
            view,
            sampler,
            width: image.width,
            height: image.height,
        })
    }

    pub fn get_or_upload(
        &mut self,
        resources: &mut dyn Resources,
        id: AssetId,
        image: &vault::Image,
    ) -> Result<CachedTexture, chassis::Error> {
        if let Some(cached) = self.entries.get(&id) {
            return Ok(*cached);
        }
        let cached = self.upload(resources, image)?;
        self.entries.insert(id, cached);
        Ok(cached)
    }

    /// Destroys a texture and forgets it. The returned handle is what a
    /// material cache is invalidated with.
    pub fn destroy(&mut self, resources: &mut dyn Resources, id: AssetId) -> Option<Texture> {
        let cached = self.entries.remove(&id)?;
        resources.destroy_texture(cached.texture);
        Some(cached.texture)
    }

    pub fn clear(&mut self, resources: &mut dyn Resources) {
        for (_, cached) in self.entries.drain() {
            resources.destroy_texture(cached.texture);
        }
        for cached in [self.white.take(), self.neutral.take()]
            .into_iter()
            .flatten()
        {
            resources.destroy_texture(cached.texture);
        }
        if let Some(sampler) = self.linear.take() {
            resources.destroy_sampler(sampler);
        }
    }
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::new()
    }
}
