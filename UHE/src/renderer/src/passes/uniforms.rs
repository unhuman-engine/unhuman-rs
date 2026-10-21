//! The uniform blocks the shaders read, and the buffers they live in.
//!
//! Push constants are not usable through this RHI: neither `PipelineDesc` nor
//! `ComputePipelineDesc` declares a push-constant range, so Vulkan has nothing
//! to address, and wgpu has no push constants at all. `set_push_constants`
//! reports `Unsupported` on both. The same values travel as uniform buffers
//! bound through a descriptor set.
//!
//! Per-draw uniforms live in one arena: [`DrawArena`] keeps a single buffer and
//! one bind group per offset across frames, refilled through
//! [`chassis::Resources::write_buffer`]. Re-creating the buffer and its sets every
//! frame -- the only option while the RHI had no in-place write -- cost one
//! allocation and one group per draw, which was the per-draw CPU cost. The
//! frame-wide sets ([`BindSet`], [`ImageSet`]) are rebuilt only when their inputs
//! change: a steady frame with identical uniforms and the same bound resources
//! allocates nothing.

use bytemuck::{Pod, Zeroable};
use chassis::{
    BindGroup, Buffer, BufferDesc, BufferUsage, DescriptorSetLayoutDesc, ImageView, Resources,
    Sampler,
};

/// Byte alignment between blocks in one buffer.
///
/// `VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment` is 256 on every
/// desktop driver and 64 on some mobile ones; the RHI does not expose it, so the
/// larger of the two is used rather than a value that would be rejected where
/// the limit is 256.
pub const UNIFORM_ALIGNMENT: u64 = 256;

/// Set 0's uniform: everything that is the same for every draw in the frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub shadow_view_projection: [[f32; 4]; 4],
    pub camera_position: [f32; 4],
    /// `x` = light count, `y` = lighting enabled (1/0), `z`/`w` unused.
    pub params: [f32; 4],
}

impl FrameUniforms {
    pub fn new(
        view_projection: [[f32; 4]; 4],
        shadow_view_projection: [[f32; 4]; 4],
        camera_position: [f32; 3],
        light_count: usize,
        lighting: bool,
    ) -> Self {
        Self {
            view_projection,
            shadow_view_projection,
            camera_position: [
                camera_position[0],
                camera_position[1],
                camera_position[2],
                1.0,
            ],
            params: [
                light_count as f32,
                if lighting { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        }
    }
}

/// Set 0 of the prepass: the light-space matrix the depth is measured in.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ShadowUniforms {
    pub light_view_projection: [[f32; 4]; 4],
}

impl ShadowUniforms {
    pub fn new(light_view_projection: [[f32; 4]; 4]) -> Self {
        Self {
            light_view_projection,
        }
    }
}

/// The composite pass's exposure and display transform inputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CompositeUniforms {
    pub exposure: f32,
    pub _pad: [f32; 3],
}

impl CompositeUniforms {
    pub fn new(exposure: f32) -> Self {
        Self {
            exposure,
            _pad: [0.0; 3],
        }
    }
}

/// Set 2's uniform: everything that changes per draw.
///
/// The bind groups are the addressing, so this carries only the model matrix, the
/// entity id and the tint.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DrawUniforms {
    pub model: [[f32; 4]; 4],
    /// The entity's tint.
    pub tint: [f32; 4],
    /// `x` = entity id, `y` = use vertex colour, `z` = first palette matrix,
    /// `w` = palette matrix count (0 for an unskinned draw).
    pub params: [f32; 4],
}

impl DrawUniforms {
    pub fn new(model: [[f32; 4]; 4], tint: [f32; 4], entity_id: i32) -> Self {
        Self {
            model,
            tint,
            params: [entity_id as f32, 0.0, 0.0, 0.0],
        }
    }

    /// Points the shader at a contiguous range of the frame's palette.
    pub fn with_palette(mut self, first: u32, count: u32) -> Self {
        self.params[2] = first as f32;
        self.params[3] = count as f32;
        self
    }
}

/// The inputs a [`BindSet`] was last built from. Identical inputs mean the
/// existing buffer and group still describe the set, so nothing is rebuilt.
///
/// The uniform bytes are compared, not the values that produced them: two
/// cameras at the same position produce identical matrices and the same set.
#[derive(Clone, PartialEq)]
struct BindSetKey {
    layout: DescriptorSetLayoutDesc,
    bytes: Vec<u8>,
    /// The caller's extra buffers by binding number; the set's own uniform is
    /// binding 0 and is not part of the inputs.
    buffers: Vec<(u32, Buffer, u64)>,
    images: Vec<(u32, ImageView)>,
    samplers: Vec<(u32, Sampler)>,
}

/// One descriptor set whose contents are uploaded as a unit: a uniform block, plus
/// whatever else its layout binds.
///
/// The buffer and group are kept while their inputs are unchanged; a new set is
/// made only when the uniform bytes or a bound resource differ, and then the
/// previous pair is released first.
pub struct BindSet {
    buffer: Option<Buffer>,
    group: Option<BindGroup>,
    capacity: u64,
    /// The inputs the live buffer and group were built from.
    key: Option<BindSetKey>,
}

impl BindSet {
    pub fn new() -> Self {
        Self {
            buffer: None,
            group: None,
            capacity: 0,
            key: None,
        }
    }

    /// Uploads the block and binds it together with `buffers`, `images` and
    /// `samplers`, addressed by binding number.
    ///
    /// A rebuild with the same inputs as the last one returns the existing group
    /// and creates nothing.
    pub fn rebuild<T: Pod>(
        &mut self,
        resources: &mut dyn Resources,
        layout: &DescriptorSetLayoutDesc,
        value: &T,
        buffers: &[(u32, Buffer, u64)],
        images: &[(u32, ImageView)],
        samplers: &[(u32, Sampler)],
    ) -> Result<BindGroup, chassis::Error> {
        let bytes = bytemuck::bytes_of(value);
        let key = BindSetKey {
            layout: layout.clone(),
            bytes: bytes.to_vec(),
            buffers: buffers.to_vec(),
            images: images.to_vec(),
            samplers: samplers.to_vec(),
        };
        if self.key.as_ref() == Some(&key) {
            return Ok(self.group.expect("a cached key has a live group"));
        }

        // A camera move changes the uniform bytes every frame. When the layout
        // and the bound resources are the same, overwrite the uniform block and
        // keep the group, which is what makes a steady frame allocate nothing.
        let reusable = self.group.is_some()
            && self.capacity >= bytes.len() as u64
            && self.key.as_ref().is_some_and(|old| {
                old.layout == key.layout
                    && old.buffers == key.buffers
                    && old.images == key.images
                    && old.samplers == key.samplers
            });
        if reusable {
            let buffer = self.buffer.expect("a reusable key has a live buffer");
            resources.write_buffer(&buffer, 0, bytes)?;
            self.key = Some(key);
            return Ok(self.group.expect("reusable checked the group"));
        }

        self.destroy(resources);

        let buffer = resources.create_buffer(
            BufferDesc::new(
                bytes.len() as u64,
                BufferUsage::UNIFORM | BufferUsage::TRANSFER_DST,
            ),
            Some(bytes),
        )?;
        self.capacity = bytes.len() as u64;

        // The set's own uniform is binding 0, which is what every layout here
        // declares.
        let mut entries = Vec::with_capacity(buffers.len() + 1);
        entries.push((0u32, buffer, 0u64));
        entries.extend_from_slice(buffers);

        let group = resources.create_bind_group(layout, &entries, images, samplers)?;
        self.buffer = Some(buffer);
        self.group = Some(group);
        self.key = Some(key);
        Ok(group)
    }

    pub fn group(&self) -> Option<BindGroup> {
        self.group
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources) {
        if let Some(group) = self.group.take() {
            resources.destroy_bind_group(group);
        }
        if let Some(buffer) = self.buffer.take() {
            resources.destroy_buffer(buffer);
        }
        self.capacity = 0;
        self.key = None;
    }
}

impl Default for BindSet {
    fn default() -> Self {
        Self::new()
    }
}

/// A descriptor set with no uniform block at all: only images and samplers.
///
/// The composite pass is the one that needs this, reading two textures and
/// uploading nothing; it is separate from [`BindSet`] because a `BindSet` always
/// allocates a uniform buffer for binding 0, which a layout without that binding
/// cannot accept.
pub struct ImageSet {
    group: Option<BindGroup>,
    /// The layout and resources the live group was built from.
    key: Option<ImageSetKey>,
}

/// The inputs an [`ImageSet`] was last built from.
#[derive(Clone, PartialEq)]
struct ImageSetKey {
    layout: DescriptorSetLayoutDesc,
    images: Vec<(u32, ImageView)>,
    samplers: Vec<(u32, Sampler)>,
}

impl ImageSet {
    pub fn new() -> Self {
        Self {
            group: None,
            key: None,
        }
    }

    /// A rebuild with the same layout and resources returns the existing group
    /// and creates nothing.
    pub fn rebuild(
        &mut self,
        resources: &mut dyn Resources,
        layout: &DescriptorSetLayoutDesc,
        images: &[(u32, ImageView)],
        samplers: &[(u32, Sampler)],
    ) -> Result<BindGroup, chassis::Error> {
        let key = ImageSetKey {
            layout: layout.clone(),
            images: images.to_vec(),
            samplers: samplers.to_vec(),
        };
        if self.key.as_ref() == Some(&key) {
            return Ok(self.group.expect("a cached key has a live group"));
        }

        self.destroy(resources);
        let group = resources.create_bind_group(layout, &[], images, samplers)?;
        self.group = Some(group);
        self.key = Some(key);
        Ok(group)
    }

    pub fn group(&self) -> Option<BindGroup> {
        self.group
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources) {
        if let Some(group) = self.group.take() {
            resources.destroy_bind_group(group);
        }
        self.key = None;
    }
}

impl Default for ImageSet {
    fn default() -> Self {
        Self::new()
    }
}

/// The frame's skin matrices, bound as one storage buffer.
///
/// A frame with no skinned draws still binds one identity: the pipeline layout
/// has the set, and a zero-sized buffer is not a descriptor a backend accepts.
pub struct PaletteSet {
    buffer: Option<Buffer>,
    capacity: u64,
    group: Option<BindGroup>,
    layout: Option<DescriptorSetLayoutDesc>,
}

impl PaletteSet {
    pub fn new() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            group: None,
            layout: None,
        }
    }

    /// Uploads this frame's palette and returns its group.
    pub fn rebuild(
        &mut self,
        resources: &mut dyn Resources,
        layout: &DescriptorSetLayoutDesc,
        matrices: &[[[f32; 4]; 4]],
    ) -> Result<BindGroup, chassis::Error> {
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let count = matrices.len().max(1);
        let needed = count as u64 * std::mem::size_of::<[[f32; 4]; 4]>() as u64;

        let layout_changed = self.layout.as_ref() != Some(layout);
        let grown = self.buffer.is_none() || self.capacity < needed;
        if layout_changed || grown {
            if let Some(group) = self.group.take() {
                resources.destroy_bind_group(group);
            }
            if let Some(buffer) = self.buffer.take() {
                resources.destroy_buffer(buffer);
            }
            let buffer = resources.create_buffer(
                BufferDesc::new(needed, BufferUsage::STORAGE | BufferUsage::TRANSFER_DST),
                None,
            )?;
            self.buffer = Some(buffer);
            self.capacity = needed;
            self.layout = Some(layout.clone());
        }

        let buffer = self.buffer.expect("created above when absent");
        if matrices.is_empty() {
            resources.write_buffer(&buffer, 0, bytemuck::bytes_of(&identity))?;
        } else {
            resources.write_buffer(&buffer, 0, bytemuck::cast_slice(matrices))?;
        }

        if let Some(group) = self.group {
            return Ok(group);
        }
        let group = resources.create_bind_group(layout, &[(0, buffer, 0)], &[], &[])?;
        self.group = Some(group);
        Ok(group)
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources) {
        if let Some(group) = self.group.take() {
            resources.destroy_bind_group(group);
        }
        if let Some(buffer) = self.buffer.take() {
            resources.destroy_buffer(buffer);
        }
        self.capacity = 0;
        self.layout = None;
    }
}

impl Default for PaletteSet {
    fn default() -> Self {
        Self::new()
    }
}

/// One buffer holding a block per draw, and the group that binds each block.
///
/// The buffer and its groups outlive a frame. Each frame writes every draw's
/// block into the same buffer at the same offset and reuses the group built for
/// that offset, so the per-frame cost is one `write_buffer` rather than a buffer
/// and a bind group per draw. The buffer is rebuilt only when the draw count
/// grows past its capacity or the layout changes; either invalidates every group.
///
/// NOTE: refilling the buffer assumes the previous frame has retired, the same
/// assumption the per-frame destroy-and-recreate it replaced made.
pub struct DrawArena {
    buffer: Option<Buffer>,
    /// Bytes allocated. The buffer is kept until a frame needs more than this.
    capacity: u64,
    /// Byte offset between blocks, or 0 before the first `begin`.
    stride: u64,
    groups: Vec<BindGroup>,
    /// The layout the groups were built from. A different one cannot bind them.
    layout: Option<DescriptorSetLayoutDesc>,
}

impl DrawArena {
    pub fn new() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            stride: 0,
            groups: Vec::new(),
            layout: None,
        }
    }

    /// Refills the arena in place: one block per draw, in order. `group(index)`
    /// then matches `blocks[index]`.
    pub fn begin(
        &mut self,
        resources: &mut dyn Resources,
        layout: &DescriptorSetLayoutDesc,
        blocks: &[Vec<u8>],
    ) -> Result<usize, chassis::Error> {
        if blocks.is_empty() {
            // A frame with nothing to draw has no per-draw blocks, and a buffer
            // of zero bytes is nothing a backend will accept. Drop the groups and
            // keep the buffer for the next non-empty frame.
            self.drop_groups(resources);
            self.stride = 0;
            return Ok(0);
        }

        let stride = align_up(
            blocks
                .iter()
                .map(|block| block.len() as u64)
                .max()
                .unwrap_or(0),
            UNIFORM_ALIGNMENT,
        );
        let needed = stride * blocks.len() as u64;

        // A new buffer, or a layout the cached groups were not built for,
        // invalidates every group; the two are rebuilt together.
        let layout_changed = self.layout.as_ref() != Some(layout);
        let grown = self.buffer.is_none() || self.capacity < needed;
        if layout_changed || grown {
            self.drop_groups(resources);
            if let Some(buffer) = self.buffer.take() {
                resources.destroy_buffer(buffer);
            }
            // `TRANSFER_DST` is what makes the buffer writable in place on both
            // backends: Vulkan gives a `TRANSFER_DST` buffer host-visible memory
            // and wgpu needs `COPY_DST` for `queue.write_buffer`.
            let buffer = resources.create_buffer(
                BufferDesc::new(needed, BufferUsage::UNIFORM | BufferUsage::TRANSFER_DST),
                None,
            )?;
            self.buffer = Some(buffer);
            self.capacity = needed;
            self.layout = Some(layout.clone());
        }

        let buffer = self.buffer.expect("created above when absent");

        // One write for the whole frame. Bytes past `needed`, left over from a
        // larger earlier frame, are stale and unread.
        let mut bytes = vec![0u8; needed as usize];
        for (index, block) in blocks.iter().enumerate() {
            let start = index * stride as usize;
            bytes[start..start + block.len()].copy_from_slice(block);
        }
        resources.write_buffer(&buffer, 0, &bytes)?;

        // Groups are cached by offset: a stride change moves every offset, a
        // count change adds or removes one.
        if self.stride != stride || self.groups.len() != blocks.len() {
            self.drop_groups(resources);
            for index in 0..blocks.len() {
                let offset = index as u64 * stride;
                let group =
                    resources.create_bind_group(layout, &[(0, buffer, offset)], &[], &[])?;
                self.groups.push(group);
            }
        }
        self.stride = stride;
        Ok(self.groups.len())
    }

    fn drop_groups(&mut self, resources: &mut dyn Resources) {
        for group in self.groups.drain(..) {
            resources.destroy_bind_group(group);
        }
    }

    pub fn group(&self, index: usize) -> Option<BindGroup> {
        self.groups.get(index).copied()
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn stride(&self) -> u64 {
        self.stride
    }

    pub fn destroy(&mut self, resources: &mut dyn Resources) {
        self.drop_groups(resources);
        if let Some(buffer) = self.buffer.take() {
            resources.destroy_buffer(buffer);
        }
        self.capacity = 0;
        self.stride = 0;
        self.layout = None;
    }
}

impl Default for DrawArena {
    fn default() -> Self {
        Self::new()
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}
