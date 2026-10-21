//! Command recording for the wgpu backend.
//!
//! ## The lifetime problem, and the answer wgpu already provides
//!
//! `begin_render_pass` borrows the `CommandEncoder` for as long as the
//! `RenderPassEncoder` lives, so a struct holding both cannot hold them at the
//! same time. Three ways out, and only one of them is good:
//!
//! 1. A lifetime transmute. `unsafe`, and wrong.
//! 2. Buffer the commands and replay. Safe, but a second pass over every command
//!    and a second resolution of every handle, for nothing.
//! 3. [`wgpu::RenderPassEncoder::forget_lifetime`].
//!
//! (3) is the one. It is **safe**, and wgpu's own documentation says why: the
//! encoder is locked while a pass is open, whether or not the lifetime says so,
//! so touching it is a *runtime* error rather than undefined behaviour. The
//! borrow checker stops helping here and the driver takes over -- which is the
//! right trade, because the rule is one line ("do not touch the encoder while a
//! pass is open") and wgpu enforces it in the one place that can.
//!
//! So this is immediate recording, like the Vulkan backend: no buffering, no
//! replay, no `unsafe`.
//!
//! ## The invariant
//!
//! While [`Recording::pass`] is `Some`, the encoder is locked and must not be
//! touched. Every method below either goes to the open pass or to the encoder,
//! never both, and the test `a_draw_never_touches_the_encoder_while_a_pass_is_open`
//! pins it.

use std::borrow::Cow;

use chassis::CommandList;

use crate::wgpu_resources::WgpuResources;

/// One recorded command list. One per render-graph pass.
pub struct Recording {
    encoder: wgpu::CommandEncoder,
    /// `Some` between `begin_rendering` and `end_rendering`.
    ///
    /// `'static` from `forget_lifetime`, which is safe because the encoder is
    /// locked behind it and wgpu errors at runtime on a violation rather than
    /// letting it be undefined.
    pass: Option<wgpu::RenderPass<'static>>,
    /// `Some` between `begin_compute` and `end_compute`.
    ///
    /// wgpu will not set a compute pipeline or bind group outside a compute
    /// pass, so the scope is what makes a dispatch possible at all.
    compute: Option<wgpu::ComputePass<'static>>,
    /// The pipeline's layouts, so a caller that inspects them sees the same
    /// shape on both backends. wgpu has one flat push-constant block rather than
    /// Vulkan's ranged list, and this backend has none, so nothing here splits
    /// one.
    pipeline_layouts: Vec<wgpu::BindGroupLayout>,
    current_pipeline: Option<chassis::Pipeline>,
    commands: u32,
}

impl Recording {
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        Self {
            encoder: device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) }),
            pass: None,
            compute: None,
            pipeline_layouts: Vec::new(),
            current_pipeline: None,
            commands: 0,
        }
    }

    /// The pass's layouts, from the bound pipeline. Empty when none is bound.
    pub fn bind_group_layouts(&self) -> &[wgpu::BindGroupLayout] {
        &self.pipeline_layouts
    }

    pub fn command_count(&self) -> u32 {
        self.commands
    }

    pub fn is_inside_render_pass(&self) -> bool {
        self.pass.is_some()
    }

    pub fn is_inside_compute_pass(&self) -> bool {
        self.compute.is_some()
    }

    /// Finishes the encoder and hands back the command buffer.
    ///
    /// Ending an open pass first. Dropping the pass would end it too, but doing
    /// it here means an unbalanced list is a bug in this type rather than a
    /// silent truncation of the pass.
    pub fn finish(mut self) -> wgpu::CommandBuffer {
        self.pass = None;
        self.compute = None;
        self.encoder.finish()
    }

    /// A view of the encoder, for the replay of a wgpu backend that keeps one.
    fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.encoder
    }

    fn begin(&mut self) -> &mut wgpu::RenderPass<'static> {
        self.pass
            .as_mut()
            .expect("a draw outside a render pass is a bug in the caller")
    }
}

/// `chassis::CommandList` over a [`Recording`].
///
/// The facade: generational handles in, wgpu objects resolved here. Mirrors
/// `VulkanCommandBuffer.rs`, where `Recording` is the raw thing and `CommandList`
/// is the RHI-shaped wrapper over it.
pub struct WgpuCommandList<'a> {
    recording: &'a mut Recording,
    resources: &'a WgpuResources,
}

impl<'a> WgpuCommandList<'a> {
    pub fn new(recording: &'a mut Recording, resources: &'a WgpuResources) -> Self {
        Self {
            recording,
            resources,
        }
    }

    pub fn recording(&mut self) -> &mut Recording {
        self.recording
    }
}

impl CommandList for WgpuCommandList<'_> {
    fn begin_rendering(&mut self, desc: &chassis::RenderPassDesc) {
        let mut colors = Vec::with_capacity(desc.colors.len());
        let mut resolve = None;
        for color in &desc.colors {
            let Some(texture) = self.resources.get_texture(color.texture) else {
                continue;
            };
            colors.push(Some(wgpu::RenderPassColorAttachment {
                view: texture.view(),
                // A resolve target belongs to the *last* colour slot, and only
                // one attachment may name it, so it is filled in below.
                resolve_target: None,
                // `None` is the whole array. wgpu 30 added 2D-array layering, and
                // leaving it out means "no layer", which is a colour target in
                // layer 0 -- correct, and worth saying rather than relying on the
                // field not existing.
                depth_slice: None,
                ops: wgpu::Operations {
                    load: match color.load {
                        chassis::LoadOp::Clear => {
                            let [r, g, b, a] = color.clear;
                            wgpu::LoadOp::<wgpu::Color>::Clear(wgpu::Color {
                                r: r as f64,
                                g: g as f64,
                                b: b as f64,
                                a: a as f64,
                            })
                        }
                        chassis::LoadOp::Load => wgpu::LoadOp::<wgpu::Color>::Load,
                        // `DontCare` maps to `Load`, not to wgpu's `DontCare`.
                        //
                        // wgpu makes `LoadOpDontCare` constructible only inside an
                        // `unsafe fn`, and its documentation says why: reading
                        // undefined attachment contents is undefined behaviour, as
                        // is blending against them. `chassis::LoadOp::DontCare` is
                        // a *safe* variant, so forwarding it would launder an
                        // unsafe operation behind a safe one -- invisible at the
                        // call site, and the worst of both.
                        //
                        // `Load` is the safe superset. On a tiler it costs a load
                        // that could have been skipped, which is real, and the
                        // alternative is an unsafe in a backend that does not
                        // otherwise have one.
                        chassis::LoadOp::DontCare => wgpu::LoadOp::<wgpu::Color>::Load,
                    },
                    store: wgpu::StoreOp::Store,
                },
            }));
            if !color.resolve.is_null() {
                resolve = self
                    .resources
                    .get_texture(color.resolve)
                    .map(|t| t.view().clone());
            }
        }
        // The resolve view goes on the *last* colour slot, and only one
        // attachment may name it. It is bound here -- before the pass is begun --
        // so it outlives the descriptor that borrows it.
        if let Some(resolve) = resolve.as_ref() {
            if let Some(Some(last)) = colors.last_mut() {
                last.resolve_target = Some(resolve);
            }
        }

        let depth_attachment = desc
            .depth
            .as_ref()
            .and_then(|d| {
                self.resources
                    .get_texture(d.texture)
                    .map(|texture| (d, texture))
            })
            .map(|(d, texture)| wgpu::RenderPassDepthStencilAttachment {
                view: texture.view(),
                // `LoadOp<f32>`, not `LoadOp<Color>`: depth loads a number and
                // a colour attachment loads a colour, and wgpu's generic is what
                // makes that a compile error rather than a silent mistake.
                depth_ops: Some(wgpu::Operations {
                    load: match d.load {
                        chassis::LoadOp::Clear => wgpu::LoadOp::<f32>::Clear(d.clear_depth),
                        chassis::LoadOp::Load => wgpu::LoadOp::<f32>::Load,
                        // Same reasoning as the colour arm: wgpu's `DontCare` is
                        // unsafe to construct, and `Load` is the safe superset.
                        chassis::LoadOp::DontCare => wgpu::LoadOp::<f32>::Load,
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            });

        let pass = self
            .recording
            .encoder()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&desc.label),
                color_attachments: &colors,
                // Filled on the last colour slot, after the views exist.
                // (The resolve view is bound above and borrowed here.)
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                // No multiview. The engine is not a VR renderer, and a multiview
                // mask makes wgpu take a code path nothing here has measured.
                multiview_mask: None,
            });
        // Safe: the encoder is locked behind the pass, and wgpu turns a
        // violation into a runtime error rather than undefined behaviour.
        self.recording.pass = Some(pass.forget_lifetime());
        self.recording.commands += 1;
    }

    fn end_rendering(&mut self) {
        if self.recording.pass.take().is_some() {
            // Dropping ends the pass. Nothing else to do: wgpu submits the
            // attachment operations from the descriptor above.
            self.recording.commands += 1;
        }
    }

    fn begin_compute(&mut self) {
        debug_assert!(
            self.recording.pass.is_none(),
            "a render pass cannot be open when a compute pass begins",
        );
        let pass = self
            .recording
            .encoder()
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute"),
                // Timestamp queries are not requested by this backend, and a
                // write against a set that does not exist is a validation error.
                timestamp_writes: None,
            });
        // Safe for the same reason as the render pass: the encoder is locked
        // while a pass is open, and wgpu reports a violation at runtime.
        self.recording.compute = Some(pass.forget_lifetime());
        self.recording.commands += 1;
    }

    fn end_compute(&mut self) {
        if self.recording.compute.take().is_some() {
            // Dropping ends the pass.
            self.recording.commands += 1;
        }
    }

    fn set_viewport(&mut self, extent: chassis::Extent2d) {
        // Goes to the open pass when there is one. Dynamic state in wgpu is set
        // on the pass, not the encoder, so this is the only correct target.
        if self.recording.pass.is_some() {
            self.recording.begin().set_viewport(
                0.0,
                0.0,
                extent.width as f32,
                extent.height as f32,
                0.0,
                1.0,
            );
        }
    }

    fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32) {
        if self.recording.pass.is_some() {
            self.recording.begin().set_scissor_rect(x, y, width, height);
        }
    }

    fn bind_pipeline(&mut self, pipeline: &chassis::Pipeline) {
        let Some(resource) = self.resources.get_pipeline(*pipeline) else {
            return;
        };
        self.recording.pipeline_layouts = resource.bind_group_layouts().to_vec();
        self.recording.current_pipeline = Some(*pipeline);

        if let Some(compute) = resource.as_compute() {
            let Some(pass) = self.recording.compute.as_mut() else {
                panic!(
                    "bind_pipeline on compute pipeline {:?} outside a compute pass; \
                     call begin_compute first",
                    pipeline
                );
            };
            pass.set_pipeline(compute);
        } else if let Some(render) = resource.as_render() {
            if self.recording.pass.is_some() {
                self.recording.begin().set_pipeline(render);
            }
        }
    }

    fn bind_group(&mut self, set: u32, group: &chassis::BindGroup) {
        let Some(bind_group) = self.resources.bind_groups.get(*group) else {
            // Same rule as a stale buffer handle: a bind against a destroyed
            // group is a caller bug, and dropping the bind would draw with
            // whatever the last pass left bound -- which looks exactly like a
            // shader bug and is not one.
            panic!("bind on a stale bind group handle {group:?}");
        };
        // No pipeline layout argument: wgpu's `set_bind_group` takes the index
        // and resolves the layout against the bound pipeline, and rejects a
        // group whose layout was not declared there. That check is why the shared
        // layout cache matters -- see `WgpuResources::bind_group_layout`.
        // `Some(...)` rather than a bare reference: wgpu's dynamic-offset
        // overload takes `Option<&BindGroup>`, where `None` means "use the
        // fallback bind group". Passing the fallback would bind nothing and look
        // like it worked.
        if let Some(pass) = self.recording.compute.as_mut() {
            pass.set_bind_group(set, Some(bind_group), &[]);
        } else {
            self.recording
                .begin()
                .set_bind_group(set, Some(bind_group), &[]);
        }
    }

    fn set_push_constants(&mut self, _data: &[u8]) -> chassis::Result<()> {
        // wgpu has no push constants at all. Its pipeline layout has an
        // `immediate_size`, but a shader cannot declare a push-constant block
        // and the RHI has no range to fill. Reporting the alternative beats a
        // panic and beats a no-op that silently drops the caller's values.
        Err(chassis::Error::Unsupported(
            "wgpu has no push constants; pass the values in a small uniform buffer \
             through a bind group"
                .to_owned(),
        ))
    }

    fn bind_vertex_buffer(&mut self, slot: u32, buffer: &chassis::Buffer, offset: u64) {
        let Some(resource) = self.resources.get_buffer(*buffer) else {
            return;
        };
        if self.recording.pass.is_some() {
            self.recording
                .begin()
                .set_vertex_buffer(slot, resource.inner().slice(offset..));
        }
    }

    fn bind_index_buffer(&mut self, buffer: &chassis::Buffer, offset: u64) {
        let Some(resource) = self.resources.get_buffer(*buffer) else {
            return;
        };
        if self.recording.pass.is_some() {
            self.recording
                .begin()
                .set_index_buffer(resource.inner().slice(offset..), wgpu::IndexFormat::Uint32);
        }
    }

    fn draw(&mut self, vertices: u32, instances: u32) {
        if self.recording.pass.is_some() {
            self.recording.begin().draw(0..vertices, 0..instances);
        }
    }

    fn draw_indexed(&mut self, indices: u32, instances: u32) {
        if self.recording.pass.is_some() {
            self.recording
                .begin()
                .draw_indexed(0..indices, 0, 0..instances);
        }
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let Some(pass) = self.recording.compute.as_mut() else {
            // A dispatch outside a compute pass is a validation error in wgpu,
            // and the message would name the encoder rather than the pass that
            // forgot its scope.
            panic!("dispatch outside a compute pass; call begin_compute first");
        };
        pass.dispatch_workgroups(x, y, z);
        self.recording.commands += 1;
    }

    fn copy_buffer(&mut self, src: &chassis::Buffer, dst: &chassis::Buffer) {
        let (Some(src), Some(dst)) = (
            self.resources.get_buffer(*src),
            self.resources.get_buffer(*dst),
        ) else {
            return;
        };
        self.recording
            .encoder()
            .copy_buffer_to_buffer(src.inner(), 0, dst.inner(), 0, 1);
    }

    fn copy_texture_to_buffer(&mut self, src: &chassis::Texture, dst: &chassis::Buffer) {
        let (Some(texture), Some(buffer)) = (
            self.resources.get_texture(*src),
            self.resources.get_buffer(*dst),
        ) else {
            return;
        };
        let extent = texture.extent();
        // Three arguments in wgpu 30: source info, a destination *info* that
        // carries the layout, and the size.
        self.recording.encoder().copy_texture_to_buffer(
            texture.inner().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: buffer.inner(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // The aligned pitch, not a tightly packed row: wgpu refuses a
                    // `bytes_per_row` that is not a multiple of 256, which is
                    // every width that is not a multiple of 64 texels. The
                    // caller that reads the buffer back strips the padding.
                    bytes_per_row: Some(chassis::copy_row_pitch(extent.width, 4)),
                    rows_per_image: Some(extent.height),
                },
            },
            wgpu::Extent3d {
                width: extent.width,
                height: extent.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// wgpu's label type, which is a `Cow`. Named so the descriptors read clearly.
pub type Label<'a> = Cow<'a, str>;
