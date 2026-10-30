use core::num::NonZeroU32;

use arrayvec::ArrayVec;
use ash::vk;

pub(super) struct MultiviewResolve {
    source: super::TempTextureViewKey,
    destination: super::TempTextureViewKey,
    source_attachment: super::AttachmentKey,
    destination_attachment: super::AttachmentKey,
    extent: wgt::Extent3d,
    samples: u32,
    mask: NonZeroU32,
}

pub(super) struct PreparedResolve {
    pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    area: vk::Rect2D,
}

impl MultiviewResolve {
    pub(super) fn new(
        color: &crate::ColorAttachment<'_, super::TextureView>,
        destination: &crate::Attachment<'_, super::TextureView>,
        extent: wgt::Extent3d,
        samples: u32,
        mask: NonZeroU32,
    ) -> Self {
        let key = |view: &super::TextureView| super::TempTextureViewKey {
            texture: view.raw_texture,
            texture_identity: view.texture_identity,
            format: view.raw_format,
            mip_level: view.base_mip_level,
            depth_slice: view.base_array_layer,
        };
        Self {
            source: key(color.target.view),
            destination: key(destination.view),
            source_attachment: color.target.make_attachment_key(
                crate::AttachmentOps::LOAD
                    | (color.ops
                        & (crate::AttachmentOps::STORE | crate::AttachmentOps::STORE_DISCARD)),
            ),
            destination_attachment: destination.make_attachment_key(
                crate::AttachmentOps::LOAD_DONT_CARE | crate::AttachmentOps::STORE,
            ),
            extent,
            samples,
            mask,
        }
    }
}

impl super::CommandEncoder {
    pub(super) fn prepare_multiview_resolve(
        &mut self,
        resolve: MultiviewResolve,
    ) -> Result<(), crate::DeviceError> {
        let mut views = resolve.mask.get();
        while views != 0 {
            let layer = views.trailing_zeros();
            views &= views - 1;
            let mut source = resolve.source.clone();
            let mut destination = resolve.destination.clone();
            source.depth_slice += layer;
            destination.depth_slice += layer;
            let source = self.make_temp_texture_view(source)?;
            let destination = self.make_temp_texture_view(destination)?;
            let mut colors = ArrayVec::new();
            colors.push(Some(super::ColorAttachmentKey {
                base: resolve.source_attachment.clone(),
                resolve: Some(resolve.destination_attachment.clone()),
            }));
            let pass = self.device.make_render_pass(super::RenderPassKey {
                colors,
                sample_count: resolve.samples,
                ..Default::default()
            })?;
            let mut framebuffer = super::FramebufferKey {
                raw_pass: pass,
                attachment_views: ArrayVec::new(),
                attachment_identities: ArrayVec::new(),
                extent: wgt::Extent3d {
                    depth_or_array_layers: 1,
                    ..resolve.extent
                },
            };
            framebuffer.push_view(source);
            framebuffer.push_view(destination);
            let framebuffer = self.make_framebuffer(framebuffer)?;
            let area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: resolve.extent.width,
                    height: resolve.extent.height,
                },
            };
            self.multiview_resolves.push(PreparedResolve {
                pass,
                framebuffer,
                area,
            });
        }
        Ok(())
    }

    pub(super) fn finish_multiview_resolves(&mut self) {
        if self.multiview_resolves.is_empty() {
            return;
        }
        let dependency = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );
        unsafe {
            self.device.raw.cmd_pipeline_barrier(
                self.active,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[dependency],
                &[],
                &[],
            );
        }
        for resolve in self.multiview_resolves.drain(..) {
            unsafe {
                self.device.raw.cmd_begin_render_pass(
                    self.active,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(resolve.pass)
                        .framebuffer(resolve.framebuffer)
                        .render_area(resolve.area),
                    vk::SubpassContents::INLINE,
                );
                self.device.raw.cmd_end_render_pass(self.active);
            }
        }
        unsafe {
            self.device.raw.cmd_pipeline_barrier(
                self.active,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[dependency],
                &[],
                &[],
            );
        }
    }
}
