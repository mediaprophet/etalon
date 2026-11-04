//! Persistent bounded picking snapshot, copy recording, and asynchronous completion.
use crate::picking::{resolve_pick, snapshot_ids, PickResult};
use etalon as wgpu;
use etalon_wasm::readback::{ready_worker, PendingMap, SharedWorker};

pub struct PickingReadback<R> {
    buffer: wgpu::Buffer,
    _reservation: R,
    worker: SharedWorker,
    pending: Option<(u32, u32)>,
    copied: bool,
    mapping: Option<PendingMap>,
    snapshot: Vec<u64>,
    snapshot_count: usize,
}
impl<R> PickingReadback<R> {
    pub fn new(buffer: wgpu::Buffer, reservation: R, max_instances: usize) -> Self {
        Self {
            buffer,
            _reservation: reservation,
            worker: ready_worker(false),
            pending: None,
            copied: false,
            mapping: None,
            snapshot: vec![0; max_instances],
            snapshot_count: 0,
        }
    }
    pub fn queue(&mut self, x: f32, y: f32, width: u32, height: u32) {
        if self.mapping.is_some() {
            return;
        }
        self.pending = Some((
            (x.floor().max(0.0) as u32).min(width.saturating_sub(1)),
            (y.floor().max(0.0) as u32).min(height.saturating_sub(1)),
        ));
        self.copied = false;
    }
    pub fn is_pending(&self) -> bool {
        self.pending.is_some() || self.copied || self.mapping.is_some()
    }
    pub fn needs_recording(&self) -> bool {
        self.pending.is_some()
    }
    pub fn snapshot<T>(&mut self, source: &[T], identity: impl Fn(&T) -> u64) -> bool {
        let Some(count) = snapshot_ids(source, &mut self.snapshot, identity) else {
            self.invalidate(false);
            return false;
        };
        self.snapshot_count = count;
        true
    }
    pub fn invalidate(&mut self, retain_pointer: bool) {
        if !retain_pointer {
            self.pending = None;
        }
        self.mapping = None;
        self.copied = false;
        self.snapshot_count = 0;
        let mut worker = self.worker.lock().unwrap();
        let stamp = worker.completion_stamp();
        if let Some(frame) = stamp.frame.checked_add(1) {
            let _ = worker.set_completion_context(frame, stamp.model);
        } else {
            worker.terminate();
        }
    }
    pub fn notify_device_lost(&mut self) {
        self.invalidate(false);
        self.worker.lock().unwrap().notify_device_lost();
    }
    pub fn record_copy(&mut self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        let Some((x, y)) = self.pending.take() else {
            return;
        };
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.copied = true;
    }
    pub fn poll(&mut self, device: &wgpu::Device) -> Option<PickResult> {
        if !self.copied {
            return None;
        }
        if self.mapping.is_none() {
            self.mapping = PendingMap::begin(self.buffer.slice(..), self.worker.clone()).ok();
            if self.mapping.is_none() {
                self.copied = false;
                return None;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            self.notify_device_lost();
            return None;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let mapped = self.mapping.as_mut()?.try_complete()?;
        self.copied = false;
        let result = if mapped {
            self.buffer
                .slice(..)
                .get_mapped_range()
                .ok()
                .and_then(|view| {
                    view.get(..4)
                        .and_then(|bytes| bytes.try_into().ok())
                        .map(u32::from_le_bytes)
                })
                .map(|raw| resolve_pick(raw, &self.snapshot[..self.snapshot_count]))
        } else {
            None
        };
        self.mapping = None;
        result
    }
}
impl<R> Drop for PickingReadback<R> {
    fn drop(&mut self) {
        self.mapping = None;
        self.worker.lock().unwrap().terminate();
    }
}

pub fn record_pass(
    encoder: &mut wgpu::CommandEncoder,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    draw: impl FnOnce(&mut wgpu::RenderPass<'_>),
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("portal-picking-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        occlusion_query_set: None,
        multiview_mask: None,
        timestamp_writes: None,
    });
    draw(&mut pass);
}
