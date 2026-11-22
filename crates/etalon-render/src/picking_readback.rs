//! Persistent bounded picking snapshot, copy recording, and asynchronous completion.
use crate::picking::{decode_mesh_slot, resolve_pick, snapshot_ids, PickResult};
use core::mem::size_of;
use etalon_wasm::readback::{ready_worker, PendingMap, SharedWorker};

pub struct PickingReadback<R> {
    buffer: etalon::Buffer,
    _reservation: R,
    worker: SharedWorker,
    pending: Option<PickWindow>,
    copied: Option<PickWindow>,
    mapping: Option<PendingMap>,
    snapshot: Vec<u64>,
    source_slots: Vec<u32>,
    source_regions: Vec<u8>,
    snapshot_count: usize,
}

pub const MAX_MESH_PICK_RADIUS: u8 = 3;
pub const MESH_REGION_PICK_STAGING_BYTES: u64 =
    (2 * MAX_MESH_PICK_RADIUS as u64 + 1) * etalon::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
const ROW_PITCH: usize = etalon::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshRegionPickError {
    InvalidInput,
    Busy,
    BufferTooSmall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PickWindow {
    origin: [u32; 2],
    extent: [u32; 2],
    center: [u32; 2],
    region: Option<(u8, u8)>,
}

impl PickWindow {
    fn single(x: u32, y: u32) -> Self {
        Self {
            origin: [x, y],
            extent: [1, 1],
            center: [0, 0],
            region: None,
        }
    }

    fn around(x: u32, y: u32, width: u32, height: u32, radius: u8, region: u8) -> Self {
        let reach = u32::from(radius);
        let origin = [x.saturating_sub(reach), y.saturating_sub(reach)];
        let last = [
            x.saturating_add(reach).min(width - 1),
            y.saturating_add(reach).min(height - 1),
        ];
        Self {
            origin,
            extent: [last[0] - origin[0] + 1, last[1] - origin[1] + 1],
            center: [x - origin[0], y - origin[1]],
            region: Some((region, radius)),
        }
    }

    fn required_bytes(self) -> usize {
        (self.extent[1] as usize - 1) * ROW_PITCH + self.extent[0] as usize * size_of::<u32>()
    }

    fn select(self, bytes: &[u8], identities: &[u64], regions: &[u8]) -> Option<u32> {
        let bytes = bytes.get(..self.required_bytes())?;
        let row_bytes = self.extent[0] as usize * size_of::<u32>();
        let mut pixels = (0..self.extent[1] as usize)
            .flat_map(|row| {
                bytes[row * ROW_PITCH..row * ROW_PITCH + row_bytes].chunks_exact(size_of::<u32>())
            })
            .map(|pixel| u32::from_le_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]));
        let Some((region, radius)) = self.region else {
            return pixels.next();
        };
        let stride = self.extent[0] as usize;
        Some(
            pixels
                .enumerate()
                .filter_map(|(index, raw)| {
                    let slot = decode_mesh_slot(raw)?;
                    let dx = self.center[0].abs_diff((index % stride) as u32);
                    let dy = self.center[1].abs_diff((index / stride) as u32);
                    let distance_sq = dx * dx + dy * dy;
                    (distance_sq <= u32::from(radius).pow(2)
                        && regions.get(slot) == Some(&region)
                        && identities.get(slot).is_some_and(|&id| id != 0))
                    .then_some(((distance_sq, index), raw))
                })
                .min_by_key(|&(rank, _)| rank)
                .map_or(0, |(_, raw)| raw),
        )
    }
}

/// CPU-side metadata for one packed mesh pick slot. The R32Uint and mesh ABIs
/// remain unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshPickMetadata {
    pub source_slot: u32,
    pub surface_region: u8,
}
impl<R> PickingReadback<R> {
    pub fn new(buffer: etalon::Buffer, reservation: R, max_instances: usize) -> Self {
        Self {
            buffer,
            _reservation: reservation,
            worker: ready_worker(false),
            pending: None,
            copied: None,
            mapping: None,
            snapshot: vec![0; max_instances],
            source_slots: vec![0; max_instances],
            source_regions: vec![0; max_instances],
            snapshot_count: 0,
        }
    }
    pub fn queue(&mut self, x: f32, y: f32, width: u32, height: u32) {
        if self.mapping.is_some() {
            return;
        }
        self.pending = Some(PickWindow::single(
            (x.floor().max(0.0) as u32).min(width.saturating_sub(1)),
            (y.floor().max(0.0) as u32).min(height.saturating_sub(1)),
        ));
        self.copied = None;
    }

    /// Queue the nearest visible mesh pixel of the requested semantic region.
    /// Ties resolve top-to-bottom and then left-to-right.
    pub fn queue_nearest_mesh_region(
        &mut self,
        x: f32,
        y: f32,
        width: u32,
        height: u32,
        radius: u8,
        region: u8,
    ) -> Result<(), MeshRegionPickError> {
        if self.is_pending() {
            return Err(MeshRegionPickError::Busy);
        }
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x as u32 >= width
            || y as u32 >= height
            || radius > MAX_MESH_PICK_RADIUS
            || region > 2
        {
            return Err(MeshRegionPickError::InvalidInput);
        }
        let window = PickWindow::around(x as u32, y as u32, width, height, radius, region);
        if window.required_bytes() as u64 > self.buffer.size() {
            return Err(MeshRegionPickError::BufferTooSmall);
        }
        self.pending = Some(window);
        Ok(())
    }
    pub fn is_pending(&self) -> bool {
        self.pending.is_some() || self.copied.is_some() || self.mapping.is_some()
    }
    pub fn needs_recording(&self) -> bool {
        self.pending.is_some()
    }
    pub fn is_mesh_region_request(&self) -> bool {
        self.pending
            .or(self.copied)
            .is_some_and(|window| window.region.is_some())
    }
    pub fn mesh_region_scissor(&self) -> Option<[u32; 4]> {
        let window = self.pending?;
        window.region?;
        let [x, y] = window.origin;
        let [width, height] = window.extent;
        Some([x, y, width, height])
    }
    pub fn snapshot<T>(&mut self, source: &[T], identity: impl Fn(&T) -> u64) -> bool {
        self.snapshot_with_source_slots(source, identity, |slot| u32::try_from(slot).ok())
    }

    /// Snapshot authored source slots alongside full semantic identities. Readback
    /// must use this mapping from the recorded frame, even after another cull.
    pub fn snapshot_with_source_slots<T>(
        &mut self,
        source: &[T],
        identity: impl Fn(&T) -> u64,
        source_slot: impl Fn(usize) -> Option<u32>,
    ) -> bool {
        self.snapshot_with_metadata(source, identity, |index| {
            Some(MeshPickMetadata {
                source_slot: source_slot(index)?,
                surface_region: 0,
            })
        })
    }

    /// Snapshot region and physical source slot together with the frame ID.
    pub fn snapshot_with_metadata<T>(
        &mut self,
        source: &[T],
        identity: impl Fn(&T) -> u64,
        metadata: impl Fn(usize) -> Option<MeshPickMetadata>,
    ) -> bool {
        let Some(count) = snapshot_ids(source, &mut self.snapshot, identity) else {
            self.invalidate(false);
            return false;
        };
        for index in 0..count {
            let Some(metadata) = metadata(index) else {
                self.invalidate(false);
                return false;
            };
            if metadata.surface_region > 2 {
                self.invalidate(false);
                return false;
            }
            self.source_slots[index] = metadata.source_slot;
            self.source_regions[index] = metadata.surface_region;
        }
        self.snapshot_count = count;
        true
    }
    pub fn invalidate(&mut self, retain_pointer: bool) {
        // A copied or mapped pick describes the frame the visitor clicked.
        // Camera/pose updates must not drop that readback on wasm, where
        // mapAsync completes one or more frames later.
        if retain_pointer && (self.copied.is_some() || self.mapping.is_some()) {
            return;
        }
        if !retain_pointer {
            self.pending = None;
        }
        self.mapping = None;
        self.copied = None;
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
    pub fn record_copy(&mut self, encoder: &mut etalon::CommandEncoder, texture: &etalon::Texture) {
        let Some(window) = self.pending.take() else {
            return;
        };
        let [x, y] = window.origin;
        let [width, height] = window.extent;
        encoder.copy_texture_to_buffer(
            etalon::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: etalon::Origin3d { x, y, z: 0 },
                aspect: etalon::TextureAspect::All,
            },
            etalon::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: etalon::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(etalon::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(height),
                },
            },
            etalon::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.copied = Some(window);
    }
    pub fn poll(&mut self, device: &etalon::Device) -> Option<PickResult> {
        self.poll_with_source_slot(device).map(|(result, _)| result)
    }

    pub fn poll_with_source_slot(
        &mut self,
        device: &etalon::Device,
    ) -> Option<(PickResult, Option<u32>)> {
        self.poll_with_metadata(device)
            .map(|(result, metadata)| (result, metadata.map(|metadata| metadata.source_slot)))
    }

    pub fn poll_with_metadata(
        &mut self,
        device: &etalon::Device,
    ) -> Option<(PickResult, Option<MeshPickMetadata>)> {
        let window = self.copied?;
        if self.mapping.is_none() {
            self.mapping = PendingMap::begin(self.buffer.slice(..), self.worker.clone()).ok();
            if self.mapping.is_none() {
                self.copied = None;
                return None;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if device.poll(etalon::PollType::wait_indefinitely()).is_err() {
            self.notify_device_lost();
            return None;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let mapped = self.mapping.as_mut()?.try_complete()?;
        self.copied = None;
        let result = if mapped {
            self.buffer
                .slice(..)
                .get_mapped_range()
                .ok()
                .and_then(|view| {
                    window.select(
                        &view,
                        &self.snapshot[..self.snapshot_count],
                        &self.source_regions[..self.snapshot_count],
                    )
                })
                .map(|raw| {
                    let result = resolve_pick(raw, &self.snapshot[..self.snapshot_count]);
                    let metadata = decode_mesh_slot(raw)
                        .filter(|&index| index < self.snapshot_count)
                        .map(|index| MeshPickMetadata {
                            source_slot: self.source_slots[index],
                            surface_region: self.source_regions[index],
                        });
                    (result, metadata)
                })
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
    encoder: &mut etalon::CommandEncoder,
    color: &etalon::TextureView,
    depth: &etalon::TextureView,
    draw: impl FnOnce(&mut etalon::RenderPass<'_>),
) {
    let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
        label: Some("portal-picking-pass"),
        color_attachments: &[Some(etalon::RenderPassColorAttachment {
            view: color,
            depth_slice: None,
            resolve_target: None,
            ops: etalon::Operations {
                load: etalon::LoadOp::Clear(etalon::Color::TRANSPARENT),
                store: etalon::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(etalon::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(etalon::Operations {
                load: etalon::LoadOp::Clear(1.0),
                store: etalon::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        occlusion_query_set: None,
        multiview_mask: None,
        timestamp_writes: None,
    });
    draw(&mut pass);
}

#[cfg(test)]
mod tests {
    use super::{PickWindow, MESH_REGION_PICK_STAGING_BYTES, ROW_PITCH};
    use crate::picking::encode_mesh_slot;

    fn write_pixel(bytes: &mut [u8], x: usize, y: usize, raw: u32) {
        let offset = y * ROW_PITCH + x * 4;
        bytes[offset..offset + 4].copy_from_slice(&raw.to_le_bytes());
    }

    #[test]
    fn region_pick_uses_aligned_rows_rejects_other_regions_and_prefers_row_major_ties() {
        let window = PickWindow::around(6, 6, 10, 10, 2, 2);
        let mut bytes = [0; MESH_REGION_PICK_STAGING_BYTES as usize];
        let slot = |index| encode_mesh_slot(index).unwrap();
        write_pixel(&mut bytes, 2, 2, slot(0));
        write_pixel(&mut bytes, 3, 2, slot(1));
        write_pixel(&mut bytes, 2, 1, slot(2));
        write_pixel(&mut bytes, 1, 2, slot(3));
        write_pixel(&mut bytes, 2, 3, slot(4));
        assert_eq!(
            window.select(&bytes, &[5, 9, 0, 11], &[1, 2, 2, 2]),
            Some(slot(3))
        );
        write_pixel(&mut bytes, 2, 1, 17);
        assert_eq!(
            window.select(&bytes, &[5, 9, 3, 11], &[1, 2, 2, 2]),
            Some(slot(3))
        );
    }

    #[test]
    fn clipped_window_honors_circular_radius_and_legacy_reads_the_pointer() {
        let window = PickWindow::around(0, 0, 12, 12, 3, 2);
        assert_eq!(window.origin, [0, 0]);
        assert_eq!(window.extent, [4, 4]);
        assert_eq!(window.required_bytes(), 3 * ROW_PITCH + 16);
        let mut bytes = [0; MESH_REGION_PICK_STAGING_BYTES as usize];
        let mesh = encode_mesh_slot(0).unwrap();
        write_pixel(&mut bytes, 0, 0, 1);
        write_pixel(&mut bytes, 3, 3, mesh);
        assert_eq!(window.select(&bytes, &[41], &[2]), Some(0));
        write_pixel(&mut bytes, 3, 0, mesh);
        assert_eq!(window.select(&bytes, &[41], &[2]), Some(mesh));
        assert_eq!(PickWindow::single(0, 0).select(&bytes, &[], &[]), Some(1));
    }
}
