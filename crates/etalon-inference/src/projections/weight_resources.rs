use crate::gpu_resources::ResourceBudget;
use std::collections::HashMap;
/// Cold caller-owned cache. Host retains the immutable model source and invalidates replacement.
#[derive(Default)]
pub struct ResidentWeightCache {
    device: Option<wgpu::Device>,
    buffers: HashMap<(u64, usize), wgpu::Buffer>,
    retained_bytes: u64,
}
impl ResidentWeightCache {
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.retained_bytes = 0;
        self.device = None;
    }
    pub fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }
}
pub fn resident_weight_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut ResidentWeightCache,
    key: u64,
    raw: &[u8],
    budget: ResourceBudget,
) -> Option<wgpu::Buffer> {
    if cache.device.as_ref().is_some_and(|owner| owner != device) {
        cache.clear();
    }
    if let Some(buffer) = cache.buffers.get(&(key, raw.len())) {
        return Some(buffer.clone());
    }
    if raw.is_empty() || raw.len() % 4 != 0 {
        return None;
    }
    let bytes = u64::try_from(raw.len().checked_add(255)? & !255).ok()?;
    let retained = cache.retained_bytes.checked_add(bytes)?;
    if bytes > budget.max_single_buffer_bytes
        || bytes > device.limits().max_buffer_size
        || bytes > u64::from(device.limits().max_storage_buffer_binding_size)
        || retained > budget.max_buffer_bytes
    {
        return None;
    }
    // At least 256 budgeted GPU bytes per entry bounds metadata entry count too.
    cache.buffers.try_reserve(1).ok()?;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ResidentWeight"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, raw);
    cache.buffers.insert((key, raw.len()), buffer.clone());
    cache.retained_bytes = retained;
    cache.device = Some(device.clone());
    Some(buffer)
}
pub fn write_weight_words(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    raw: &[u8],
    max_bytes: usize,
) {
    queue.write_buffer(buffer, 0, &raw[..raw.len().min(max_bytes)]);
}
