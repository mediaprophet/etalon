//! Native resident execution; model access and authority are supplied by the caller.
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
mod resources;
pub mod planning;
pub use resources::ResidentResources;
mod prefill_arena;
pub mod prefill_arena_plan;
pub mod prefill_arena_slots;
mod resident_decode;
pub mod resident_decode_plan;
pub mod resident_decode_slots;
mod verify_arena;
pub mod verify_arena_plan;
pub mod verify_arena_slots;
pub struct ResidentEngine<'a, P> {
    pub host: &'a P,
}
pub trait ResidentProvider {
    type Mask: AsRef<[u32]>;
    fn resident_resources(&self) -> ResidentResources<'_>;
    fn gpu_device(&self) -> &wgpu::Device;
    fn gpu_queue(&self) -> &wgpu::Queue;
    fn submit(&self, command: wgpu::CommandBuffer) -> bool;
    fn poll_wait(&self) -> bool;
    fn attention_kv_mask(&self, token: u32, capacity: u32) -> (Self::Mask, u32);
    fn dual_multirow(&self) -> bool;
    fn coop_gemv_workgroups(&self, rows: u32) -> u32;
    fn pass_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'static>>;
    fn resolve(&self, encoder: &mut wgpu::CommandEncoder);
    fn accumulate_fused(&self);
}
struct MappedRead<'a> {
    buffer: &'a wgpu::Buffer,
}
impl Drop for MappedRead<'_> {
    fn drop(&mut self) {
        self.buffer.unmap();
    }
}
fn map_read<'a, P: ResidentProvider>(
    host: &P,
    buffer: &'a wgpu::Buffer,
    bytes: u64,
    status: &Arc<AtomicU64>,
) -> Option<MappedRead<'a>> {
    let pending = status.fetch_add(4, Ordering::AcqRel).wrapping_add(4) & !3;
    status.store(pending, Ordering::Release);
    let completion = Arc::clone(status);
    let guard = MappedRead { buffer };
    buffer
        .slice(..bytes)
        .map_async(wgpu::MapMode::Read, move |result| {
            let state = pending | if result.is_ok() { 1 } else { 2 };
            let _ =
                completion.compare_exchange(pending, state, Ordering::AcqRel, Ordering::Acquire);
        });
    if !host.poll_wait() || status.load(Ordering::Acquire) != (pending | 1) {
        return None;
    }
    Some(guard)
}
