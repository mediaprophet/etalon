//! Native readback driven by the caller's device poll. Prepare completion state once;
//! callback clones reuse it without application heap allocation. wgpu internals may allocate.
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
#[derive(Default)]
struct Signal {
    next: AtomicU64,
    state: AtomicU64,
}
#[derive(Default)]
pub struct ReadbackCompletion {
    signal: Arc<Signal>,
}
pub fn map_read(
    completion: &ReadbackCompletion,
    buffer: &wgpu::Buffer,
    bytes: u64,
    poll: impl FnOnce(),
) -> bool {
    if bytes == 0 || bytes > buffer.size() || bytes % 4 != 0 {
        return false;
    }
    let signal = &completion.signal;
    // Distinct tokens prevent a late cancellation callback completing a newer map.
    let token = signal.next.fetch_add(4, Ordering::Relaxed).wrapping_add(4);
    if token == 0
        || signal
            .state
            .compare_exchange(0, token, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
    {
        return false;
    }
    let callback = signal.clone();
    buffer
        .slice(..bytes)
        .map_async(wgpu::MapMode::Read, move |result| {
            let done = token | if result.is_ok() { 1 } else { 2 };
            let _ =
                callback
                    .state
                    .compare_exchange(token, done, Ordering::AcqRel, Ordering::Acquire);
        });
    poll();
    let success = signal.state.load(Ordering::Acquire) == (token | 1);
    if !success {
        buffer.unmap();
    }
    signal.state.store(0, Ordering::Release);
    success
}
pub fn readback_into(
    completion: &ReadbackCompletion,
    buffer: &wgpu::Buffer,
    bytes: u64,
    out: &mut [f32],
    poll: impl FnOnce(),
) -> bool {
    if out.len().checked_mul(4).and_then(|n| u64::try_from(n).ok()) != Some(bytes)
        || !map_read(completion, buffer, bytes, poll)
    {
        return false;
    }
    let success = match buffer.slice(..bytes).get_mapped_range() {
        Ok(data) => match bytemuck::try_cast_slice::<u8, f32>(&data) {
            Ok(values) if values.len() == out.len() => {
                out.copy_from_slice(values);
                true
            }
            _ => false,
        },
        Err(_) => false,
    };
    buffer.unmap();
    success
}
pub fn readback_owned(
    completion: &ReadbackCompletion,
    buffer: &wgpu::Buffer,
    bytes: u64,
    byte_budget: usize,
    poll: impl FnOnce(),
) -> Option<Vec<f32>> {
    let bytes = usize::try_from(bytes).ok()?;
    if bytes == 0 || bytes % 4 != 0 || bytes > byte_budget || bytes as u64 > buffer.size() {
        return None;
    }
    let mut output = Vec::new();
    output.try_reserve_exact(bytes / 4).ok()?;
    output.resize(bytes / 4, 0.0);
    readback_into(completion, buffer, bytes as u64, &mut output, poll).then_some(output)
}
