//! Terminal disposal of the legacy shared numerical CUDA owner.
//! Epoch revocation is performed by the request coordinator before entry here.
//! Holding the device mutex quiesces all lane users before the owner is removed;
//! synchronization drains both streams before graph/slab destruction.
pub fn discard_numerical_owner() {
    let mut slot = super::device::multi_weight_device()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(mut device) = slot.take() {
        if let Some(prefetch) = device.ctx.prefetch_stream.as_ref() {
            if let Err(error) = prefetch.synchronize() {
                log::warn!("cuda_lane|terminal-prefetch-sync-failed|{error}");
            }
        }
        if let Err(error) = device.ctx.stream.synchronize() {
            log::warn!("cuda_lane|terminal-compute-sync-failed|{error}");
        }
        // Captured graphs borrow slab addresses: destroy before their context.
        device.decode_graph = None;
        drop(device);
    }
}
