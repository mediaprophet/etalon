use super::*;

#[derive(Clone)]
pub struct DmlProviderLease {
    device: ID3D12Device,
    dml_identity: u64,
    generation: u64,
    queue: ID3D12CommandQueue,
}

impl DmlProviderLease {
    pub fn capture(owner: &DmlDevice) -> Self {
        Self {
            device: owner.d3d12.clone(),
            dml_identity: Interface::as_raw(&owner.dml) as usize as u64,
            generation: owner.provider_generation,
            queue: owner.queue.clone(),
        }
    }

    pub fn matches(&self, owner: &DmlDevice) -> bool {
        self.device == owner.d3d12 && self.dml_identity == Interface::as_raw(&owner.dml) as usize as u64
            && self.generation == owner.provider_generation && self.queue == owner.queue
    }

    pub fn live(&self) -> windows::core::Result<()> {
        unsafe { self.device.GetDeviceRemovedReason() }
    }

    pub fn identity(&self) -> u64 {
        self.generation
    }
}
