//! Keep the device and queue returned by one safe device request together.
use crate::{EtalonIqPowerProvider, GpuAllocationClass, GpuError};
use etalon::{Adapter, Device, DeviceDescriptor, Queue, RequestDeviceError};
use etalon_signal::execution::DeviceId;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRequestStage {
    BeforeRequest,
    AfterRequest,
}

/// Host must bind the adapter and opaque registry ID to current policy.
/// Callbacks cannot make an in-flight device request atomic with revocation.
pub trait DeviceRequestGate {
    fn permit(&mut self, adapter: &Adapter, device_id: DeviceId, stage: DeviceRequestStage)
        -> bool;
}

#[derive(Debug)]
pub enum PairRequestError {
    PolicyDenied,
    Request(RequestDeviceError),
}

/// Created only by one `Adapter::request_device` operation. Pairing establishes
/// construction consistency, not physical adapter identity or qualification.
pub struct DeviceQueuePair {
    device: Device,
    queue: Queue,
    device_id: DeviceId,
    lost: Arc<AtomicBool>,
}

impl DeviceQueuePair {
    /// Retains the safe Etalon request's feature, limit and single-request
    /// preconditions, including its documented panic behavior. Setup may
    /// allocate and the host controls request cancellation and runtime policy.
    pub async fn request(
        adapter: &Adapter,
        descriptor: &DeviceDescriptor<'_>,
        device_id: DeviceId,
        gate: &mut impl DeviceRequestGate,
    ) -> Result<Self, PairRequestError> {
        if !gate.permit(adapter, device_id, DeviceRequestStage::BeforeRequest) {
            return Err(PairRequestError::PolicyDenied);
        }
        let (device, queue) = adapter
            .request_device(descriptor)
            .await
            .map_err(PairRequestError::Request)?;
        if !gate.permit(adapter, device_id, DeviceRequestStage::AfterRequest) {
            return Err(PairRequestError::PolicyDenied);
        }
        let lost = Arc::new(AtomicBool::new(false));
        let callback_state = Arc::clone(&lost);
        device.set_device_lost_callback(move |_, _| {
            callback_state.store(true, Ordering::Release);
        });
        Ok(Self {
            device,
            queue,
            device_id,
            lost,
        })
    }

    pub const fn device_id(&self) -> DeviceId {
        self.device_id
    }
    /// Whether the installed callback has reported loss. Registration does
    /// not replay earlier loss; polling and backend validation may detect it
    /// later. False is not proof of current device validity.
    pub fn is_device_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Provider borrows the exact retained pair. The same cold allocation and
    /// buffer-retention restrictions as the direct provider constructor apply.
    pub fn iq_power_provider(
        &self,
        max_pairs: usize,
        allocation: GpuAllocationClass,
        gpu_buffer_budget_bytes: u64,
        poll_timeout: Duration,
    ) -> Result<EtalonIqPowerProvider<'_>, GpuError> {
        if self.is_device_lost() {
            return Err(GpuError::DeviceLost);
        }
        let mut provider = EtalonIqPowerProvider::new_with_allocation_class(
            &self.device,
            &self.queue,
            self.device_id,
            max_pairs,
            allocation,
            gpu_buffer_budget_bytes,
            poll_timeout,
        )?;
        provider.loss_state = Some(&self.lost);
        if self.is_device_lost() {
            return Err(GpuError::DeviceLost);
        }
        Ok(provider)
    }
}
