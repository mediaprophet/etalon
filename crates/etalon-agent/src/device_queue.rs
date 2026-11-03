//! Checked device/queue ownership shared by compute and rendering clients.
use std::sync::Arc;

use crate::{HalComputeLane, QueueId};

/// Failure to establish backend ownership of a device/queue pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DeviceQueueError {
    #[error("queue belongs to a different device")]
    Mismatch,
    #[error("backend does not expose verifiable queue ownership")]
    Unverifiable,
}

/// Immutable handles whose backend ownership was checked at construction.
///
/// Clones retain the same pair. This does not attest device health, submission
/// completion, resource budgets, or exclusive use of either handle.
#[derive(Clone, Debug)]
pub struct SharedDeviceQueue {
    device: Arc<etalon::Device>,
    queue: Arc<etalon::Queue>,
}

impl SharedDeviceQueue {
    /// Refuses mismatched and uninspectable pairs before constructing a lane.
    pub fn try_new(
        device: Arc<etalon::Device>,
        queue: Arc<etalon::Queue>,
    ) -> Result<Self, DeviceQueueError> {
        match device.owns_queue(&queue) {
            Some(true) => Ok(Self { device, queue }),
            Some(false) => Err(DeviceQueueError::Mismatch),
            None => Err(DeviceQueueError::Unverifiable),
        }
    }

    pub fn device(&self) -> &Arc<etalon::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<etalon::Queue> {
        &self.queue
    }
}

impl HalComputeLane {
    /// Creates a lane from a checked pair. `queue_id` remains caller metadata.
    pub fn from_shared_pair(pair: SharedDeviceQueue, queue_id: QueueId) -> Self {
        Self::new(pair.device, pair.queue, queue_id)
    }
}
