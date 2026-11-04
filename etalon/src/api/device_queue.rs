use crate::{Device, Queue};

impl Device {
    /// Current backend validity, when the native backend exposes it.
    pub fn is_valid_state(&self) -> Option<bool> {
        #[cfg(etalon_core)]
        if let Some(device) = self.as_core() {
            return Some(device.is_valid());
        }
        None
    }
    /// Checks whether `queue` belongs to this device using backend ownership.
    ///
    /// Returns `None` when ownership cannot be inspected (including WebGPU and
    /// custom backends). This checks identity only, not device health or completion.
    pub fn owns_queue(&self, queue: &Queue) -> Option<bool> {
        #[cfg(etalon_core)]
        {
            use wgc::resource::ParentDevice;

            if let (Some(device), Some(queue)) = (self.as_core(), queue.as_core()) {
                return Some(alloc::sync::Arc::ptr_eq(&device, queue.device()));
            }
            #[cfg(webgpu)]
            if (self.as_core().is_some() && queue.as_webgpu().is_some())
                || (self.as_webgpu().is_some() && queue.as_core().is_some())
            {
                return Some(false);
            }
        }
        let _ = queue;
        None
    }
}
