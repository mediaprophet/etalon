use etalon_wasm::readback::{PendingMap, SharedWorker};

pub struct ComputeReadback {
    map: PendingMap,
    bytes: usize,
}

impl ComputeReadback {
    pub fn begin(
        buffer: &etalon::Buffer,
        bytes: usize,
        worker: SharedWorker,
    ) -> Result<Self, String> {
        if bytes == 0 || bytes as u64 > buffer.size() {
            return Err("Compute readback byte range exceeds staging buffer".into());
        }
        let map = PendingMap::begin(buffer.slice(..), worker)
            .map_err(|error| format!("Compute map admission: {error:?}"))?;
        Ok(Self { map, bytes })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn poll(&mut self, device: &etalon::Device) -> Result<Option<bool>, String> {
        #[cfg(not(target_arch = "wasm32"))]
        device
            .poll(etalon::PollType::wait_indefinitely())
            .map_err(|error| format!("Compute device poll: {error}"))?;
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        Ok(self.map.try_complete())
    }

    pub fn copy_into(&mut self, out: &mut [u8]) -> Result<usize, String> {
        if out.len() < self.bytes {
            return Err("Compute readback output buffer too small".into());
        }
        if self.map.try_complete() != Some(true) {
            return Err("Compute map failed, pending, or stale".into());
        }
        let mapped = self
            .map
            .buffer()
            .slice(..)
            .get_mapped_range()
            .map_err(|error| format!("Compute mapped range: {error}"))?;
        out[..self.bytes].copy_from_slice(&mapped[..self.bytes]);
        Ok(self.bytes)
    }
}
