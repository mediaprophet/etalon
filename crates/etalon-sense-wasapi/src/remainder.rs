//! Caller-owned packet tails; no OS packet lease crosses a capture call.
use etalon_sense_core::adapter::AdapterError;

#[derive(Clone, Copy, Debug)]
pub(super) struct PacketMetadata {
    pub device_position: Option<u64>,
    pub qpc_position: Option<u64>,
    pub silent: bool,
    pub discontinuity: bool,
    pub timestamp_error: bool,
}

pub(super) struct Remainder<'a> {
    bytes: &'a mut [u8],
    start: usize,
    end: usize,
    metadata: Option<PacketMetadata>,
}

impl<'a> Remainder<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Self {
        bytes.fill(0);
        Self {
            bytes,
            start: 0,
            end: 0,
            metadata: None,
        }
    }
    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
    pub fn clear(&mut self) {
        self.bytes.fill(0);
        self.start = 0;
        self.end = 0;
        self.metadata = None;
    }
    pub fn stage(&mut self, source: &[u8], metadata: PacketMetadata) -> Result<(), AdapterError> {
        if source.is_empty() || !self.is_empty() || source.len() > self.bytes.len() {
            return Err(AdapterError::Unsupported);
        }
        self.bytes[..source.len()].copy_from_slice(source);
        self.start = 0;
        self.end = source.len();
        self.metadata = Some(metadata);
        Ok(())
    }
    pub fn stage_silence(
        &mut self,
        length: usize,
        metadata: PacketMetadata,
    ) -> Result<(), AdapterError> {
        if length == 0 || !self.is_empty() || length > self.bytes.len() {
            return Err(AdapterError::Unsupported);
        }
        self.bytes[..length].fill(0);
        self.start = 0;
        self.end = length;
        self.metadata = Some(metadata);
        Ok(())
    }
    pub fn drain(
        &mut self,
        output: &mut [u8],
        block_align: usize,
    ) -> Result<(usize, PacketMetadata), AdapterError> {
        if block_align == 0
            || output.is_empty()
            || self.is_empty()
            || (self.end - self.start) % block_align != 0
            || output.len() % block_align != 0
        {
            return Err(AdapterError::BackendFailure);
        }
        let count = output.len().min(self.end - self.start);
        let metadata = self.metadata.ok_or(AdapterError::BackendFailure)?;
        output[..count].copy_from_slice(&self.bytes[self.start..self.start + count]);
        self.bytes[self.start..self.start + count].fill(0);
        self.start += count;
        if self.is_empty() {
            self.clear();
        } else {
            self.metadata = Some(PacketMetadata {
                device_position: metadata
                    .device_position
                    .and_then(|position| position.checked_add((count / block_align) as u64)),
                qpc_position: None,
                discontinuity: false,
                ..metadata
            });
        }
        Ok((count / block_align, metadata))
    }
}
impl Drop for Remainder<'_> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata() -> PacketMetadata {
        PacketMetadata {
            device_position: Some(100),
            qpc_position: None,
            silent: false,
            discontinuity: false,
            timestamp_error: true,
        }
    }

    #[test]
    fn partial_drains_preserve_bytes_and_advance_only_device_position() {
        let mut scratch = [0xff; 12];
        {
            let mut pending = Remainder::new(&mut scratch);
            pending
                .stage(&[1, 2, 3, 4, 5, 6, 7, 8], metadata())
                .unwrap();
            let mut first = [0; 4];
            let (frames, info) = pending.drain(&mut first, 2).unwrap();
            assert_eq!(frames, 2);
            assert_eq!(first, [1, 2, 3, 4]);
            assert_eq!(info.device_position, Some(100));
            let mut second = [0; 8];
            let (frames, info) = pending.drain(&mut second, 2).unwrap();
            assert_eq!(frames, 2);
            assert_eq!(&second[..4], &[5, 6, 7, 8]);
            assert_eq!(info.device_position, Some(102));
            assert_eq!(info.qpc_position, None);
            assert!(info.timestamp_error);
            assert!(pending.is_empty());
        }
        assert_eq!(scratch, [0; 12]);
    }

    #[test]
    fn capacity_rejection_retains_packet_and_drop_wipes_scratch() {
        let mut scratch = [0; 8];
        {
            let mut pending = Remainder::new(&mut scratch);
            assert!(pending.stage(&[1; 9], metadata()).is_err());
            pending.stage(&[7; 8], metadata()).unwrap();
            assert!(pending.stage(&[1; 2], metadata()).is_err());
        }
        assert_eq!(scratch, [0; 8]);
    }

    #[test]
    fn silent_partial_packet_remains_silent() {
        let mut scratch = [9; 8];
        let mut pending = Remainder::new(&mut scratch);
        pending
            .stage_silence(
                8,
                PacketMetadata {
                    silent: true,
                    ..metadata()
                },
            )
            .unwrap();
        let mut output = [7; 4];
        let (_, info) = pending.drain(&mut output, 2).unwrap();
        assert!(info.silent);
        assert_eq!(output, [0; 4]);
    }
}
