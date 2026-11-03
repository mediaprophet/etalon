#[cfg(windows)]
use {super::SnapshotErrorKind, crate::lease::LeaseIdentity};

pub const COPY_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct SnapshotLimits {
    /// Includes the one-byte EOF probe after the expected payload.
    pub max_source_bytes: u64,
    /// Copied file payload; not filesystem allocation/metadata overhead.
    pub max_disk_bytes: u64,
    pub max_mapped_bytes: u64,
    /// Source + disk payload + mapped range + fixed copy buffer, not RSS.
    pub max_total_bytes: u64,
    /// Includes interrupted calls and the final EOF probe.
    pub max_read_calls: u64,
    /// Includes interrupted and partial File writes; does not bound syscall latency.
    pub max_write_calls: u64,
}
#[cfg(windows)]
impl SnapshotLimits {
    pub(crate) fn admit(self, expected: LeaseIdentity) -> Result<(), SnapshotErrorKind> {
        if expected.content_sha256 == [0; 32] {
            return Err(SnapshotErrorKind::InvalidIdentity);
        }
        let len = expected.byte_len;
        if len == 0 || usize::try_from(len).map_or(true, |n| n > isize::MAX as usize) {
            return Err(SnapshotErrorKind::InvalidLength);
        }
        let reads = len.checked_add(1).ok_or(SnapshotErrorKind::Budget)?;
        let total = len
            .checked_mul(3)
            .and_then(|n| n.checked_add(1 + COPY_BUFFER_BYTES as u64))
            .ok_or(SnapshotErrorKind::Budget)?;
        let minimum_calls = len
            .div_ceil(COPY_BUFFER_BYTES as u64)
            .checked_add(1)
            .ok_or(SnapshotErrorKind::Budget)?;
        if self.max_read_calls < minimum_calls {
            return Err(SnapshotErrorKind::ReadCallBudget);
        }
        if self.max_write_calls < minimum_calls - 1 {
            return Err(SnapshotErrorKind::WriteCallBudget);
        }
        if reads > self.max_source_bytes
            || len > self.max_disk_bytes
            || len > self.max_mapped_bytes
            || total > self.max_total_bytes
        {
            return Err(SnapshotErrorKind::Budget);
        }
        Ok(())
    }
}
