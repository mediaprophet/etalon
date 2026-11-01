use thiserror::Error;

pub const DEFAULT_FORWARD_ARENA_BUDGET: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TensorArenaError {
    #[error("forward batch is outside arena capacity or speculative verification limit")]
    InvalidBatch,
    #[error("forward readback copy exceeds source or staging buffer size")]
    ReadbackBufferTooSmall,
    #[error("arena dimensions must be nonzero")]
    EmptyDimension,
    #[error("arena allocation size overflow")]
    SizeOverflow,
    #[error("arena buffer exceeds the device buffer-size limit")]
    BufferLimit,
    #[error("arena storage buffer exceeds the device storage-binding limit")]
    StorageBindingLimit,
    #[error("arena allocations exceed the caller byte budget")]
    BudgetExceeded,
}

pub(super) struct AllocationSizes {
    pub storage: u64,
    pub staging: u64,
}

impl AllocationSizes {
    pub(super) fn checked(
        batch: u32,
        hidden: u32,
        layers: u32,
        limits: &etalon::Limits,
        budget: u64,
    ) -> Result<Self, TensorArenaError> {
        if batch == 0 || hidden == 0 || layers == 0 {
            return Err(TensorArenaError::EmptyDimension);
        }
        let storage = u64::from(batch)
            .checked_mul(u64::from(hidden))
            .and_then(|n| n.checked_mul(4))
            .ok_or(TensorArenaError::SizeOverflow)?
            .max(256);
        let staging = u64::from(batch)
            .checked_mul(4)
            .ok_or(TensorArenaError::SizeOverflow)?
            .max(64);
        if storage > limits.max_buffer_size || staging > limits.max_buffer_size {
            return Err(TensorArenaError::BufferLimit);
        }
        if storage > limits.max_storage_buffer_binding_size {
            return Err(TensorArenaError::StorageBindingLimit);
        }
        let total = storage
            .checked_mul(3)
            .and_then(|n| n.checked_add(staging))
            .ok_or(TensorArenaError::SizeOverflow)?;
        if total > budget {
            return Err(TensorArenaError::BudgetExceeded);
        }
        Ok(Self { storage, staging })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_overflow_and_zero_dimensions() {
        let limits = etalon::Limits::default();
        assert!(matches!(
            AllocationSizes::checked(u32::MAX, u32::MAX, 1, &limits, u64::MAX),
            Err(TensorArenaError::SizeOverflow)
        ));
        assert!(matches!(
            AllocationSizes::checked(0, 1, 1, &limits, u64::MAX),
            Err(TensorArenaError::EmptyDimension)
        ));
    }
    #[test]
    fn charges_all_padded_buffers_to_budget() {
        let limits = etalon::Limits::default();
        assert!(AllocationSizes::checked(1, 1, 1, &limits, 832).is_ok());
        assert!(matches!(
            AllocationSizes::checked(1, 1, 1, &limits, 831),
            Err(TensorArenaError::BudgetExceeded)
        ));
    }
    #[test]
    fn checks_storage_and_buffer_limits_before_allocation() {
        let mut limits = etalon::Limits {
            max_storage_buffer_binding_size: 255,
            ..Default::default()
        };
        assert!(matches!(
            AllocationSizes::checked(1, 1, 1, &limits, 832),
            Err(TensorArenaError::StorageBindingLimit)
        ));
        limits.max_buffer_size = 255;
        assert!(matches!(
            AllocationSizes::checked(1, 1, 1, &limits, 832),
            Err(TensorArenaError::BufferLimit)
        ));
    }
}
