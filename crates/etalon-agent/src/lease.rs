//! Safe resource ownership, memory-region suballocation, and submission leases.

use super::compute_lane::PerQueueCompletionToken;

/// Unique identifier for an underlying driver buffer or host allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationId {
    pub device_id: u64,
    pub slot: u32,
    pub generation: u32,
}

/// Memory access rights for a leased region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessRights {
    ReadOnly,
    ReadWrite,
}

/// Validated slice within an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferRegion {
    pub allocation: AllocationId,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub alignment: u32,
    pub access: AccessRights,
}

impl BufferRegion {
    /// Validates alignment and capacity boundaries.
    pub fn validate(&self, total_capacity: u64, min_align: u32) -> Result<(), &'static str> {
        let align = self.alignment.max(min_align);
        if align > 0 && (self.byte_offset % align as u64 != 0) {
            return Err("BufferRegion alignment violation");
        }
        let end = self.byte_offset.checked_add(self.byte_len).ok_or("BufferRegion offset overflow")?;
        if end > total_capacity {
            return Err("BufferRegion exceeds allocation bounds");
        }
        Ok(())
    }
}

/// A safe lease that holds access rights until a hardware queue token completes.
pub struct SubmissionLease<T> {
    pub resource: T,
    pub region: BufferRegion,
    pub token: PerQueueCompletionToken,
}

impl<T> SubmissionLease<T> {
    pub fn new(resource: T, region: BufferRegion, token: PerQueueCompletionToken) -> Self {
        Self {
            resource,
            region,
            token,
        }
    }
}
