//! Safe resource ownership, memory-region suballocation, and submission leases.

use super::{
    compute_lane::{ComputeLane, PerQueueCompletionToken},
    hal_lane::HalComputeLane,
};
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
        if align > 0 && !self.byte_offset.is_multiple_of(u64::from(align)) {
            return Err("BufferRegion alignment violation");
        }
        let end = self
            .byte_offset
            .checked_add(self.byte_len)
            .ok_or("BufferRegion offset overflow")?;
        if end > total_capacity {
            return Err("BufferRegion exceeds allocation bounds");
        }
        Ok(())
    }
}

/// Queue-retained resource ownership until hardware completion.
///
/// Construct through [`HalComputeLane::lease_submission`]. The former unrestricted `new`
/// constructor and public resource fields are removed: only the issuing queue can prove completion.
/// Dropping a lease releases its resource only after the queue's retained reference is released.
pub struct SubmissionLease<T> {
    resource: Arc<T>,
    region: BufferRegion,
    token: PerQueueCompletionToken,
    completion: Arc<AtomicU64>,
}

impl<T> SubmissionLease<T> {
    pub(crate) fn queue_bound(
        resource: T,
        region: BufferRegion,
        token: PerQueueCompletionToken,
        completion: Arc<AtomicU64>,
    ) -> (Self, Arc<T>) {
        let resource = Arc::new(resource);
        let retained = Arc::clone(&resource);
        (
            Self {
                resource,
                region,
                token,
                completion,
            },
            retained,
        )
    }

    pub fn region(&self) -> BufferRegion {
        self.region
    }

    pub fn token(&self) -> PerQueueCompletionToken {
        self.token
    }

    /// Return ownership only after the issuing lane's queue callback released its retention.
    /// Another lane with the same public queue ID cannot reclaim this lease.
    pub fn try_reclaim(self, lane: &HalComputeLane) -> Result<T, Self> {
        if !lane.owns_lease(self.token, &self.completion) {
            return Err(self);
        }
        let _ = lane.poll_completion(self.token);
        self.reclaim_completed()
    }

    fn reclaim_completed(self) -> Result<T, Self> {
        if self.completion.load(Ordering::Acquire) < self.token.fence_value {
            return Err(self);
        }
        let Self {
            resource,
            region,
            token,
            completion,
        } = self;
        match Arc::try_unwrap(resource) {
            Ok(resource) => Ok(resource),
            Err(resource) => Err(Self {
                resource,
                region,
                token,
                completion,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_lane::QueueId;
    use std::sync::atomic::AtomicUsize;

    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn region() -> BufferRegion {
        BufferRegion {
            allocation: AllocationId {
                device_id: 1,
                slot: 0,
                generation: 1,
            },
            byte_offset: 0,
            byte_len: 16,
            alignment: 16,
            access: AccessRights::ReadWrite,
        }
    }

    fn lease(
        completion: Arc<AtomicU64>,
        drops: Arc<AtomicUsize>,
    ) -> (SubmissionLease<Resource>, Arc<Resource>) {
        SubmissionLease::queue_bound(
            Resource(drops),
            region(),
            PerQueueCompletionToken::new(QueueId(3), 1),
            completion,
        )
    }

    #[test]
    fn dropping_before_completion_retains_resource_until_callback_releases_it() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (lease, queue_retained) = lease(Arc::new(AtomicU64::new(0)), Arc::clone(&drops));
        drop(lease);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(queue_retained);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn incomplete_and_counter_only_completion_cannot_reclaim_retained_resource() {
        let drops = Arc::new(AtomicUsize::new(0));
        let completion = Arc::new(AtomicU64::new(0));
        let (lease, queue_retained) = lease(Arc::clone(&completion), Arc::clone(&drops));
        let lease = lease.reclaim_completed().err().unwrap();
        completion.store(1, Ordering::Release);
        let lease = lease.reclaim_completed().err().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(queue_retained);
        let resource = lease.reclaim_completed().ok().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(resource);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_after_completion_releases_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (lease, queue_retained) = lease(Arc::new(AtomicU64::new(1)), Arc::clone(&drops));
        drop(queue_retained);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(lease);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
