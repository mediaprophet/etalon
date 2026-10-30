//! Cold prepare() and hot submit() compute lane abstraction.

use super::capability::CapabilityState;

/// Hardware queue identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueId(pub u32);

/// Per-queue monotonically increasing completion token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PerQueueCompletionToken {
    pub queue: QueueId,
    pub fence_value: u64,
}

/// Type alias for a queue submission fence token.
pub type SubmissionToken = PerQueueCompletionToken;

impl PerQueueCompletionToken {
    pub const fn new(queue: QueueId, fence_value: u64) -> Self {
        Self { queue, fence_value }
    }
}

/// Core interface for compute execution backends (Etalon-HAL, CUDA, DirectML).
pub trait ComputeLane {
    type Error;
    type PreparedPlan;
    type Submission;

    /// Current capability state of this lane.
    fn state(&self) -> CapabilityState;

    /// Cold path: allocate resources, compile pipelines, and validate layouts.
    fn prepare(&mut self, plan: &Self::PreparedPlan) -> Result<(), Self::Error>;

    /// Hot path: zero-allocation submission to hardware queues.
    fn submit(
        &mut self,
        submission: Self::Submission,
    ) -> Result<PerQueueCompletionToken, Self::Error>;

    /// Non-blocking check for token completion.
    fn poll_completion(&self, token: PerQueueCompletionToken) -> bool;
}
