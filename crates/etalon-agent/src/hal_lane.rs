//! Etalon-HAL compute lane execution harness.

use super::capability::CapabilityState;
use super::compute_lane::{ComputeLane, PerQueueCompletionToken, QueueId};

/// Compute lane leveraging native Etalon HAL backends.
pub struct HalComputeLane {
    state: CapabilityState,
    queue: QueueId,
    fence_counter: u64,
}

impl HalComputeLane {
    pub fn new(queue: QueueId) -> Self {
        Self {
            state: CapabilityState::Detected,
            queue,
            fence_counter: 0,
        }
    }
}

impl ComputeLane for HalComputeLane {
    type Error = &'static str;
    type PreparedPlan = ();
    type Submission = ();

    fn state(&self) -> CapabilityState {
        self.state
    }

    fn prepare(&mut self, _plan: &Self::PreparedPlan) -> Result<(), Self::Error> {
        self.state = CapabilityState::Prepared;
        Ok(())
    }

    fn submit(&mut self, _submission: Self::Submission) -> Result<PerQueueCompletionToken, Self::Error> {
        self.fence_counter += 1;
        Ok(PerQueueCompletionToken::new(self.queue, self.fence_counter))
    }

    fn poll_completion(&self, token: PerQueueCompletionToken) -> bool {
        token.queue == self.queue && token.fence_value <= self.fence_counter
    }
}
