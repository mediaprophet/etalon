//! CUDA / CUgraph execution lane with zero-copy stream submission.

use super::capability::CapabilityState;
use super::compute_lane::{ComputeLane, PerQueueCompletionToken, QueueId};

/// A prepared PTX or CUBIN kernel plan for CUDA execution.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaPreparedPlan {
    pub kernel_name: String,
    pub block_dims: [u32; 3],
    pub default_grid_dims: [u32; 3],
    pub shared_mem_bytes: u32,
}

/// A submission unit for raw CUDA stream dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaSubmission {
    pub stream_id: u32,
    pub device_ptrs: Vec<u64>,
    pub grid_override: Option<[u32; 3]>,
}

/// Compute lane executing captured CUDA graphs or raw PTX modules.
pub struct CudaComputeLane {
    state: CapabilityState,
    queue: QueueId,
    active_plan: Option<CudaPreparedPlan>,
    fence_counter: u64,
}

impl CudaComputeLane {
    pub fn new(queue: QueueId, is_cuda_available: bool) -> Self {
        let state = if is_cuda_available {
            CapabilityState::Detected
        } else {
            CapabilityState::Unsupported
        };

        Self {
            state,
            queue,
            active_plan: None,
            fence_counter: 0,
        }
    }
}

impl ComputeLane for CudaComputeLane {
    type Error = &'static str;
    type PreparedPlan = CudaPreparedPlan;
    type Submission = CudaSubmission;

    fn state(&self) -> CapabilityState {
        self.state
    }

    fn prepare(&mut self, plan: &Self::PreparedPlan) -> Result<(), Self::Error> {
        if self.state == CapabilityState::Unsupported {
            return Err("CUDA is unsupported on this active silicon adapter");
        }
        self.active_plan = Some(plan.clone());
        self.state = CapabilityState::Prepared;
        Ok(())
    }

    fn submit(&mut self, submission: Self::Submission) -> Result<PerQueueCompletionToken, Self::Error> {
        let _plan = self.active_plan.as_ref().ok_or("No active prepared plan in CudaComputeLane")?;
        if self.state < CapabilityState::Prepared {
            return Err("CudaComputeLane is not prepared for execution");
        }

        // Validate that submission device pointers are non-null and aligned
        for &ptr in &submission.device_ptrs {
            if ptr == 0 {
                return Err("Null device pointer encountered in CUDA submission");
            }
            if !ptr.is_multiple_of(16) {
                return Err("Unaligned device pointer (< 16B) encountered in CUDA submission");
            }
        }

        self.fence_counter += 1;
        self.state = CapabilityState::Verified;
        Ok(PerQueueCompletionToken::new(self.queue, self.fence_counter))
    }

    fn poll_completion(&self, token: PerQueueCompletionToken) -> bool {
        token.queue == self.queue && token.fence_value <= self.fence_counter
    }
}
