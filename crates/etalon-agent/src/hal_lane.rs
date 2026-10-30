//! Etalon-HAL compute lane execution harness.

use super::capability::CapabilityState;
use super::compute_lane::{ComputeLane, PerQueueCompletionToken, QueueId, SubmissionToken};
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A prepared compute pipeline and workgroup sizing on an Etalon device.
#[derive(Clone)]
pub struct HalPreparedPlan {
    pub pipeline: Arc<etalon::ComputePipeline>,
    pub default_workgroups: [u32; 3],
}

/// A submission unit containing the bind groups and optional workgroup overrides.
#[derive(Clone)]
pub struct HalSubmission {
    pub bind_groups: Vec<Arc<etalon::BindGroup>>,
    pub workgroups_override: Option<[u32; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HalOracleError {
    #[error("the oracle submission belongs to another queue")]
    WrongQueue,
    #[error("the oracle submission was not issued by this lane")]
    UnknownSubmission,
    #[error("the oracle submission predates the latest preparation or submission")]
    StaleSubmission,
    #[error("the oracle submission has not completed")]
    IncompleteSubmission,
    #[error("an empty CPU reference cannot verify execution")]
    EmptyReference,
    #[error("CPU reference and device readback lengths differ")]
    LengthMismatch,
    #[error("device readback differs from the CPU reference at byte {index}")]
    OutputMismatch { index: usize },
}

/// Native Etalon compute lane requiring synchronous completion indices.
pub struct HalComputeLane {
    device: Arc<etalon::Device>,
    queue: Arc<etalon::Queue>,
    queue_id: QueueId,
    state: CapabilityState,
    active_plan: Option<HalPreparedPlan>,
    submitted_counter: u64,
    completed_counter: Arc<AtomicU64>,
    preparation_fence: u64,
    verified_submission: Option<SubmissionToken>,
}

impl HalComputeLane {
    pub fn new(device: Arc<etalon::Device>, queue: Arc<etalon::Queue>, queue_id: QueueId) -> Self {
        Self {
            device,
            queue,
            queue_id,
            state: CapabilityState::Detected,
            active_plan: None,
            submitted_counter: 0,
            completed_counter: Arc::new(AtomicU64::new(0)),
            preparation_fence: 0,
            verified_submission: None,
        }
    }

    pub fn device(&self) -> &etalon::Device {
        &self.device
    }

    pub fn queue(&self) -> &etalon::Queue {
        &self.queue
    }

    pub fn submitted_counter(&self) -> u64 {
        self.submitted_counter
    }

    pub fn completed_counter(&self) -> u64 {
        self.completed_counter.load(Ordering::Acquire)
    }

    pub fn completed_counter_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.completed_counter)
    }

    pub fn verified_submission(&self) -> Option<SubmissionToken> {
        self.verified_submission
    }

    /// Compare a CPU reference with caller-supplied readback from the latest completed submission.
    pub fn verify_submission(
        &mut self,
        token: SubmissionToken,
        expected: &[u8],
        actual: &[u8],
    ) -> Result<(), HalOracleError> {
        self.clear_verification();
        validate_oracle(
            token,
            self.queue_id,
            self.submitted_counter,
            self.preparation_fence,
            self.completed_counter(),
            expected,
            actual,
        )?;
        self.verified_submission = Some(token);
        self.state = CapabilityState::Verified;
        Ok(())
    }

    fn clear_verification(&mut self) {
        self.verified_submission = None;
        self.state = if self.active_plan.is_some() {
            CapabilityState::Prepared
        } else {
            CapabilityState::Detected
        };
    }

    /// Submit a recorded command buffer and track GPU completion via `on_submitted_work_done`.
    pub fn submit_command_buffer(
        &mut self,
        cmd: etalon::CommandBuffer,
    ) -> Result<SubmissionToken, &'static str> {
        self.clear_verification();
        if cfg!(target_arch = "wasm32") {
            return Err("HalComputeLane submission requires native completion indices");
        }
        let next_fence = self
            .submitted_counter
            .checked_add(1)
            .ok_or("Submission fence exhausted")?;
        self.queue
            .submit(std::iter::once(cmd))
            .ok_or("Queue submission did not start")?;
        self.submitted_counter = next_fence;
        let fence_value = self.submitted_counter;
        let completed = Arc::clone(&self.completed_counter);
        self.queue.on_submitted_work_done(move || {
            completed.fetch_max(fence_value, Ordering::Release);
        });
        Ok(PerQueueCompletionToken::new(self.queue_id, fence_value))
    }

    /// Block until the GPU has completed all work up to `token`.
    pub fn wait_for(&self, token: SubmissionToken) -> bool {
        if token.queue != self.queue_id
            || token.fence_value == 0
            || token.fence_value > self.submitted_counter
        {
            return false;
        }
        while self.completed_counter.load(Ordering::Acquire) < token.fence_value {
            if self
                .device
                .poll(etalon::PollType::wait_indefinitely())
                .is_err()
            {
                break;
            }
        }
        self.completed_counter.load(Ordering::Acquire) >= token.fence_value
    }
}

impl ComputeLane for HalComputeLane {
    type Error = &'static str;
    type PreparedPlan = HalPreparedPlan;
    type Submission = HalSubmission;

    fn state(&self) -> CapabilityState {
        self.state
    }

    fn prepare(&mut self, plan: &Self::PreparedPlan) -> Result<(), Self::Error> {
        self.active_plan = Some(plan.clone());
        self.preparation_fence = self.submitted_counter;
        self.verified_submission = None;
        self.state = CapabilityState::Prepared;
        Ok(())
    }

    fn submit(
        &mut self,
        submission: Self::Submission,
    ) -> Result<PerQueueCompletionToken, Self::Error> {
        let plan = self
            .active_plan
            .as_ref()
            .ok_or("No prepared plan active in HalComputeLane")?;
        let mut encoder = self
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor {
                label: Some("etalon-hal-compute-encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: Some("etalon-hal-compute-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&plan.pipeline);
            for (idx, bg) in submission.bind_groups.iter().enumerate() {
                cpass.set_bind_group(idx as u32, &**bg, &[]);
            }
            let wg = submission
                .workgroups_override
                .unwrap_or(plan.default_workgroups);
            cpass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }

        self.submit_command_buffer(encoder.finish())
    }

    fn poll_completion(&self, token: PerQueueCompletionToken) -> bool {
        if token.queue != self.queue_id
            || token.fence_value == 0
            || token.fence_value > self.submitted_counter
        {
            return false;
        }
        let _ = self.device.poll(etalon::PollType::Poll);
        self.completed_counter.load(Ordering::Acquire) >= token.fence_value
    }
}

fn validate_oracle(
    token: SubmissionToken,
    queue: QueueId,
    latest_fence: u64,
    preparation_fence: u64,
    completed_fence: u64,
    expected: &[u8],
    actual: &[u8],
) -> Result<(), HalOracleError> {
    if token.queue != queue {
        return Err(HalOracleError::WrongQueue);
    }
    if token.fence_value == 0 || token.fence_value > latest_fence {
        return Err(HalOracleError::UnknownSubmission);
    }
    if token.fence_value != latest_fence || token.fence_value <= preparation_fence {
        return Err(HalOracleError::StaleSubmission);
    }
    if token.fence_value > completed_fence {
        return Err(HalOracleError::IncompleteSubmission);
    }
    if expected.is_empty() {
        return Err(HalOracleError::EmptyReference);
    }
    if expected.len() != actual.len() {
        return Err(HalOracleError::LengthMismatch);
    }
    if let Some(index) = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
    {
        return Err(HalOracleError::OutputMismatch { index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(
        token: SubmissionToken,
        prepared: u64,
        completed: u64,
        expected: &[u8],
        actual: &[u8],
    ) -> Result<(), HalOracleError> {
        validate_oracle(token, QueueId(7), 3, prepared, completed, expected, actual)
    }

    #[test]
    fn submission_without_completion_does_not_verify() {
        assert_eq!(
            check(SubmissionToken::new(QueueId(7), 3), 0, 2, &[1], &[1]),
            Err(HalOracleError::IncompleteSubmission)
        );
    }

    #[test]
    fn a_completed_submission_requires_matching_nonempty_readback() {
        let token = SubmissionToken::new(QueueId(7), 3);
        assert_eq!(
            check(token, 0, 3, &[], &[]),
            Err(HalOracleError::EmptyReference)
        );
        assert_eq!(
            check(token, 0, 3, &[1, 2], &[1]),
            Err(HalOracleError::LengthMismatch)
        );
        assert_eq!(
            check(token, 0, 3, &[1, 2], &[1, 3]),
            Err(HalOracleError::OutputMismatch { index: 1 })
        );
        assert_eq!(check(token, 0, 3, &[1, 2], &[1, 2]), Ok(()));
    }

    #[test]
    fn stale_or_foreign_tokens_cannot_verify_new_work() {
        assert_eq!(
            check(SubmissionToken::new(QueueId(8), 3), 0, 3, &[1], &[1]),
            Err(HalOracleError::WrongQueue)
        );
        for fence in [0, 4] {
            assert_eq!(
                check(SubmissionToken::new(QueueId(7), fence), 0, 3, &[1], &[1]),
                Err(HalOracleError::UnknownSubmission)
            );
        }
        assert_eq!(
            check(SubmissionToken::new(QueueId(7), 2), 0, 3, &[1], &[1]),
            Err(HalOracleError::StaleSubmission)
        );
        assert_eq!(
            check(SubmissionToken::new(QueueId(7), 3), 3, 3, &[1], &[1]),
            Err(HalOracleError::StaleSubmission)
        );
    }
}
