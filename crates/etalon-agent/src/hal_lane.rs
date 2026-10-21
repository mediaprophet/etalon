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

/// Real hardware compute lane leveraging native Etalon HAL (Vulkan, DX12, Metal, GLES, WebGPU).
pub struct HalComputeLane {
    device: Arc<etalon::Device>,
    queue: Arc<etalon::Queue>,
    queue_id: QueueId,
    state: CapabilityState,
    active_plan: Option<HalPreparedPlan>,
    submitted_counter: u64,
    completed_counter: Arc<AtomicU64>,
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

    /// Submit a recorded command buffer and track GPU completion via `on_submitted_work_done`.
    pub fn submit_command_buffer(&mut self, cmd: etalon::CommandBuffer) -> SubmissionToken {
        self.submitted_counter += 1;
        let fence_value = self.submitted_counter;
        self.queue.submit(std::iter::once(cmd));
        let completed = Arc::clone(&self.completed_counter);
        self.queue.on_submitted_work_done(move || {
            completed.fetch_max(fence_value, Ordering::Release);
        });
        self.state = CapabilityState::Verified;
        PerQueueCompletionToken::new(self.queue_id, fence_value)
    }

    /// Block until the GPU has completed all work up to `token`.
    pub fn wait_for(&self, token: SubmissionToken) -> bool {
        if token.queue != self.queue_id {
            return false;
        }
        while self.completed_counter.load(Ordering::Acquire) < token.fence_value {
            if self.device.poll(etalon::PollType::wait_indefinitely()).is_err() {
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
        self.state = CapabilityState::Prepared;
        Ok(())
    }

    fn submit(&mut self, submission: Self::Submission) -> Result<PerQueueCompletionToken, Self::Error> {
        let plan = self.active_plan.as_ref().ok_or("No prepared plan active in HalComputeLane")?;
        let mut encoder = self.device.create_command_encoder(&etalon::CommandEncoderDescriptor {
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
            let wg = submission.workgroups_override.unwrap_or(plan.default_workgroups);
            cpass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }

        Ok(self.submit_command_buffer(encoder.finish()))
    }

    fn poll_completion(&self, token: PerQueueCompletionToken) -> bool {
        if token.queue != self.queue_id {
            return false;
        }
        let _ = self.device.poll(etalon::PollType::Poll);
        self.completed_counter.load(Ordering::Acquire) >= token.fence_value
    }
}
