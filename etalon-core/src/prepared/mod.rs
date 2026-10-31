//! Private, validated, fixed-shape native compute execution.

use alloc::sync::Arc;
use core::{mem::ManuallyDrop, sync::atomic::Ordering, time::Duration};

use wgt::{PrepareComputeError, PreparedComputeDescriptor, PreparedComputeError};

use crate::instance::Adapter;

mod setup;

/// A private device, its resources, and one reusable Vulkan submission.
///
/// No ordinary device, queue, buffer, or submission index escapes this owner.
/// Execution is synchronous and exclusively borrowed. Destruction is cold work
/// and may wait for a timed-out submission to retire.
pub struct PreparedCompute {
    raw: ManuallyDrop<hal::vulkan::prepared::PreparedCompute>,
    resources: ManuallyDrop<setup::Resources>,
    state: RunState,
    input_size: usize,
    parameter_size: usize,
    output_size: usize,
}

impl PreparedCompute {
    /// Validate and prepare a fixed compute program on a newly requested private device.
    pub fn new(
        adapter: &Arc<Adapter>,
        desc: &PreparedComputeDescriptor<'_>,
    ) -> Result<Self, PrepareComputeError> {
        if adapter.backend() != wgt::Backend::Vulkan {
            return Err(PreparedComputeError::UnsupportedBackend.into());
        }
        let diagnostic_flags = wgt::InstanceFlags::DEBUG
            | wgt::InstanceFlags::VALIDATION
            | wgt::InstanceFlags::GPU_BASED_VALIDATION
            | wgt::InstanceFlags::DEBUG_PRINTF;
        if adapter.instance.flags.intersects(diagnostic_flags) {
            return Err(PreparedComputeError::UnsupportedConfiguration.into());
        }
        desc.validate(&wgt::Limits::default())?;
        let resources = setup::Resources::new(adapter, desc)?;
        let raw = resources.prepare_hal(desc)?;
        Ok(Self {
            raw: ManuallyDrop::new(raw),
            resources: ManuallyDrop::new(resources),
            state: RunState::default(),
            input_size: desc.input_size as usize,
            parameter_size: desc.parameter_size as usize,
            output_size: desc.output_size as usize,
        })
    }

    /// Upload, run, wait, and copy a readback prefix into caller storage.
    ///
    /// Returns a nonwrapping sequence local to this owner, not a queue submission index.
    /// Every error clears `output`. Capacity/parameter errors leave the owner reusable;
    /// any backend failure or exhausted sequence permanently poisons it.
    pub fn execute(
        &mut self,
        input: &[u8],
        parameters: &[u8],
        output: &mut [u8],
        timeout: Duration,
    ) -> Result<u64, PreparedComputeError> {
        output.fill(0);
        self.state.check()?;
        if input.len() > self.input_size || output.len() > self.output_size {
            return Err(PreparedComputeError::Capacity);
        }
        if parameters.len() != self.parameter_size {
            return Err(PreparedComputeError::InvalidDescriptor);
        }
        if !self.resources.device.valid.load(Ordering::Acquire) {
            self.state.poisoned = true;
            return Err(PreparedComputeError::DeviceLost);
        }
        let generation = self.state.begin()?;
        // The private resources remain alive and inaccessible for the entire submission.
        let result = unsafe { self.raw.execute(input, parameters, output, timeout) };
        match result {
            Ok(()) => {
                self.state.completed = generation;
                Ok(generation)
            }
            Err(error) => {
                output.fill(0);
                self.state.poisoned = true;
                Err(error)
            }
        }
    }

    /// Whether this owner has permanently stopped accepting work.
    pub fn is_poisoned(&self) -> bool {
        self.state.poisoned
    }

    /// Number of successful, fully retired executions.
    pub fn completed_runs(&self) -> u64 {
        self.state.completed
    }

    /// Permanently stop execution and wait until private GPU resources can be freed.
    /// This cold operation may block beyond any earlier execution timeout.
    pub fn retire(&mut self) -> Result<(), PreparedComputeError> {
        self.state.poisoned = true;
        self.raw.retire()
    }
}

impl Drop for PreparedCompute {
    fn drop(&mut self) {
        if self.raw.retire().is_err() {
            // Unknown completion: quarantine the entire private device and its resources.
            return;
        }
        unsafe {
            ManuallyDrop::drop(&mut self.raw);
            ManuallyDrop::drop(&mut self.resources);
        }
    }
}

#[derive(Default)]
struct RunState {
    generation: u64,
    completed: u64,
    poisoned: bool,
}

impl RunState {
    fn check(&self) -> Result<(), PreparedComputeError> {
        if self.poisoned {
            Err(PreparedComputeError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn begin(&mut self) -> Result<u64, PreparedComputeError> {
        self.check()?;
        let Some(next) = self.generation.checked_add(1) else {
            self.poisoned = true;
            return Err(PreparedComputeError::GenerationExhausted);
        };
        self.generation = next;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generations_never_wrap_or_revive_a_poisoned_owner() {
        let mut state = RunState {
            generation: u64::MAX - 1,
            ..RunState::default()
        };
        assert_eq!(state.begin(), Ok(u64::MAX));
        assert_eq!(state.begin(), Err(PreparedComputeError::GenerationExhausted));
        assert_eq!(state.begin(), Err(PreparedComputeError::Poisoned));
        assert_eq!(state.generation, u64::MAX);
        assert_eq!(state.completed, 0);
    }

    #[test]
    fn failed_runs_cannot_be_reused() {
        let mut state = RunState::default();
        assert_eq!(state.begin(), Ok(1));
        state.poisoned = true;
        assert_eq!(state.begin(), Err(PreparedComputeError::Poisoned));
        assert_eq!(state.completed, 0);
    }
}
