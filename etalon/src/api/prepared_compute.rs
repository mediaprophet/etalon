use core::{fmt, time::Duration};

use crate::*;

/// A fixed compute program with private upload, submission, and readback resources.
///
/// Create this using [`Adapter::prepare_compute`]. Preparation compiles and validates
/// WGSL, creates a dedicated device and five buffers, records reusable commands, and
/// allocates completion state. No device, queue, buffer, or mapped view escapes it.
///
/// The Vulkan execution path is designed to use fixed storage without calling the
/// Rust allocator, including when returning errors. Source review does not replace
/// target allocator and GPU qualification, and does not certify allocations inside
/// drivers, OS services, injected layers, or external hooks.
/// Preparation and destruction are cold operations and may allocate or block.
pub struct PreparedCompute {
    #[cfg(all(vulkan, feature = "wgsl"))]
    inner: wgc::prepared::PreparedCompute,
    #[cfg(not(all(vulkan, feature = "wgsl")))]
    _unavailable: core::convert::Infallible,
}

impl Adapter {
    /// Prepare a private synchronous compute lane, without exposing its device or queue.
    ///
    /// Currently requires the native Vulkan backend and the `wgsl` feature. Other
    /// backends return [`PreparedComputeError::UnsupportedBackend`] before creating
    /// resources. Instance debugging, backend validation, GPU validation, and debug
    /// printf must be disabled; normal Etalon shader/API validation remains enabled.
    /// The private device requests default limits and no optional features.
    ///
    /// The three bindings and fixed dispatch are described by
    /// [`PreparedComputeDescriptor`]. Only WGSL validated with runtime checks is accepted.
    /// This consumes an adapter's device request under strict WebGPU compliance.
    pub fn prepare_compute(
        &self,
        desc: &PreparedComputeDescriptor<'_>,
    ) -> Result<PreparedCompute, PrepareComputeError> {
        #[cfg(all(vulkan, feature = "wgsl"))]
        {
            let adapter = self
                .inner
                .as_core_opt()
                .ok_or(PreparedComputeError::UnsupportedBackend)?;
            let inner = wgc::prepared::PreparedCompute::new(&adapter.etalon_adapter, desc)?;
            Ok(PreparedCompute { inner })
        }
        #[cfg(not(all(vulkan, feature = "wgsl")))]
        {
            let _ = desc;
            Err(PreparedComputeError::UnsupportedBackend.into())
        }
    }
}

impl PreparedCompute {
    /// Execute one prepared dispatch and copy a readback prefix into `output`.
    ///
    /// Input may be shorter than the prepared capacity; the remaining bytes are
    /// zeroed on every run. Parameters must have exactly the prepared length.
    /// Output may be any prefix of the prepared output buffer. The entire GPU
    /// output is cleared before every dispatch, including bytes the shader does not write.
    /// Every error clears the caller's output slice.
    ///
    /// Capacity and parameter-length errors happen before upload/submission and allow
    /// retry. A timeout or backend failure permanently poisons this owner. A timeout
    /// does not cancel GPU execution: all resources remain owned until cold destruction
    /// can establish completion. If retirement cannot establish safety, destruction
    /// quarantines the entire private device and its resources instead of freeing them.
    ///
    /// Returns a nonwrapping successful-run sequence local to this owner. This is not
    /// a [`SubmissionIndex`] and cannot be passed to [`Device::poll`]. Exclusive mutable
    /// borrowing permits one execution at a time and prevents buffer reuse during a run.
    /// The timeout bounds fence waiting, not driver calls or the host copies.
    pub fn execute(
        &mut self,
        input: &[u8],
        parameters: &[u8],
        output: &mut [u8],
        timeout: Duration,
    ) -> Result<u64, PreparedComputeError> {
        #[cfg(all(vulkan, feature = "wgsl"))]
        {
            self.inner.execute(input, parameters, output, timeout)
        }
        #[cfg(not(all(vulkan, feature = "wgsl")))]
        {
            let _ = (input, parameters, output, timeout);
            match self._unavailable {}
        }
    }

    /// Whether execution is permanently disabled after failure, exhaustion, or retirement.
    pub fn is_poisoned(&self) -> bool {
        #[cfg(all(vulkan, feature = "wgsl"))]
        {
            self.inner.is_poisoned()
        }
        #[cfg(not(all(vulkan, feature = "wgsl")))]
        {
            match self._unavailable {}
        }
    }

    /// Number of successfully completed and retired runs.
    pub fn completed_runs(&self) -> u64 {
        #[cfg(all(vulkan, feature = "wgsl"))]
        {
            self.inner.completed_runs()
        }
        #[cfg(not(all(vulkan, feature = "wgsl")))]
        {
            match self._unavailable {}
        }
    }

    /// Permanently stop accepting work and establish that resources can be destroyed.
    ///
    /// This is a cold, potentially blocking operation; it is not bounded by a prior
    /// execution timeout. A timeout never cancels GPU work. If this returns an error,
    /// completion is still unknown. The owner retains every resource, permits another
    /// retirement attempt, and quarantines them if destruction also cannot establish
    /// completion. Success does not erase buffer contents or re-enable execution.
    pub fn retire(&mut self) -> Result<(), PreparedComputeError> {
        #[cfg(all(vulkan, feature = "wgsl"))]
        {
            self.inner.retire()
        }
        #[cfg(not(all(vulkan, feature = "wgsl")))]
        {
            match self._unavailable {}
        }
    }
}

impl fmt::Debug for PreparedCompute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCompute")
            .field("completed_runs", &self.completed_runs())
            .field("poisoned", &self.is_poisoned())
            .finish_non_exhaustive()
    }
}
