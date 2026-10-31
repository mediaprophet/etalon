use alloc::string::String;
use core::fmt;

/// A fixed compute program for a private, synchronous native execution lane.
///
/// Group zero contains binding 0 (read-only storage input), binding 1
/// (read-write storage output), and binding 2 (uniform parameters). The shader
/// is validated normally; passthrough shaders and optional features are not accepted.
#[derive(Clone, Copy, Debug)]
pub struct PreparedComputeDescriptor<'a> {
    /// WGSL source, compiled during preparation.
    pub shader: &'a str,
    /// Compute entry point.
    pub entry_point: &'a str,
    /// Fixed input storage capacity in bytes, nonzero and divisible by four.
    pub input_size: u64,
    /// Exact uniform parameter size in bytes, nonzero and divisible by four.
    pub parameter_size: u64,
    /// Fixed output storage capacity in bytes, nonzero and divisible by four.
    pub output_size: u64,
    /// Fixed dispatch dimensions. Variable workloads must use parameters to mask work.
    pub workgroups: [u32; 3],
    /// Maximum sum of the five logical buffer sizes, including upload and readback.
    ///
    /// This excludes allocator padding, device initialization resources, shaders,
    /// command pools, and driver memory. It is not a total GPU memory budget.
    pub buffer_budget: u64,
}

impl PreparedComputeDescriptor<'_> {
    /// Logical bytes for input, parameters, output, upload, and readback.
    pub fn buffer_bytes(&self) -> Result<u64, PreparedComputeError> {
        self.input_size
            .checked_add(self.parameter_size)
            .and_then(|size| size.checked_add(self.output_size))
            .and_then(|size| size.checked_mul(2))
            .ok_or(PreparedComputeError::InvalidDescriptor)
    }

    /// Validate capacities and dispatch dimensions before creating resources.
    pub fn validate(&self, limits: &crate::Limits) -> Result<(), PreparedComputeError> {
        let sizes = [self.input_size, self.parameter_size, self.output_size];
        if self.shader.is_empty()
            || self.entry_point.is_empty()
            || sizes.iter().any(|&size| {
                size == 0
                    || !size.is_multiple_of(crate::COPY_BUFFER_ALIGNMENT)
                    || size > isize::MAX as u64
                    || size > limits.max_buffer_size
            })
            || self.input_size > limits.max_storage_buffer_binding_size
            || self.output_size > limits.max_storage_buffer_binding_size
            || self.parameter_size > limits.max_uniform_buffer_binding_size
            || self
                .workgroups
                .iter()
                .any(|&count| count == 0 || count > limits.max_compute_workgroups_per_dimension)
        {
            return Err(PreparedComputeError::InvalidDescriptor);
        }
        let upload_size = self
            .input_size
            .checked_add(self.parameter_size)
            .ok_or(PreparedComputeError::InvalidDescriptor)?;
        if upload_size > limits.max_buffer_size || upload_size > isize::MAX as u64 {
            return Err(PreparedComputeError::InvalidDescriptor);
        }
        if self.buffer_bytes()? > self.buffer_budget {
            return Err(PreparedComputeError::Capacity);
        }
        Ok(())
    }
}

/// Fixed-size prepared execution errors; producing these values does not format text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreparedComputeError {
    /// Input/output capacity or the setup buffer budget was exceeded.
    Capacity,
    /// A descriptor, parameter length, or shader is invalid.
    InvalidDescriptor,
    /// This backend does not implement prepared compute.
    UnsupportedBackend,
    /// The configuration is incompatible with the prepared execution contract.
    UnsupportedConfiguration,
    /// The owner has exhausted its nonwrapping execution sequence.
    GenerationExhausted,
    /// Waiting timed out; GPU work may still be running and the owner is poisoned.
    Timeout,
    /// The device was lost.
    DeviceLost,
    /// The backend reported insufficient memory.
    OutOfMemory,
    /// The backend reported another failure.
    Unexpected,
    /// An earlier execution failure permanently retired this owner.
    Poisoned,
}

impl fmt::Display for PreparedComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Capacity => "prepared compute capacity exceeded",
            Self::InvalidDescriptor => "invalid prepared compute descriptor or parameters",
            Self::UnsupportedBackend => "prepared compute is unsupported on this backend",
            Self::UnsupportedConfiguration => "unsupported prepared compute configuration",
            Self::GenerationExhausted => "prepared compute sequence exhausted",
            Self::Timeout => "prepared compute timed out",
            Self::DeviceLost => "prepared compute device lost",
            Self::OutOfMemory => "prepared compute out of memory",
            Self::Unexpected => "prepared compute backend failure",
            Self::Poisoned => "prepared compute owner is poisoned",
        })
    }
}

impl core::error::Error for PreparedComputeError {}

/// A cold preparation error, including an optional validation diagnostic.
#[derive(Clone, Debug)]
pub struct PrepareComputeError {
    /// Machine-readable failure category.
    pub reason: PreparedComputeError,
    /// Preparation diagnostic. Unlike execution, preparation may allocate.
    pub message: String,
}

impl From<PreparedComputeError> for PrepareComputeError {
    fn from(reason: PreparedComputeError) -> Self {
        Self {
            reason,
            message: String::new(),
        }
    }
}

impl fmt::Display for PrepareComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.reason, self.message)
    }
}

impl core::error::Error for PrepareComputeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> PreparedComputeDescriptor<'static> {
        PreparedComputeDescriptor {
            shader: "@compute @workgroup_size(1) fn main() {}",
            entry_point: "main",
            input_size: 64,
            parameter_size: 16,
            output_size: 64,
            workgroups: [1; 3],
            buffer_budget: 288,
        }
    }

    #[test]
    fn budget_includes_upload_and_readback() {
        let mut desc = descriptor();
        assert_eq!(desc.buffer_bytes(), Ok(288));
        assert_eq!(desc.validate(&crate::Limits::default()), Ok(()));
        desc.buffer_budget -= 1;
        assert_eq!(
            desc.validate(&crate::Limits::default()),
            Err(PreparedComputeError::Capacity)
        );
    }

    #[test]
    fn invalid_sizes_dispatch_and_overflow_are_rejected() {
        for size in [0, 1, 3, u64::MAX] {
            let mut desc = descriptor();
            desc.input_size = size;
            assert_eq!(
                desc.validate(&crate::Limits::default()),
                Err(PreparedComputeError::InvalidDescriptor)
            );
        }
        let mut desc = descriptor();
        desc.workgroups[0] = 0;
        assert_eq!(
            desc.validate(&crate::Limits::default()),
            Err(PreparedComputeError::InvalidDescriptor)
        );
        desc.input_size = u64::MAX;
        assert_eq!(
            desc.buffer_bytes(),
            Err(PreparedComputeError::InvalidDescriptor)
        );
    }
}
