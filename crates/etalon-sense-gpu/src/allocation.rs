/// Requested allocation class for a sensing GPU provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuAllocationClass {
    /// Strict Tier-1 class. The current Etalon command and map APIs cannot
    /// satisfy it, so providers must reject this class before allocating.
    HotZeroHeap,
    /// Cold execution with persistent provider-owned GPU buffers admitted by a byte budget.
    /// Backend, pipeline, and per-command allocations remain outside that budget.
    ColdBoundedBuffers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdmissionError {
    HotPathUnsupported,
    BudgetExceeded,
}

pub(super) fn admit(
    class: GpuAllocationClass,
    budget_bytes: u64,
    required_buffer_bytes: u64,
) -> Result<(), AdmissionError> {
    if class == GpuAllocationClass::HotZeroHeap {
        return Err(AdmissionError::HotPathUnsupported);
    }
    if required_buffer_bytes > budget_bytes {
        return Err(AdmissionError::BudgetExceeded);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{admit, AdmissionError, GpuAllocationClass};

    #[test]
    fn strict_zero_heap_is_rejected_even_with_budget() {
        assert_eq!(
            admit(GpuAllocationClass::HotZeroHeap, u64::MAX, 1),
            Err(AdmissionError::HotPathUnsupported),
        );
    }

    #[test]
    fn cold_class_requires_buffer_budget() {
        assert_eq!(
            admit(GpuAllocationClass::ColdBoundedBuffers, 99, 100),
            Err(AdmissionError::BudgetExceeded),
        );
        assert_eq!(admit(GpuAllocationClass::ColdBoundedBuffers, 100, 100), Ok(()));
    }
}
