use super::{permitted, PreparedIqPower, PreparedPowerError, PreparedPowerStage};
use etalon_sense_core::{
    buffer::SampleLayout,
    processing_runtime::{
        LiveProcessingProvider, LiveProviderDescriptor, ProviderDispatchGrant, ProviderLifecycle,
        ProviderReport,
    },
    quantity::SampleEncoding,
    ContractError,
};
use etalon_signal::{decode::ByteOrder, execution::i16_iq_power_upper_bound};

impl PreparedIqPower<'_> {
    /// Output is one LE f64 per I/Q pair, computed in f32. Workspace needs 4*N
    /// bytes. Its used prefix is wiped on return. The full output is cleared
    /// on failure. No physical unit, calibration or execution attestation is added.
    pub fn compute_into(
        &mut self,
        input: &[u8],
        order: ByteOrder,
        scale: f32,
        workspace: &mut [u8],
        output: &mut [u8],
    ) -> Result<usize, PreparedPowerError> {
        output.fill(0);
        self.last_error = None;
        let count = preflight(input, scale, workspace.len(), output.len(), self.max_pairs)
            .inspect_err(|error| self.last_error = Some(*error))?;
        let used = count * 4;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.compute_inner(input, order, scale, &mut workspace[..used], output, count)
        }))
        .unwrap_or(Err(PreparedPowerError::Poisoned));
        workspace[..used].fill(0);
        match result {
            Ok(count) => {
                self.last_error = None;
                Ok(count)
            }
            Err(error) => {
                output.fill(0);
                self.faulted = true;
                self.last_error = Some(error);
                Err(error)
            }
        }
    }

    fn compute_inner(
        &mut self,
        input: &[u8],
        order: ByteOrder,
        scale: f32,
        workspace: &mut [u8],
        output: &mut [u8],
        count: usize,
    ) -> Result<usize, PreparedPowerError> {
        if self.is_poisoned() {
            return Err(PreparedPowerError::Poisoned);
        }
        let upper =
            i16_iq_power_upper_bound(scale).map_err(|_| PreparedPowerError::InvalidScale)?;
        let scope = self.scope(Some(count), Some(order), Some(scale));
        if !permitted(self.gate, scope, PreparedPowerStage::BeforeDispatch) {
            return Err(PreparedPowerError::PolicyDenied);
        }
        let parameters = crate::numeric::make_params(scale, order, count);
        self.lane
            .execute(input, &parameters, workspace, self.timeout)
            .map_err(PreparedPowerError::Backend)?;
        for bytes in workspace.chunks_exact(4) {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if !value.is_finite() || value < 0.0 || f64::from(value) > upper {
                return Err(PreparedPowerError::NumericRange);
            }
        }
        for (bytes, destination) in workspace.chunks_exact(4).zip(output.chunks_exact_mut(8)) {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            destination.copy_from_slice(&f64::from(value).to_le_bytes());
        }
        if !permitted(self.gate, scope, PreparedPowerStage::AfterDispatch) {
            return Err(PreparedPowerError::PolicyDenied);
        }
        Ok(count)
    }
}

pub(super) fn preflight(
    input: &[u8],
    scale: f32,
    workspace: usize,
    output: usize,
    capacity: usize,
) -> Result<usize, PreparedPowerError> {
    if !scale.is_finite() {
        return Err(PreparedPowerError::InvalidScale);
    }
    if input.is_empty() || !input.len().is_multiple_of(4) {
        return Err(PreparedPowerError::InvalidInput);
    }
    let count = input.len() / 4;
    if count > capacity || workspace < count * 4 || output != count * 8 {
        return Err(PreparedPowerError::Capacity);
    }
    Ok(count)
}

impl LiveProcessingProvider for PreparedIqPower<'_> {
    fn descriptor(&self) -> LiveProviderDescriptor {
        self.descriptor
    }
    fn lifecycle(&self) -> ProviderLifecycle {
        if self.is_poisoned() {
            ProviderLifecycle::Faulted
        } else {
            ProviderLifecycle::Ready
        }
    }
    fn dispatch(
        &mut self,
        grant: ProviderDispatchGrant,
        input_layout: &SampleLayout,
        input: &[u8],
        output_layout: &SampleLayout,
        output: &mut [u8],
        workspace: &mut [u8],
    ) -> Result<ProviderReport, ContractError> {
        output.fill(0);
        if self.lifecycle() != ProviderLifecycle::Ready
            || grant.provider_id() != self.descriptor.provider_id()
            || grant.generation() != self.descriptor.generation()
            || grant.operation() != self.descriptor.operation()
        {
            return Err(ContractError::InvalidSource);
        }
        validate_layouts(
            input_layout,
            input.len(),
            output_layout,
            output.len(),
            self.max_pairs,
        )?;
        self.compute_into(input, ByteOrder::LittleEndian, 1.0, workspace, output)
            .map_err(|_| ContractError::InvalidSource)?;
        Ok(ProviderReport::new(
            self.descriptor.provider_id(),
            self.descriptor.generation(),
            self.descriptor.operation(),
            input.len() as u64,
            output.len() as u64,
        ))
    }
}

pub(super) fn validate_layouts(
    input: &SampleLayout,
    input_len: usize,
    output: &SampleLayout,
    output_len: usize,
    capacity: usize,
) -> Result<(), ContractError> {
    if input.rank() != 1
        || output.rank() != 1
        || *input.encoding() != SampleEncoding::ComplexI16
        || *output.encoding() != SampleEncoding::F64
        || input.extents()[0] != output.extents()[0]
    {
        return Err(ContractError::IncompatibleEncoding);
    }
    let count = usize::try_from(input.extents()[0]).map_err(|_| ContractError::OutOfBounds)?;
    if count == 0 || count > capacity {
        return Err(ContractError::OutOfBounds);
    }
    if count > 1 && (input.strides_bytes()[0] != 4 || output.strides_bytes()[0] != 8) {
        return Err(ContractError::IncompatibleEncoding);
    }
    let input_bytes = count
        .checked_mul(4)
        .ok_or(ContractError::ArithmeticOverflow)?;
    let output_bytes = count
        .checked_mul(8)
        .ok_or(ContractError::ArithmeticOverflow)?;
    if input_len != input_bytes || output_len != output_bytes {
        return Err(ContractError::OutOfBounds);
    }
    Ok(())
}
