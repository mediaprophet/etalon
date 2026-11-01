use super::{
    preparation_permitted, PreparePowerError, PreparedIqPower, PreparedPowerGate,
    PreparedPowerScope, PreparedPowerStage,
};
use crate::{
    live::{GPU_IQ_POWER_F32_OPERATION_ID, GPU_IQ_POWER_F32_REVISION},
    MAX_IQ_PAIRS, MAX_POLL_TIMEOUT,
};
use etalon_sense_core::{
    processing::OperationId,
    processing::ProcessorKind,
    processing_runtime::{AllocationClass, LiveProviderDescriptor},
    quantity::SampleEncoding,
};
use etalon_signal::execution::DeviceId;
use std::time::Duration;

pub(super) const SHADER: &str = include_str!("iq_power.wgsl");

impl<'gate> PreparedIqPower<'gate> {
    /// Prepare on the explicitly selected adapter. Requires native Vulkan,
    /// WGSL, and instance diagnostic flags disabled, as required by Etalon.
    /// The budget covers five logical buffers (16*N+32), not total GPU memory.
    pub fn new(
        adapter: &etalon::Adapter,
        device_id: DeviceId,
        generation: u32,
        max_pairs: usize,
        buffer_budget: u64,
        timeout: Duration,
        gate: &'gate mut dyn PreparedPowerGate,
    ) -> Result<Self, PreparePowerError> {
        let logical_buffer_bytes = buffer_bytes(max_pairs)?;
        if timeout.is_zero() || timeout > MAX_POLL_TIMEOUT {
            return Err(PreparePowerError::InvalidTimeout);
        }
        if logical_buffer_bytes > buffer_budget {
            return Err(PreparePowerError::BudgetExceeded);
        }
        let descriptor = descriptor(device_id, generation, max_pairs)?;
        let scope = PreparedPowerScope {
            device_id,
            generation,
            capacity: max_pairs,
            pair_count: None,
            byte_order: None,
            scale: None,
        };
        if !preparation_permitted(gate, adapter, scope, PreparedPowerStage::BeforePrepare) {
            return Err(PreparePowerError::PolicyDenied);
        }
        let lane = adapter
            .prepare_compute(&etalon::PreparedComputeDescriptor {
                shader: SHADER,
                entry_point: "main",
                input_size: (max_pairs * 4) as u64,
                parameter_size: 16,
                output_size: (max_pairs * 4) as u64,
                workgroups: [max_pairs.div_ceil(64) as u32, 1, 1],
                buffer_budget,
            })
            .map_err(PreparePowerError::Backend)?;
        if !preparation_permitted(gate, adapter, scope, PreparedPowerStage::AfterPrepare) {
            return Err(PreparePowerError::PolicyDenied);
        }
        Ok(Self {
            lane,
            gate,
            descriptor,
            device_id,
            max_pairs,
            logical_buffer_bytes,
            timeout,
            faulted: false,
            last_error: None,
        })
    }
}

pub(super) fn buffer_bytes(max_pairs: usize) -> Result<u64, PreparePowerError> {
    if max_pairs == 0 || max_pairs > MAX_IQ_PAIRS {
        return Err(PreparePowerError::InvalidCapacity);
    }
    Ok((max_pairs as u64) * 16 + 32)
}

pub(super) fn descriptor(
    device_id: DeviceId,
    generation: u32,
    max_pairs: usize,
) -> Result<LiveProviderDescriptor, PreparePowerError> {
    buffer_bytes(max_pairs)?;
    let operation = OperationId::new(GPU_IQ_POWER_F32_OPERATION_ID, GPU_IQ_POWER_F32_REVISION)
        .map_err(|_| PreparePowerError::InvalidIdentity)?;
    LiveProviderDescriptor::new(
        device_id.value(),
        generation,
        ProcessorKind::Gpu,
        operation,
        SampleEncoding::ComplexI16,
        SampleEncoding::F64,
        1,
        [max_pairs as u32, 0, 0, 0],
        (max_pairs * 4) as u64,
        AllocationClass::ColdUnbounded,
        None,
    )
    .map_err(|_| PreparePowerError::InvalidIdentity)
}
