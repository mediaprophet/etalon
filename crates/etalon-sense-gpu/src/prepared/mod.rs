//! IQ power on a private prepared Vulkan lane. Setup/retirement are cold.
//! Policy callbacks and driver allocation remain outside a hard memory bound;
//! the live descriptor therefore remains ColdUnbounded pending qualification.
mod execution;
mod setup;

use etalon_sense_core::processing_runtime::LiveProviderDescriptor;
use etalon_signal::{decode::ByteOrder, execution::DeviceId};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedPowerStage {
    BeforePrepare,
    AfterPrepare,
    BeforeDispatch,
    AfterDispatch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedPowerScope {
    pub device_id: DeviceId,
    pub generation: u32,
    pub capacity: usize,
    pub pair_count: Option<usize>,
    pub byte_order: Option<ByteOrder>,
    pub scale: Option<f32>,
}

/// Hosts bind the declared adapter ID and exact operation to an immutable
/// authority lease. Gates must not unwind; checks cannot cancel submitted work.
pub trait PreparedPowerGate {
    /// Inspect the actual selected adapter before/after cold preparation.
    fn permit_preparation(
        &mut self,
        adapter: &etalon::Adapter,
        scope: PreparedPowerScope,
        stage: PreparedPowerStage,
    ) -> bool;
    fn permit(&mut self, scope: PreparedPowerScope, stage: PreparedPowerStage) -> bool;
}

#[derive(Debug)]
pub enum PreparePowerError {
    InvalidCapacity,
    InvalidTimeout,
    BudgetExceeded,
    InvalidIdentity,
    PolicyDenied,
    Backend(etalon::PrepareComputeError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedPowerError {
    InvalidInput,
    InvalidScale,
    Capacity,
    PolicyDenied,
    NumericRange,
    Poisoned,
    Backend(etalon::PreparedComputeError),
}

/// Fixed resources with caller-owned per-dispatch readback/output storage.
/// Device IDs and generation are host declarations, not hardware attestation.
pub struct PreparedIqPower<'gate> {
    lane: etalon::PreparedCompute,
    gate: &'gate mut dyn PreparedPowerGate,
    descriptor: LiveProviderDescriptor,
    device_id: DeviceId,
    max_pairs: usize,
    logical_buffer_bytes: u64,
    timeout: Duration,
    faulted: bool,
    last_error: Option<PreparedPowerError>,
}

impl PreparedIqPower<'_> {
    pub const fn max_pairs(&self) -> usize {
        self.max_pairs
    }
    pub const fn logical_buffer_bytes(&self) -> u64 {
        self.logical_buffer_bytes
    }
    pub const fn last_error(&self) -> Option<PreparedPowerError> {
        self.last_error
    }
    pub fn completed_runs(&self) -> u64 {
        self.lane.completed_runs()
    }
    pub fn is_poisoned(&self) -> bool {
        self.faulted || self.lane.is_poisoned()
    }

    /// Cold retirement may block. A failed retirement retains resources, and
    /// the underlying owner quarantines them if drop cannot establish safety.
    pub fn retire(&mut self) -> Result<(), etalon::PreparedComputeError> {
        self.faulted = true;
        self.lane.retire()
    }

    fn scope(
        &self,
        count: Option<usize>,
        order: Option<ByteOrder>,
        scale: Option<f32>,
    ) -> PreparedPowerScope {
        PreparedPowerScope {
            device_id: self.device_id,
            generation: self.descriptor.generation(),
            capacity: self.max_pairs,
            pair_count: count,
            byte_order: order,
            scale,
        }
    }
}

fn permitted(
    gate: &mut dyn PreparedPowerGate,
    scope: PreparedPowerScope,
    stage: PreparedPowerStage,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gate.permit(scope, stage)))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests;

impl std::fmt::Display for PreparePowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PreparePowerError {}
impl std::fmt::Display for PreparedPowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PreparedPowerError {}

fn preparation_permitted(
    gate: &mut dyn PreparedPowerGate,
    adapter: &etalon::Adapter,
    scope: PreparedPowerScope,
    stage: PreparedPowerStage,
) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gate.permit_preparation(adapter, scope, stage)
    }))
    .unwrap_or(false)
}

impl std::fmt::Debug for PreparedIqPower<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedIqPower")
            .field("descriptor", &self.descriptor)
            .field("logical_buffer_bytes", &self.logical_buffer_bytes)
            .field("poisoned", &self.is_poisoned())
            .finish_non_exhaustive()
    }
}
