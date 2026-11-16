//! Production WGSL lowering and admission over existing Forge emit.
//!
//! This is not a second compiler. Public `KernelSpec` / `emit_wgsl` / `Schedule`
//! APIs stay unchanged. Native WGSL `f64` is not claimed; existing df64 is
//! paired f32.

pub mod admission;
pub mod wgsl;

#[cfg(test)]
mod tests;

pub use admission::{
    admit, layout_from_contract, precision_from_contract, AdmissionReport, AdmissionRequest,
    DeclaredOccupancy, FallbackKind, FallbackRecord, LayoutKind, OccupancyAssumption,
    OccupancyReport, PrecisionAdmission, RequestedPrecision,
    SharedMemoryAdmission, SubgroupAdmission, CLUSTER_NODE_STRIDE_SCALARS, DF64_PAIR_STRIDE,
    GEOMETRY_PACKED_QUERY_STRIDE, MAX_FALLBACKS, OCCUPANCY_NOTE, PORTABLE_MAX_SHARED_BYTES,
};
pub use wgsl::{
    lower_contract, BindingDecl, EmitterKind, LoweredWgsl, NagaStep, MAX_BINDINGS,
};

use crate::ir::ForgeError;
use crate::production_ir::ProductionContract;
use crate::schedule::AdapterConstraints;

pub const PRODUCTION_LOWERING_SCHEMA_ID: &str = "etalon-production-lowering-v1";
pub const PRODUCTION_LOWERING_SCHEMA_V1: &[u8] = PRODUCTION_LOWERING_SCHEMA_ID.as_bytes();
pub const PRODUCTION_LOWERING_SCHEMA_VERSION: u32 = 1;

/// Occupancy / precision / layout overrides applied after contract-derived defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionOverride {
    pub precision: Option<RequestedPrecision>,
    pub layout: Option<LayoutKind>,
    pub occupancy: OccupancyAssumption,
    pub subgroup_required: bool,
    pub assumed_subgroup_size: Option<u32>,
}

impl AdmissionOverride {
    pub fn from_contract(_contract: &ProductionContract) -> Self {
        Self {
            precision: None,
            layout: None,
            occupancy: OccupancyAssumption::None,
            subgroup_required: false,
            assumed_subgroup_size: None,
        }
    }
}

/// Lowers a production contract through existing emit, then runs P14 admission.
pub fn lower_admitted(
    contract: &ProductionContract,
    constraints: &AdapterConstraints,
    override_req: &AdmissionOverride,
    bindings_out: &mut [BindingDecl],
    fallbacks_out: &mut [FallbackRecord],
) -> Result<(LoweredWgsl, AdmissionReport), ForgeError> {
    let lowered = lower_contract(contract, constraints, bindings_out)?;
    let request = AdmissionRequest {
        contract,
        constraints,
        precision: override_req
            .precision
            .unwrap_or_else(|| precision_from_contract(contract)),
        layout: override_req
            .layout
            .unwrap_or_else(|| layout_from_contract(contract)),
        occupancy: override_req.occupancy,
        subgroup_required: override_req.subgroup_required,
        assumed_subgroup_size: override_req.assumed_subgroup_size,
    };
    let report = admit(&request, Some(&lowered.shader.source), fallbacks_out)?;
    Ok((lowered, report))
}
