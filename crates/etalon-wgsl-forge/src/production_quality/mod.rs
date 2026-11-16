//! Kernel-schedule quality gates over existing Forge `schedule` / `autotune`.
//!
//! Occupancy and timing reports are admissions/measurements with method labels,
//! never synthetic GPU timing. Public `KernelSpec` / `Schedule` / `tune_with`
//! APIs stay unchanged.
//!
//! Integrated decode/upload/compaction/draw accounting is out of scope (P22).

pub mod costs;
pub mod gates;
pub mod schedule_gate;
pub mod search;

#[cfg(test)]
mod tests;

pub use costs::{
    break_even, BreakEvenReport, BreakEvenStatus, CostPresence, PhaseCosts,
    INTEGRATED_PHASE_COST_NOTE,
};
pub use gates::{
    admit_oracle, crc_is_not_a_signature, parse_content_digest, FrozenQualityGates,
    QualityThresholds, FROZEN_MAX_ABSOLUTE_ERROR, FROZEN_MAX_ORACLE_MISMATCHES,
    FROZEN_MAX_RELATIVE_ERROR, FROZEN_MAX_SHARED_BYTES, FROZEN_MAX_WORKGROUP_THREADS,
    LEARNED_SEARCH_EXCLUDED_NOTE,
};
pub use schedule_gate::{
    admit_schedule, admit_timing, gate_tuning, label_timing, labeled_timing_from_occupancy,
    occupancy_report_as_gpu_timestamp, record_generated_search, record_search_space,
    record_tuning_result, require_timing_label, CandidateRecordStatus, CandidateSearchRecord,
    LabeledTiming, RecordedCandidate, ScheduleGateReport, ScheduleGateRequest, TuningQualityReport,
    WorkgroupAdmission, MAX_RECORDED_CANDIDATES,
};
pub use search::{
    gated_search, ranking_key, AdmittedBaseline, GatedSearchRequest, GatedTuneReport, ParetoPoint,
    SearchBudget, WinnerSource,
};

pub use crate::production_ir::Digest;
pub use crate::production_lowering::{
    DeclaredOccupancy, OccupancyAssumption, OccupancyReport, SharedMemoryAdmission,
    SubgroupAdmission, OCCUPANCY_NOTE, PORTABLE_MAX_SHARED_BYTES,
};

pub const PRODUCTION_QUALITY_SCHEMA_ID: &str = "etalon-production-quality-v1";
pub const PRODUCTION_QUALITY_SCHEMA_V1: &[u8] = PRODUCTION_QUALITY_SCHEMA_ID.as_bytes();
pub const PRODUCTION_QUALITY_SCHEMA_VERSION: u32 = 1;
