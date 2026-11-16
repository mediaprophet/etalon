//! Kernel-schedule quality gates over existing Forge `schedule` / `autotune`.
//!
//! Occupancy and timing reports are admissions/measurements with method labels,
//! never synthetic GPU timing. Public `KernelSpec` / `Schedule` / `tune_with`
//! APIs stay unchanged.
//!
//! Integrated decode/upload/compaction/draw accounting is out of scope (P22).

pub mod schedule_gate;

#[cfg(test)]
mod tests;

pub use schedule_gate::{
    admit_schedule, admit_timing, gate_tuning, label_timing, record_generated_search,
    record_search_space, record_tuning_result, require_timing_label, CandidateRecordStatus,
    CandidateSearchRecord, LabeledTiming, RecordedCandidate, ScheduleGateReport,
    ScheduleGateRequest, TuningQualityReport, WorkgroupAdmission, MAX_RECORDED_CANDIDATES,
};

pub use crate::production_lowering::{
    DeclaredOccupancy, OccupancyAssumption, OccupancyReport, SharedMemoryAdmission,
    SubgroupAdmission, OCCUPANCY_NOTE, PORTABLE_MAX_SHARED_BYTES,
};

pub const PRODUCTION_QUALITY_SCHEMA_ID: &str = "etalon-production-quality-v1";
pub const PRODUCTION_QUALITY_SCHEMA_V1: &[u8] = PRODUCTION_QUALITY_SCHEMA_ID.as_bytes();
pub const PRODUCTION_QUALITY_SCHEMA_VERSION: u32 = 1;
