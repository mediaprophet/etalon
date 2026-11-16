//! Admit schedules, label timings, and bound autotune candidate search.

use serde::{Deserialize, Serialize};

use crate::autotune::{generate_search_candidates, TuningResult};
use crate::cache::{TimingSource, TimingSummary};
use crate::ir::{ForgeError, IntrinsicClass, KernelSpec};
use crate::production_lowering::{
    DeclaredOccupancy, OccupancyAssumption, OccupancyReport, SharedMemoryAdmission,
    SubgroupAdmission, OCCUPANCY_NOTE, PORTABLE_MAX_SHARED_BYTES,
};
use crate::schedule::{AdapterConstraints, Schedule, ScheduleSpace};

use super::PRODUCTION_QUALITY_SCHEMA_VERSION;

/// Matches `TuningConfig::default().max_candidates`. Overflow is fail-closed.
pub const MAX_RECORDED_CANDIDATES: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScheduleGateRequest<'a> {
    pub kernel: &'a KernelSpec,
    pub schedule: Schedule,
    pub constraints: &'a AdapterConstraints,
    pub occupancy: OccupancyAssumption,
    pub subgroup_required: bool,
    pub assumed_subgroup_size: Option<u32>,
    pub element_count: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkgroupAdmission {
    pub threads: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub warp_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleGateReport {
    pub schema_version: u32,
    pub accepted: bool,
    pub schedule: Schedule,
    pub workgroup: WorkgroupAdmission,
    pub shared_memory: SharedMemoryAdmission,
    pub subgroup: SubgroupAdmission,
    pub occupancy: OccupancyReport,
}

/// Timing measurement with an explicit `TimingSource` label.
/// Occupancy estimates must not be stored in this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabeledTiming {
    pub source: TimingSource,
    pub sample_count: usize,
    pub minimum_ns: u64,
    pub median_ns: u64,
    pub p95_ns: u64,
}

impl LabeledTiming {
    pub fn from_summary(summary: TimingSummary) -> Self {
        Self {
            source: summary.source,
            sample_count: summary.sample_count,
            minimum_ns: summary.minimum_ns,
            median_ns: summary.median_ns,
            p95_ns: summary.p95_ns,
        }
    }

    pub fn summary(&self) -> TimingSummary {
        TimingSummary {
            source: self.source,
            sample_count: self.sample_count,
            minimum_ns: self.minimum_ns,
            median_ns: self.median_ns,
            p95_ns: self.p95_ns,
        }
    }

    /// GPU timing is only the `GpuTimestamp` method, never synthetic values.
    pub fn as_gpu_timing(&self) -> Result<&Self, ForgeError> {
        require_timing_label(self.source, TimingSource::GpuTimestamp)?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRecordStatus {
    Searched,
    Admitted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedCandidate {
    pub schedule: Schedule,
    pub status: CandidateRecordStatus,
    pub timing: Option<LabeledTiming>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSearchRecord {
    pub schema_version: u32,
    pub capacity: usize,
    pub recorded: usize,
    pub overflowed: bool,
    pub candidates: Vec<RecordedCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningQualityReport {
    pub schema_version: u32,
    pub schedule: ScheduleGateReport,
    pub winner_timing: LabeledTiming,
    pub search: CandidateSearchRecord,
}

pub fn admit_schedule(
    request: &ScheduleGateRequest<'_>,
) -> Result<ScheduleGateReport, ForgeError> {
    if matches!(request.occupancy, OccupancyAssumption::Undeclared) {
        return Err(ForgeError::Validation(
            "undeclared occupancy assumption is rejected; occupancy is an admission/report, not measured GPU timing".to_string(),
        ));
    }
    request
        .schedule
        .validate(request.kernel, request.constraints)?;
    request.constraints.supports_kernel(request.kernel)?;

    let workgroup = admit_workgroup(request.schedule, request.constraints)?;
    if let Some(count) = request.element_count {
        let groups = request.schedule.dispatch_workgroups(count);
        if groups > request.constraints.max_workgroups_per_dimension {
            return Err(ForgeError::InvalidSchedule(format!(
                "dispatch {groups} exceeds adapter max_workgroups_per_dimension {}",
                request.constraints.max_workgroups_per_dimension
            )));
        }
    }
    let shared_memory = admit_shared_memory(request.kernel, workgroup.threads)?;
    let occupancy = admit_occupancy(
        request.occupancy,
        workgroup.threads,
        shared_memory.declared_bytes,
        workgroup.warp_size,
    )?;
    let subgroup = admit_subgroup(request)?;

    Ok(ScheduleGateReport {
        schema_version: PRODUCTION_QUALITY_SCHEMA_VERSION,
        accepted: true,
        schedule: request.schedule,
        workgroup,
        shared_memory,
        subgroup,
        occupancy,
    })
}

/// Labels samples with the true method. Does not relabel synthetic as GPU.
pub fn label_timing(source: TimingSource, samples: &[u64]) -> Result<LabeledTiming, ForgeError> {
    let summary = TimingSummary::from_samples(source, samples).ok_or_else(|| {
        ForgeError::Validation("timing produced no samples".to_string())
    })?;
    Ok(LabeledTiming::from_summary(summary))
}

/// Stores a timing summary only when `claimed` matches the labeled source.
/// Synthetic values cannot be stored as `TimingSource::GpuTimestamp`.
pub fn admit_timing(
    summary: &TimingSummary,
    claimed: TimingSource,
) -> Result<LabeledTiming, ForgeError> {
    require_timing_label(summary.source, claimed)?;
    if summary.sample_count == 0 {
        return Err(ForgeError::Validation(
            "timing produced no samples".to_string(),
        ));
    }
    Ok(LabeledTiming {
        source: summary.source,
        sample_count: summary.sample_count,
        minimum_ns: summary.minimum_ns,
        median_ns: summary.median_ns,
        p95_ns: summary.p95_ns,
    })
}

pub fn require_timing_label(actual: TimingSource, claimed: TimingSource) -> Result<(), ForgeError> {
    if claimed == TimingSource::GpuTimestamp && actual != TimingSource::GpuTimestamp {
        if actual == TimingSource::Synthetic {
            return Err(ForgeError::Validation(
                "synthetic timing cannot be presented as GPU timing".to_string(),
            ));
        }
        return Err(ForgeError::Validation(
            "synthetic or completion-clock timing cannot be presented as GPU timing".to_string(),
        ));
    }
    if claimed != actual {
        return Err(ForgeError::Validation(
            "timing source does not match claimed method label".to_string(),
        ));
    }
    Ok(())
}

pub fn record_search_space(
    kernel: &KernelSpec,
    constraints: &AdapterConstraints,
    space: &ScheduleSpace,
    capacity: usize,
) -> Result<CandidateSearchRecord, ForgeError> {
    record_schedules(space.candidates(kernel, constraints), capacity)
}

pub fn record_generated_search(capacity: usize) -> Result<CandidateSearchRecord, ForgeError> {
    record_schedules(generate_search_candidates(), capacity)
}

pub fn record_tuning_result(
    result: &TuningResult,
    capacity: usize,
) -> Result<CandidateSearchRecord, ForgeError> {
    let capacity = require_capacity(capacity)?;
    if result.evaluated_candidates > capacity {
        return Err(overflow_error(result.evaluated_candidates, capacity));
    }

    let mut candidates = Vec::new();
    for finalist in &result.finalists {
        let timing = admit_timing(&finalist.timing, finalist.timing.source)?;
        push_unique(
            &mut candidates,
            capacity,
            RecordedCandidate {
                schedule: finalist.schedule,
                status: CandidateRecordStatus::Admitted,
                timing: Some(timing),
                reason: None,
            },
        )?;
    }
    if !contains_schedule(&candidates, result.winner.schedule) {
        let timing = admit_timing(&result.winner.timing, result.winner.timing.source)?;
        push_unique(
            &mut candidates,
            capacity,
            RecordedCandidate {
                schedule: result.winner.schedule,
                status: CandidateRecordStatus::Admitted,
                timing: Some(timing),
                reason: None,
            },
        )?;
    } else {
        admit_timing(&result.winner.timing, result.winner.timing.source)?;
    }
    for failure in &result.failures {
        if contains_schedule(&candidates, failure.schedule) {
            continue;
        }
        push_unique(
            &mut candidates,
            capacity,
            RecordedCandidate {
                schedule: failure.schedule,
                status: CandidateRecordStatus::Rejected,
                timing: None,
                reason: Some(failure.reason.clone()),
            },
        )?;
    }

    Ok(search_record(capacity, candidates))
}

pub fn gate_tuning(
    kernel: &KernelSpec,
    constraints: &AdapterConstraints,
    occupancy: OccupancyAssumption,
    result: &TuningResult,
    capacity: usize,
) -> Result<TuningQualityReport, ForgeError> {
    let request = ScheduleGateRequest {
        kernel,
        schedule: result.winner.schedule,
        constraints,
        occupancy,
        subgroup_required: false,
        assumed_subgroup_size: None,
        element_count: None,
    };
    let schedule = admit_schedule(&request)?;
    let winner_timing = admit_timing(&result.winner.timing, result.winner.timing.source)?;
    let search = record_tuning_result(result, capacity)?;
    Ok(TuningQualityReport {
        schema_version: PRODUCTION_QUALITY_SCHEMA_VERSION,
        schedule,
        winner_timing,
        search,
    })
}

fn admit_workgroup(
    schedule: Schedule,
    constraints: &AdapterConstraints,
) -> Result<WorkgroupAdmission, ForgeError> {
    let threads = workgroup_threads(&schedule);
    if threads == 0 {
        return Err(ForgeError::InvalidSchedule(
            "workgroup thread count must be nonzero".to_string(),
        ));
    }
    if schedule.workgroup.x > constraints.max_workgroup_size_x {
        return Err(ForgeError::InvalidSchedule(format!(
            "workgroup x {} exceeds adapter limit {}",
            schedule.workgroup.x, constraints.max_workgroup_size_x
        )));
    }
    if threads > constraints.max_invocations_per_workgroup {
        return Err(ForgeError::InvalidSchedule(format!(
            "workgroup {threads} exceeds adapter limit {}",
            constraints.max_invocations_per_workgroup
        )));
    }
    let warp = constraints.warp_size.max(1);
    if warp > 1 && threads % warp != 0 {
        return Err(ForgeError::InvalidSchedule(format!(
            "workgroup threads {threads} are not aligned to adapter warp {warp}"
        )));
    }
    Ok(WorkgroupAdmission {
        threads,
        x: schedule.workgroup.x,
        y: schedule.workgroup.y,
        z: schedule.workgroup.z,
        warp_size: warp,
    })
}

fn admit_shared_memory(
    kernel: &KernelSpec,
    workgroup_threads: u32,
) -> Result<SharedMemoryAdmission, ForgeError> {
    let declared_bytes = shared_bytes(kernel, workgroup_threads)?;
    if declared_bytes > PORTABLE_MAX_SHARED_BYTES {
        return Err(ForgeError::InvalidSchedule(format!(
            "shared memory {declared_bytes} bytes exceeds portable limit {PORTABLE_MAX_SHARED_BYTES}"
        )));
    }
    Ok(SharedMemoryAdmission {
        declared_bytes,
        limit_bytes: PORTABLE_MAX_SHARED_BYTES,
    })
}

fn admit_occupancy(
    occupancy: OccupancyAssumption,
    threads: u32,
    shared_bytes: u32,
    warp: u32,
) -> Result<OccupancyReport, ForgeError> {
    let estimated_waves_in_workgroup = threads.div_ceil(warp.max(1));
    match occupancy {
        OccupancyAssumption::Undeclared => Err(ForgeError::Validation(
            "undeclared occupancy assumption is rejected; occupancy is an admission/report, not measured GPU timing".to_string(),
        )),
        OccupancyAssumption::None => Ok(OccupancyReport {
            assumed: false,
            workgroup_threads: threads,
            shared_bytes,
            estimated_waves_in_workgroup,
            note: OCCUPANCY_NOTE,
        }),
        OccupancyAssumption::Declared(DeclaredOccupancy {
            waves_per_cu,
            shared_bytes: declared_shared,
            subgroup_size,
        }) => {
            if waves_per_cu == 0 {
                return Err(ForgeError::Validation(
                    "declared occupancy waves_per_cu must be nonzero".to_string(),
                ));
            }
            if let Some(bytes) = declared_shared {
                if bytes != shared_bytes {
                    return Err(ForgeError::Validation(format!(
                        "declared occupancy shared_bytes {bytes} does not match kernel {shared_bytes}"
                    )));
                }
            }
            if let Some(size) = subgroup_size {
                if size != warp {
                    return Err(ForgeError::Validation(format!(
                        "declared occupancy subgroup size {size} does not match adapter warp {warp}"
                    )));
                }
            }
            Ok(OccupancyReport {
                assumed: true,
                workgroup_threads: threads,
                shared_bytes,
                estimated_waves_in_workgroup,
                note: OCCUPANCY_NOTE,
            })
        }
    }
}

fn admit_subgroup(request: &ScheduleGateRequest<'_>) -> Result<SubgroupAdmission, ForgeError> {
    let required = request.subgroup_required
        || request
            .kernel
            .required_intrinsics()
            .iter()
            .any(|intrinsic| intrinsic.class() == IntrinsicClass::Subgroup);
    let adapter_supports = request.constraints.supports_subgroups;
    if required && !adapter_supports {
        return Err(ForgeError::InvalidSchedule(
            "kernel requires subgroups unavailable on this adapter".to_string(),
        ));
    }
    let size = request
        .assumed_subgroup_size
        .unwrap_or(request.constraints.warp_size);
    if let Some(assumed) = request.assumed_subgroup_size {
        if assumed == 0 || assumed != request.constraints.warp_size {
            return Err(ForgeError::Validation(format!(
                "assumed subgroup size {assumed} is undeclared relative to adapter warp {}",
                request.constraints.warp_size
            )));
        }
    }
    Ok(SubgroupAdmission {
        required,
        adapter_supports,
        size,
    })
}

fn shared_bytes(kernel: &KernelSpec, workgroup: u32) -> Result<u32, ForgeError> {
    let mut total = 0u32;
    for spec in &kernel.shared_memory {
        let bytes = spec
            .length
            .resolve(workgroup)
            .checked_mul(spec.element.size_bytes())
            .ok_or_else(|| ForgeError::InvalidKernel("shared-memory length overflow".to_string()))?;
        total = total.checked_add(bytes).ok_or_else(|| {
            ForgeError::InvalidKernel("shared-memory byte count overflow".to_string())
        })?;
    }
    Ok(total)
}

fn workgroup_threads(schedule: &Schedule) -> u32 {
    schedule
        .workgroup
        .flat_threads()
        .max(schedule.effective_workgroup_size())
}

fn record_schedules(
    schedules: Vec<Schedule>,
    capacity: usize,
) -> Result<CandidateSearchRecord, ForgeError> {
    let capacity = require_capacity(capacity)?;
    if schedules.len() > capacity {
        return Err(overflow_error(schedules.len(), capacity));
    }
    let candidates = schedules
        .into_iter()
        .map(|schedule| RecordedCandidate {
            schedule,
            status: CandidateRecordStatus::Searched,
            timing: None,
            reason: None,
        })
        .collect();
    Ok(search_record(capacity, candidates))
}

fn search_record(capacity: usize, candidates: Vec<RecordedCandidate>) -> CandidateSearchRecord {
    CandidateSearchRecord {
        schema_version: PRODUCTION_QUALITY_SCHEMA_VERSION,
        capacity,
        recorded: candidates.len(),
        overflowed: false,
        candidates,
    }
}

fn require_capacity(capacity: usize) -> Result<usize, ForgeError> {
    if capacity == 0 {
        return Err(ForgeError::InvalidSchedule(
            "candidate record capacity must be non-zero".to_string(),
        ));
    }
    Ok(capacity)
}

fn overflow_error(count: usize, capacity: usize) -> ForgeError {
    ForgeError::InvalidSchedule(format!(
        "candidate search overflow: {count} candidates exceed bound {capacity}; fail-closed"
    ))
}

fn contains_schedule(candidates: &[RecordedCandidate], schedule: Schedule) -> bool {
    candidates
        .iter()
        .any(|candidate| candidate.schedule == schedule)
}

fn push_unique(
    out: &mut Vec<RecordedCandidate>,
    capacity: usize,
    candidate: RecordedCandidate,
) -> Result<(), ForgeError> {
    if contains_schedule(out, candidate.schedule) {
        return Ok(());
    }
    if out.len() >= capacity {
        return Err(overflow_error(out.len() + 1, capacity));
    }
    out.push(candidate);
    Ok(())
}
