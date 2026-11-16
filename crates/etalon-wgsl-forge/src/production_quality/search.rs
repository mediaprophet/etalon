//! Equal-budget gated search over existing `tune_with` evaluation.

use crate::autotune::{CandidateEvaluation, TuningConfig};
use crate::certify::ComparisonReport;
use crate::ir::{ForgeError, KernelSpec};
use crate::production_lowering::OccupancyAssumption;
use crate::schedule::{AdapterConstraints, Schedule, ScheduleSpace};

use super::gates::{
    admit_oracle, FrozenQualityGates, LEARNED_SEARCH_EXCLUDED_NOTE,
};
use super::schedule_gate::{
    admit_schedule, label_timing, CandidateRecordStatus, CandidateSearchRecord, LabeledTiming,
    RecordedCandidate, ScheduleGateReport, ScheduleGateRequest, TuningQualityReport,
    MAX_RECORDED_CANDIDATES,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SearchBudget {
    pub max_recorded: usize,
    pub max_evaluations: usize,
    pub max_samples: usize,
}

impl SearchBudget {
    pub fn from_config(config: TuningConfig) -> Self {
        Self {
            max_recorded: MAX_RECORDED_CANDIDATES,
            max_evaluations: config.max_candidates,
            max_samples: usize::MAX / 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdmittedBaseline {
    pub schedule: Schedule,
    pub oracle: ComparisonReport,
    pub timing: LabeledTiming,
}

impl AdmittedBaseline {
    pub fn new(
        gates: &FrozenQualityGates,
        schedule: Schedule,
        oracle: ComparisonReport,
        timing: LabeledTiming,
    ) -> Result<Self, ForgeError> {
        admit_oracle(gates, &oracle)?;
        Ok(Self {
            schedule,
            oracle,
            timing,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WinnerSource {
    Search,
    PreservedBaseline,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParetoPoint {
    pub schedule: Schedule,
    pub median_ns: u64,
    pub max_absolute_error: f32,
    pub shared_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GatedTuneReport {
    pub quality: TuningQualityReport,
    pub winner_source: WinnerSource,
    pub gates_digest: String,
    pub preserved_baseline: bool,
    pub learned_search: bool,
    pub learned_search_note: String,
    pub pareto: Vec<ParetoPoint>,
    pub budget_exhausted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatedSearchRequest<'a> {
    pub kernel: &'a KernelSpec,
    pub constraints: &'a AdapterConstraints,
    pub occupancy: OccupancyAssumption,
    pub space: &'a ScheduleSpace,
    pub config: TuningConfig,
    pub budget: SearchBudget,
    pub gates: &'a FrozenQualityGates,
    pub baseline: Option<&'a AdmittedBaseline>,
    pub subgroup_required: bool,
    pub assumed_subgroup_size: Option<u32>,
    pub element_count: Option<usize>,
}

pub fn ranking_key(timing: &LabeledTiming, schedule: Schedule) -> (u64, u64, u32, u32, u32) {
    let (workgroup, items, vector) = schedule.sort_key();
    (timing.median_ns, timing.p95_ns, workgroup, items, vector)
}

pub fn gated_search<F>(
    request: &GatedSearchRequest<'_>,
    mut evaluate: F,
) -> Result<GatedTuneReport, ForgeError>
where
    F: FnMut(Schedule, usize) -> Result<CandidateEvaluation, ForgeError>,
{
    request.gates.validate()?;
    if request.config.initial_samples == 0 || request.config.finalist_samples == 0 {
        return Err(ForgeError::InvalidSchedule(
            "tuning sample and candidate budgets must be non-zero".to_string(),
        ));
    }
    if let Some(baseline) = request.baseline {
        admit_oracle(request.gates, &baseline.oracle)?;
        admit_candidate_schedule(request, baseline.schedule)?;
    }

    let schedules = request
        .space
        .candidates(request.kernel, request.constraints);
    if schedules.is_empty() {
        return Err(ForgeError::InvalidSchedule(
            "schedule space contains no adapter-compatible candidates".to_string(),
        ));
    }
    let baseline_extra = match request.baseline {
        Some(baseline) if !schedules.iter().any(|schedule| *schedule == baseline.schedule) => 1,
        _ => 0,
    };
    let recorded_need = schedules.len() + baseline_extra;
    if request.budget.max_recorded == 0 {
        return Err(ForgeError::InvalidSchedule(
            "candidate record capacity must be non-zero".to_string(),
        ));
    }
    if recorded_need > request.budget.max_recorded {
        return Err(ForgeError::InvalidSchedule(format!(
            "candidate search overflow: {recorded_need} candidates exceed bound {}; fail-closed",
            request.budget.max_recorded
        )));
    }

    let samples = request.config.initial_samples;
    let mut recorded = Vec::new();
    let mut admitted = Vec::new();
    let mut evaluations = 0usize;
    let mut samples_used = 0usize;
    let mut exhausted = false;

    for schedule in schedules {
        if exhausted
            || evaluations >= request.budget.max_evaluations
            || samples_used.saturating_add(samples) > request.budget.max_samples
        {
            exhausted = true;
            recorded.push(RecordedCandidate {
                schedule,
                status: CandidateRecordStatus::BudgetSkipped,
                timing: None,
                reason: Some("search budget exhausted".to_string()),
            });
            continue;
        }
        evaluations += 1;
        samples_used = samples_used.saturating_add(samples);
        match evaluate_candidate(request, schedule, samples, &mut evaluate) {
            Ok(pass) => {
                admitted.push(pass.clone());
                recorded.push(RecordedCandidate {
                    schedule,
                    status: CandidateRecordStatus::Admitted,
                    timing: Some(pass.timing.clone()),
                    reason: None,
                });
            }
            Err(reason) => {
                recorded.push(RecordedCandidate {
                    schedule,
                    status: CandidateRecordStatus::Rejected,
                    timing: None,
                    reason: Some(reason),
                });
            }
        }
    }

    if let Some(baseline) = request.baseline {
        ensure_baseline_recorded(
            &mut recorded,
            request.budget.max_recorded,
            baseline,
            exhausted,
        )?;
        if exhausted {
            if recorded.iter().any(|candidate| {
                candidate.schedule == baseline.schedule
                    && candidate.status == CandidateRecordStatus::Rejected
            }) {
                return Err(ForgeError::Validation(
                    "budget exhaustion cannot preserve a baseline that fails quality gates"
                        .to_string(),
                ));
            }
        } else if !admitted
            .iter()
            .any(|candidate| candidate.schedule == baseline.schedule)
        {
            admitted.push(AdmittedPoint {
                schedule: baseline.schedule,
                oracle: baseline.oracle.clone(),
                timing: baseline.timing.clone(),
                report: admit_candidate_schedule(request, baseline.schedule)?,
            });
        }
    }

    let (winner, winner_source, preserved_baseline) = if exhausted {
        let baseline = request.baseline.ok_or_else(|| {
            ForgeError::Validation(
                "budget exhausted with no admitted baseline to preserve".to_string(),
            )
        })?;
        let winner = AdmittedPoint {
            schedule: baseline.schedule,
            oracle: baseline.oracle.clone(),
            timing: baseline.timing.clone(),
            report: admit_candidate_schedule(request, baseline.schedule)?,
        };
        if !admitted
            .iter()
            .any(|candidate| candidate.schedule == winner.schedule)
        {
            admitted.push(winner.clone());
        }
        (winner, WinnerSource::PreservedBaseline, true)
    } else {
        if admitted.is_empty() {
            return Err(ForgeError::OracleMismatch(
                "no schedule passed frozen quality gates".to_string(),
            ));
        }
        admitted.sort_by_key(|point| ranking_key(&point.timing, point.schedule));
        let winner = admitted[0].clone();
        (winner, WinnerSource::Search, false)
    };

    if recorded
        .iter()
        .any(|candidate| {
            candidate.schedule == winner.schedule
                && candidate.status == CandidateRecordStatus::Rejected
        })
    {
        return Err(ForgeError::Validation(
            "invalid fast candidate cannot win".to_string(),
        ));
    }
    admit_oracle(request.gates, &winner.oracle)?;

    let mut search = CandidateSearchRecord::from_candidates(request.budget.max_recorded, recorded)?;
    search.budget_exhausted = exhausted;
    let quality = TuningQualityReport {
        schema_version: super::PRODUCTION_QUALITY_SCHEMA_VERSION,
        schedule: winner.report,
        winner_timing: winner.timing.clone(),
        search,
    };
    Ok(GatedTuneReport {
        quality,
        winner_source,
        gates_digest: request.gates.digest_hex().to_string(),
        preserved_baseline,
        learned_search: false,
        learned_search_note: LEARNED_SEARCH_EXCLUDED_NOTE.to_string(),
        pareto: pareto_front(&admitted),
        budget_exhausted: exhausted,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct AdmittedPoint {
    schedule: Schedule,
    oracle: ComparisonReport,
    timing: LabeledTiming,
    report: ScheduleGateReport,
}

fn admit_candidate_schedule(
    request: &GatedSearchRequest<'_>,
    schedule: Schedule,
) -> Result<ScheduleGateReport, ForgeError> {
    admit_schedule(&ScheduleGateRequest {
        kernel: request.kernel,
        schedule,
        constraints: request.constraints,
        occupancy: request.occupancy,
        subgroup_required: request.subgroup_required,
        assumed_subgroup_size: request.assumed_subgroup_size,
        element_count: request.element_count,
    })
}

fn evaluate_candidate<F>(
    request: &GatedSearchRequest<'_>,
    schedule: Schedule,
    samples: usize,
    evaluate: &mut F,
) -> Result<AdmittedPoint, String>
where
    F: FnMut(Schedule, usize) -> Result<CandidateEvaluation, ForgeError>,
{
    let report = admit_candidate_schedule(request, schedule).map_err(|error| error.to_string())?;
    if report.workgroup.threads > request.gates.thresholds().max_workgroup_threads {
        return Err(format!(
            "workgroup {} exceeds frozen quality threshold {}",
            report.workgroup.threads,
            request.gates.thresholds().max_workgroup_threads
        ));
    }
    if report.shared_memory.declared_bytes > request.gates.thresholds().max_shared_bytes {
        return Err(format!(
            "shared memory {} exceeds frozen quality threshold {}",
            report.shared_memory.declared_bytes,
            request.gates.thresholds().max_shared_bytes
        ));
    }
    let evaluation = evaluate(schedule, samples).map_err(|error| error.to_string())?;
    admit_oracle(request.gates, &evaluation.oracle).map_err(|error| error.to_string())?;
    let timing = label_timing(evaluation.timing_source, &evaluation.samples_ns)
        .map_err(|error| error.to_string())?;
    Ok(AdmittedPoint {
        schedule,
        oracle: evaluation.oracle,
        timing,
        report,
    })
}

fn ensure_baseline_recorded(
    recorded: &mut Vec<RecordedCandidate>,
    capacity: usize,
    baseline: &AdmittedBaseline,
    exhausted: bool,
) -> Result<(), ForgeError> {
    if let Some(existing) = recorded
        .iter_mut()
        .find(|candidate| candidate.schedule == baseline.schedule)
    {
        if existing.status == CandidateRecordStatus::Rejected {
            return Ok(());
        }
        if exhausted && existing.status != CandidateRecordStatus::Admitted {
            existing.status = CandidateRecordStatus::Admitted;
            existing.timing = Some(baseline.timing.clone());
            existing.reason = Some("budget exhaustion preserves admitted baseline".to_string());
        }
        return Ok(());
    }
    if recorded.len() >= capacity {
        return Err(ForgeError::InvalidSchedule(format!(
            "candidate search overflow: {} candidates exceed bound {capacity}; fail-closed",
            recorded.len() + 1
        )));
    }
    let reason = if exhausted {
        Some("budget exhaustion preserves admitted baseline".to_string())
    } else {
        Some("admitted baseline".to_string())
    };
    recorded.push(RecordedCandidate {
        schedule: baseline.schedule,
        status: CandidateRecordStatus::Admitted,
        timing: Some(baseline.timing.clone()),
        reason,
    });
    Ok(())
}

fn pareto_front(admitted: &[AdmittedPoint]) -> Vec<ParetoPoint> {
    let points: Vec<ParetoPoint> = admitted
        .iter()
        .map(|point| ParetoPoint {
            schedule: point.schedule,
            median_ns: point.timing.median_ns,
            max_absolute_error: point.oracle.max_absolute_error,
            shared_bytes: point.report.shared_memory.declared_bytes,
        })
        .collect();
    let mut front: Vec<ParetoPoint> = points
        .iter()
        .filter(|point| {
            !points.iter().any(|other| {
                other.median_ns <= point.median_ns
                    && other.max_absolute_error <= point.max_absolute_error
                    && other.shared_bytes <= point.shared_bytes
                    && (other.median_ns < point.median_ns
                        || other.max_absolute_error < point.max_absolute_error
                        || other.shared_bytes < point.shared_bytes)
            })
        })
        .cloned()
        .collect();
    front.sort_by_key(|point| {
        (
            point.median_ns,
            point.shared_bytes,
            point.schedule.sort_key(),
        )
    });
    front
}
