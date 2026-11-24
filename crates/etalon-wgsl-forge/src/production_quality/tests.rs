use crate::autotune::{
    tune_with, CandidateEvaluation, CandidateFailure, CandidateResult, TuningConfig, TuningResult,
};
use crate::cache::{TimingSource, TimingSummary};
use crate::certify::{ComparisonReport, OracleTolerance};
use crate::ir::{ForgeError, Intrinsic, KernelSpec, Op, ScalarType, SharedLen, SharedMemorySpec};
use crate::ir_builtins::BuiltinKernel;
use crate::schedule::{AdapterConstraints, Schedule, ScheduleSpace, WorkgroupSize};

use super::*;

fn portable() -> AdapterConstraints {
    AdapterConstraints::portable()
}

fn affine() -> KernelSpec {
    BuiltinKernel::AffineF32.spec()
}

fn schedule() -> Schedule {
    Schedule::default()
}

fn request<'a>(
    kernel: &'a KernelSpec,
    constraints: &'a AdapterConstraints,
) -> ScheduleGateRequest<'a> {
    ScheduleGateRequest {
        kernel,
        schedule: schedule(),
        constraints,
        occupancy: OccupancyAssumption::None,
        subgroup_required: false,
        assumed_subgroup_size: None,
        element_count: None,
    }
}

fn oracle_pass() -> ComparisonReport {
    ComparisonReport {
        compared: 4,
        mismatch_count: 0,
        first_mismatch: None,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
    }
}

fn oracle_fail() -> ComparisonReport {
    ComparisonReport {
        compared: 4,
        mismatch_count: 1,
        first_mismatch: Some((0, 0.0, 1.0)),
        max_absolute_error: 1.0,
        max_relative_error: 1.0,
    }
}

fn labeled(source: TimingSource, ns: u64) -> LabeledTiming {
    label_timing(source, &[ns]).expect("labeled timing")
}

fn small_space() -> ScheduleSpace {
    ScheduleSpace {
        workgroup_sizes: vec![32, 64, 128],
        items_per_invocation: vec![1],
        vector_widths: vec![1],
    }
}

fn unit_config() -> TuningConfig {
    TuningConfig {
        initial_samples: 1,
        finalist_samples: 1,
        finalist_count: 3,
        max_candidates: 8,
    }
}

fn gated_request<'a>(
    kernel: &'a KernelSpec,
    constraints: &'a AdapterConstraints,
    space: &'a ScheduleSpace,
    gates: &'a FrozenQualityGates,
    baseline: Option<&'a AdmittedBaseline>,
    budget: SearchBudget,
) -> GatedSearchRequest<'a> {
    GatedSearchRequest {
        kernel,
        constraints,
        occupancy: OccupancyAssumption::None,
        space,
        config: unit_config(),
        budget,
        gates,
        baseline,
        subgroup_required: false,
        assumed_subgroup_size: None,
        element_count: None,
    }
}

fn measured(ns: u64) -> CostPresence {
    CostPresence::Measured(labeled(TimingSource::Synthetic, ns))
}

fn evaluate_with(
    source: TimingSource,
) -> impl FnMut(Schedule, usize) -> Result<CandidateEvaluation, ForgeError> {
    move |_schedule, samples| {
        Ok(CandidateEvaluation {
            oracle: oracle_pass(),
            timing_source: source,
            samples_ns: vec![100; samples.max(1)],
        })
    }
}

fn evaluate_uniform(
    _schedule: Schedule,
    samples: usize,
) -> Result<CandidateEvaluation, ForgeError> {
    Ok(CandidateEvaluation {
        oracle: oracle_pass(),
        timing_source: TimingSource::GpuTimestamp,
        samples_ns: vec![100; samples.max(1)],
    })
}

fn evaluate_invalid_fast(
    schedule: Schedule,
    samples: usize,
) -> Result<CandidateEvaluation, ForgeError> {
    if schedule == Schedule::from_1d(32, 1, 1) {
        Ok(CandidateEvaluation {
            oracle: oracle_fail(),
            timing_source: TimingSource::GpuTimestamp,
            samples_ns: vec![1; samples.max(1)],
        })
    } else {
        Ok(CandidateEvaluation {
            oracle: oracle_pass(),
            timing_source: TimingSource::GpuTimestamp,
            samples_ns: vec![1_000; samples.max(1)],
        })
    }
}

fn evaluate_p95_split(
    schedule: Schedule,
    _samples: usize,
) -> Result<CandidateEvaluation, ForgeError> {
    let samples_ns = if schedule == Schedule::from_1d(32, 1, 1) {
        vec![100, 100, 400]
    } else {
        vec![100, 100, 100]
    };
    Ok(CandidateEvaluation {
        oracle: oracle_pass(),
        timing_source: TimingSource::GpuTimestamp,
        samples_ns,
    })
}

#[test]
fn portable_schedule_is_admitted_with_occupancy_report() {
    let kernel = affine();
    let constraints = portable();
    let report = admit_schedule(&request(&kernel, &constraints)).expect("portable schedule");
    assert!(report.accepted);
    assert_eq!(report.schema_version, PRODUCTION_QUALITY_SCHEMA_VERSION);
    assert_eq!(report.workgroup.threads, 64);
    assert_eq!(report.workgroup.warp_size, 32);
    assert_eq!(report.shared_memory.declared_bytes, 0);
    assert_eq!(report.shared_memory.limit_bytes, PORTABLE_MAX_SHARED_BYTES);
    assert!(!report.occupancy.assumed);
    assert_eq!(report.occupancy.note, OCCUPANCY_NOTE);
    assert!(report.occupancy.note.contains("admission/report"));
    assert!(report
        .occupancy
        .note
        .contains("not a promise of measured GPU timing"));
    assert!(!report.subgroup.required);
    assert!(!report.subgroup.adapter_supports);
}

#[test]
fn undeclared_occupancy_is_rejected() {
    let kernel = affine();
    let constraints = portable();
    let mut req = request(&kernel, &constraints);
    req.occupancy = OccupancyAssumption::Undeclared;
    match admit_schedule(&req) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("undeclared occupancy"), "{message}");
            assert!(
                message.contains("not measured GPU timing") || message.contains("admission/report"),
                "{message}"
            );
        }
        other => panic!("expected undeclared occupancy rejection, got {other:?}"),
    }
}

#[test]
fn declared_occupancy_must_match_kernel_and_adapter() {
    let kernel = BuiltinKernel::TopK.spec();
    let constraints = portable();
    let shared = 4 * 64 * 4;
    let mut req = request(&kernel, &constraints);
    req.occupancy = OccupancyAssumption::Declared(DeclaredOccupancy {
        waves_per_cu: 2,
        shared_bytes: Some(shared),
        subgroup_size: Some(32),
    });
    let report = admit_schedule(&req).expect("matching declared occupancy");
    assert!(report.occupancy.assumed);
    assert_eq!(report.occupancy.shared_bytes, shared);
    assert_eq!(report.occupancy.note, OCCUPANCY_NOTE);

    req.occupancy = OccupancyAssumption::Declared(DeclaredOccupancy {
        waves_per_cu: 2,
        shared_bytes: Some(shared + 4),
        subgroup_size: Some(32),
    });
    match admit_schedule(&req) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("shared_bytes"), "{message}");
        }
        other => panic!("expected shared_bytes mismatch, got {other:?}"),
    }

    req.occupancy = OccupancyAssumption::Declared(DeclaredOccupancy {
        waves_per_cu: 0,
        shared_bytes: Some(shared),
        subgroup_size: Some(32),
    });
    match admit_schedule(&req) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("waves_per_cu"), "{message}");
        }
        other => panic!("expected zero waves rejection, got {other:?}"),
    }
}

#[test]
fn workgroup_shared_memory_and_subgroup_are_gated() {
    let kernel = affine();
    let constraints = portable();
    let mut req = request(&kernel, &constraints);
    req.schedule = Schedule::from_1d(512, 1, 1);
    match admit_schedule(&req) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("exceeds adapter limit"), "{message}");
        }
        other => panic!("expected workgroup rejection, got {other:?}"),
    }

    req.schedule = Schedule::from_1d(8, 1, 1);
    match admit_schedule(&req) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("not aligned to adapter warp"), "{message}");
        }
        other => panic!("expected warp alignment rejection, got {other:?}"),
    }

    req.schedule = schedule();
    req.schedule.workgroup = WorkgroupSize::new(64, 8, 1);
    match admit_schedule(&req) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("exceeds adapter limit"), "{message}");
        }
        other => panic!("expected flat-thread rejection, got {other:?}"),
    }

    req.schedule = schedule();
    let width = req.schedule.elements_per_workgroup() as usize;
    req.element_count = Some(constraints.max_workgroups_per_dimension as usize * width + 1);
    match admit_schedule(&req) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("dispatch"), "{message}");
            assert!(
                message.contains("max_workgroups_per_dimension"),
                "{message}"
            );
        }
        other => panic!("expected dispatch rejection, got {other:?}"),
    }
    req.element_count = None;

    let mut heavy = affine();
    heavy.shared_memory = vec![SharedMemorySpec {
        name: "scratch".to_string(),
        element: ScalarType::F32,
        length: SharedLen::Fixed(5_000),
    }];
    let shared_req = request(&heavy, &constraints);
    match admit_schedule(&shared_req) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("shared memory"), "{message}");
            assert!(
                message.contains("16384") || message.contains("portable"),
                "{message}"
            );
        }
        other => panic!("expected shared-memory rejection, got {other:?}"),
    }

    let mut subgroup_kernel = affine();
    subgroup_kernel
        .ops
        .push(Op::Intrinsic(Intrinsic::SubgroupBallot));
    match admit_schedule(&request(&subgroup_kernel, &constraints)) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("subgroups unavailable"), "{message}");
        }
        other => panic!("expected subgroup rejection, got {other:?}"),
    }

    let mut with_subgroups = portable();
    with_subgroups.supports_subgroups = true;
    let mut ok = request(&subgroup_kernel, &with_subgroups);
    ok.assumed_subgroup_size = Some(64);
    match admit_schedule(&ok) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("assumed subgroup size"), "{message}");
        }
        other => panic!("expected assumed subgroup rejection, got {other:?}"),
    }
}

#[test]
fn timing_fields_are_labeled_and_synthetic_is_not_gpu() {
    let gpu = label_timing(TimingSource::GpuTimestamp, &[10, 20, 30]).expect("gpu samples");
    assert_eq!(gpu.source, TimingSource::GpuTimestamp);
    assert!(gpu.as_gpu_timing().is_ok());

    let synthetic = TimingSummary::from_samples(TimingSource::Synthetic, &[1, 2, 3]).unwrap();
    match admit_timing(&synthetic, TimingSource::GpuTimestamp) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("synthetic"), "{message}");
            assert!(message.contains("GPU timing"), "{message}");
        }
        other => panic!("expected synthetic-as-GPU refusal, got {other:?}"),
    }

    let labeled_synthetic = admit_timing(&synthetic, TimingSource::Synthetic).expect("labeled");
    assert_eq!(labeled_synthetic.source, TimingSource::Synthetic);
    match labeled_synthetic.as_gpu_timing() {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("synthetic"), "{message}");
        }
        other => panic!("expected as_gpu refusal, got {other:?}"),
    }

    let clock = label_timing(TimingSource::CompletionClock, &[7, 8, 9]).expect("clock");
    match require_timing_label(clock.source, TimingSource::GpuTimestamp) {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("completion-clock") || message.contains("GPU timing"),
                "{message}"
            );
        }
        other => panic!("expected completion-clock-as-GPU refusal, got {other:?}"),
    }
}

#[test]
fn candidate_search_is_bounded_and_fail_closed_on_overflow() {
    assert_eq!(
        MAX_RECORDED_CANDIDATES,
        TuningConfig::default().max_candidates
    );

    let generated = record_generated_search(MAX_RECORDED_CANDIDATES).expect("generated search");
    assert!(!generated.overflowed);
    assert_eq!(generated.recorded, generated.candidates.len());
    assert!(generated.recorded <= MAX_RECORDED_CANDIDATES);
    assert!(generated
        .candidates
        .iter()
        .all(|candidate| candidate.status == CandidateRecordStatus::Searched));

    match record_generated_search(0) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("non-zero"), "{message}");
        }
        other => panic!("expected zero-capacity rejection, got {other:?}"),
    }

    let kernel = affine();
    let mut constraints = portable();
    constraints.warp_size = 1;
    constraints.max_workgroup_size_x = 1024;
    constraints.max_invocations_per_workgroup = 1024;
    let space = ScheduleSpace {
        workgroup_sizes: vec![32, 64, 128, 256, 512],
        items_per_invocation: vec![1, 2, 4, 8],
        vector_widths: vec![1, 2, 4],
    };
    let count = space.candidates(&kernel, &constraints).len();
    assert!(
        count > MAX_RECORDED_CANDIDATES,
        "need overflow fixture, got {count}"
    );
    match record_search_space(&kernel, &constraints, &space, MAX_RECORDED_CANDIDATES) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("overflow"), "{message}");
            assert!(message.contains("fail-closed"), "{message}");
        }
        other => panic!("expected fail-closed overflow, got {other:?}"),
    }
}

#[test]
fn autotune_result_is_recorded_with_labeled_timing() {
    let kernel = affine();
    let constraints = portable();
    let space = ScheduleSpace {
        workgroup_sizes: vec![32, 64],
        items_per_invocation: vec![1],
        vector_widths: vec![1],
    };
    let config = TuningConfig {
        initial_samples: 1,
        finalist_samples: 1,
        finalist_count: 2,
        max_candidates: 4,
    };
    let tuned = tune_with(
        &kernel,
        &constraints,
        &space,
        config,
        evaluate_with(TimingSource::GpuTimestamp),
    )
    .expect("tune");
    let report = gate_tuning(
        &kernel,
        &constraints,
        OccupancyAssumption::None,
        &tuned,
        MAX_RECORDED_CANDIDATES,
    )
    .expect("gate tuning");
    assert!(report.schedule.accepted);
    assert_eq!(report.winner_timing.source, TimingSource::GpuTimestamp);
    assert!(!report.search.overflowed);
    assert!(report
        .search
        .candidates
        .iter()
        .any(|candidate| candidate.schedule == tuned.winner.schedule
            && candidate.timing.as_ref().map(|timing| timing.source)
                == Some(TimingSource::GpuTimestamp)));

    let synthetic_tuned = tune_with(
        &kernel,
        &constraints,
        &space,
        config,
        evaluate_with(TimingSource::Synthetic),
    )
    .expect("synthetic tune");
    let recorded =
        record_tuning_result(&synthetic_tuned, MAX_RECORDED_CANDIDATES).expect("synthetic record");
    assert!(recorded.candidates.iter().all(|candidate| candidate
        .timing
        .as_ref()
        .is_none_or(|timing| timing.source != TimingSource::GpuTimestamp)));
    match admit_timing(&synthetic_tuned.winner.timing, TimingSource::GpuTimestamp) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("synthetic"), "{message}");
        }
        other => panic!("expected synthetic winner refusal, got {other:?}"),
    }

    match record_tuning_result(&tuned, 0) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("non-zero"), "{message}");
        }
        other => panic!("expected zero-capacity tuning record rejection, got {other:?}"),
    }
    match record_tuning_result(&tuned, 1) {
        Err(ForgeError::InvalidSchedule(message)) => {
            assert!(message.contains("overflow"), "{message}");
            assert!(message.contains("fail-closed"), "{message}");
        }
        other => panic!("expected tuning-record overflow, got {other:?}"),
    }
}

#[test]
fn invalid_fast_candidate_cannot_win() {
    let kernel = affine();
    let constraints = portable();
    let space = ScheduleSpace {
        workgroup_sizes: vec![32, 64],
        items_per_invocation: vec![1],
        vector_widths: vec![1],
    };
    let fast = Schedule::from_1d(32, 1, 1);
    let slow = Schedule::from_1d(64, 1, 1);
    let tuned = tune_with(
        &kernel,
        &constraints,
        &space,
        unit_config(),
        evaluate_invalid_fast,
    )
    .expect("tune_with rejects oracle failures");
    assert_eq!(tuned.winner.schedule, slow);
    assert_ne!(tuned.winner.schedule, fast);
    assert!(tuned
        .failures
        .iter()
        .any(|failure| failure.schedule == fast));

    let report = gate_tuning(
        &kernel,
        &constraints,
        OccupancyAssumption::None,
        &tuned,
        MAX_RECORDED_CANDIDATES,
    )
    .expect("slow winner is admitted");
    assert_eq!(report.schedule.schedule, slow);
    assert!(report
        .search
        .candidates
        .iter()
        .any(|candidate| candidate.schedule == fast
            && candidate.status == CandidateRecordStatus::Rejected));

    let gates = FrozenQualityGates::production();
    let budget = SearchBudget::from_config(unit_config());
    let baseline = AdmittedBaseline::new(
        &gates,
        slow,
        oracle_pass(),
        labeled(TimingSource::GpuTimestamp, 1_000),
    )
    .expect("slow baseline");
    let request = gated_request(
        &kernel,
        &constraints,
        &space,
        &gates,
        Some(&baseline),
        budget,
    );
    let gated = gated_search(&request, evaluate_invalid_fast).expect("gated search");
    assert_eq!(gated.quality.schedule.schedule, slow);
    assert!(!gated.learned_search);
    assert!(gated
        .quality
        .search
        .candidates
        .iter()
        .any(|candidate| candidate.schedule == fast
            && candidate.status == CandidateRecordStatus::Rejected
            && candidate
                .reason
                .as_ref()
                .is_some_and(|reason| reason.contains("cannot win")
                    || reason.contains("oracle")
                    || reason.contains("quality"))));

    let tampered = TuningResult {
        evaluated_candidates: 2,
        rejected_candidates: 1,
        failures: vec![CandidateFailure {
            schedule: fast,
            reason: "oracle rejected 1 value(s)".to_string(),
        }],
        winner: CandidateResult {
            schedule: fast,
            oracle: oracle_fail(),
            timing: TimingSummary::from_samples(TimingSource::GpuTimestamp, &[1]).unwrap(),
        },
        finalists: vec![CandidateResult {
            schedule: slow,
            oracle: oracle_pass(),
            timing: TimingSummary::from_samples(TimingSource::GpuTimestamp, &[1_000]).unwrap(),
        }],
    };
    match gate_tuning(
        &kernel,
        &constraints,
        OccupancyAssumption::None,
        &tampered,
        MAX_RECORDED_CANDIDATES,
    ) {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("cannot win")
                    || message.contains("tamper")
                    || message.contains("oracle"),
                "{message}"
            );
        }
        other => panic!("expected invalid fast winner refusal, got {other:?}"),
    }
}

#[test]
fn immutable_thresholds_reject_tampering() {
    let gates = FrozenQualityGates::production();
    assert_eq!(
        FROZEN_MAX_ORACLE_MISMATCHES,
        QualityThresholds::production().max_oracle_mismatches
    );
    assert_eq!(
        FROZEN_MAX_ABSOLUTE_ERROR,
        OracleTolerance::default().absolute
    );
    assert_eq!(
        FROZEN_MAX_RELATIVE_ERROR,
        OracleTolerance::default().relative
    );
    assert_eq!(FROZEN_MAX_SHARED_BYTES, PORTABLE_MAX_SHARED_BYTES);
    assert_eq!(
        FROZEN_MAX_WORKGROUP_THREADS,
        AdapterConstraints::portable().max_invocations_per_workgroup
    );
    gates.validate().expect("production gates");

    let mut loosened = QualityThresholds::production();
    loosened.max_oracle_mismatches = 8;
    match FrozenQualityGates::freeze(loosened) {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("tamper") || message.contains("immutable"),
                "{message}"
            );
        }
        other => panic!("expected freeze refusal, got {other:?}"),
    }
    match gates.attempt_retune(loosened) {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("tamper") || message.contains("immutable"),
                "{message}"
            );
        }
        other => panic!("expected retune refusal, got {other:?}"),
    }
    loosened.max_oracle_mismatches = 0;
    loosened.max_absolute_error = 1.0;
    assert!(FrozenQualityGates::freeze(loosened).is_err());
    loosened.max_absolute_error = FROZEN_MAX_ABSOLUTE_ERROR;
    loosened.max_shared_bytes = PORTABLE_MAX_SHARED_BYTES + 1;
    assert!(FrozenQualityGates::freeze(loosened).is_err());

    let mut json = serde_json::to_value(&gates).expect("serialize gates");
    json["thresholds"]["max_oracle_mismatches"] = serde_json::json!(8);
    let tampered: FrozenQualityGates =
        serde_json::from_value(json).expect("tampered JSON still deserializes");
    match tampered.validate() {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("tamper") || message.contains("immutable"),
                "{message}"
            );
        }
        other => panic!("expected digest/threshold tamper refusal, got {other:?}"),
    }

    match admit_oracle(&gates, &oracle_fail()) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("cannot win"), "{message}");
        }
        other => panic!("expected oracle gate refusal, got {other:?}"),
    }
}

#[test]
fn budget_exhaustion_preserves_admitted_baseline() {
    let kernel = affine();
    let constraints = portable();
    let space = small_space();
    let gates = FrozenQualityGates::production();
    let baseline_schedule = Schedule::from_1d(64, 1, 1);
    let fast_schedule = Schedule::from_1d(32, 1, 1);
    let baseline = AdmittedBaseline::new(
        &gates,
        baseline_schedule,
        oracle_pass(),
        labeled(TimingSource::GpuTimestamp, 5_000),
    )
    .expect("baseline");
    let budget = SearchBudget {
        max_recorded: MAX_RECORDED_CANDIDATES,
        max_evaluations: 1,
        max_samples: usize::MAX / 4,
    };
    let evaluate = |schedule: Schedule, samples: usize| {
        let ns = if schedule == fast_schedule { 10 } else { 5_000 };
        Ok(CandidateEvaluation {
            oracle: oracle_pass(),
            timing_source: TimingSource::GpuTimestamp,
            samples_ns: vec![ns; samples.max(1)],
        })
    };
    let request = gated_request(
        &kernel,
        &constraints,
        &space,
        &gates,
        Some(&baseline),
        budget,
    );
    let report = gated_search(&request, evaluate).expect("exhausted search keeps baseline");
    assert!(report.budget_exhausted);
    assert!(report.preserved_baseline);
    assert_eq!(report.winner_source, WinnerSource::PreservedBaseline);
    assert_eq!(report.quality.schedule.schedule, baseline_schedule);
    assert_ne!(report.quality.schedule.schedule, fast_schedule);
    assert!(report.quality.search.skipped >= 1);
    assert!(report.quality.search.budget_exhausted);
    assert_eq!(
        report.quality.search.recorded,
        space.candidates(&kernel, &constraints).len()
    );
    assert!(report
        .quality
        .search
        .candidates
        .iter()
        .any(|candidate| candidate.schedule == fast_schedule
            && candidate.status == CandidateRecordStatus::Admitted));
    assert!(report
        .quality
        .search
        .candidates
        .iter()
        .any(|candidate| candidate.schedule == baseline_schedule
            && candidate.status == CandidateRecordStatus::Admitted
            && candidate
                .reason
                .as_ref()
                .is_some_and(|reason| reason.contains("baseline"))));
}

#[test]
fn tie_rules_are_reproducible() {
    let kernel = affine();
    let constraints = portable();
    let space = ScheduleSpace {
        workgroup_sizes: vec![32, 64],
        items_per_invocation: vec![1],
        vector_widths: vec![1],
    };
    let gates = FrozenQualityGates::production();
    let budget = SearchBudget::from_config(unit_config());
    let request = gated_request(&kernel, &constraints, &space, &gates, None, budget);
    let first = gated_search(&request, evaluate_uniform).expect("tie run 1");
    let second = gated_search(&request, evaluate_uniform).expect("tie run 2");
    assert_eq!(
        first.quality.schedule.schedule,
        second.quality.schedule.schedule
    );
    assert_eq!(first.quality.schedule.schedule, Schedule::from_1d(32, 1, 1));
    assert_eq!(
        ranking_key(
            &first.quality.winner_timing,
            first.quality.schedule.schedule
        ),
        ranking_key(
            &second.quality.winner_timing,
            second.quality.schedule.schedule
        )
    );

    let low_p95 = Schedule::from_1d(64, 1, 1);
    let p95_winner = gated_search(&request, evaluate_p95_split)
        .expect("p95 tie-break")
        .quality
        .schedule
        .schedule;
    assert_eq!(p95_winner, low_p95);

    let tuned_a = tune_with(
        &kernel,
        &constraints,
        &space,
        unit_config(),
        evaluate_uniform,
    )
    .expect("tune a");
    let tuned_b = tune_with(
        &kernel,
        &constraints,
        &space,
        unit_config(),
        evaluate_uniform,
    )
    .expect("tune b");
    assert_eq!(tuned_a.winner.schedule, tuned_b.winner.schedule);
    assert_eq!(tuned_a.winner.schedule, first.quality.schedule.schedule);
}

#[test]
fn occupancy_is_admission_never_labeled_timing() {
    let occupancy = DeclaredOccupancy {
        waves_per_cu: 4,
        shared_bytes: Some(0),
        subgroup_size: Some(32),
    };
    match labeled_timing_from_occupancy(occupancy) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("DeclaredOccupancy"), "{message}");
            assert!(message.contains("LabeledTiming"), "{message}");
            assert!(message.contains("GpuTimestamp"), "{message}");
            assert!(!message.contains("signature"), "{message}");
        }
        other => panic!("expected occupancy-as-timing refusal, got {other:?}"),
    }

    let kernel = affine();
    let constraints = portable();
    let mut req = request(&kernel, &constraints);
    req.occupancy = OccupancyAssumption::Declared(occupancy);
    let report = admit_schedule(&req).expect("declared occupancy is admission");
    assert!(report.occupancy.assumed);
    assert_eq!(report.occupancy.note, OCCUPANCY_NOTE);
    match occupancy_report_as_gpu_timestamp(report.occupancy) {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("never stored as LabeledTiming"),
                "{message}"
            );
        }
        other => panic!("expected occupancy report refusal, got {other:?}"),
    }
    let timing = labeled(TimingSource::GpuTimestamp, 40);
    assert_eq!(timing.source, TimingSource::GpuTimestamp);
    assert_ne!(timing.minimum_ns, u64::from(occupancy.waves_per_cu));
}

#[test]
fn crc_is_not_a_signature_and_synthetic_is_not_gpu() {
    match crc_is_not_a_signature("crc32c") {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("checksum"), "{message}");
            assert!(message.contains("not a signature"), "{message}");
        }
        other => panic!("expected CRC refusal, got {other:?}"),
    }
    match parse_content_digest("CRC-32", "deadbeef") {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("checksum") || message.contains("not a signature"),
                "{message}"
            );
        }
        other => panic!("expected CRC parse refusal, got {other:?}"),
    }
    match parse_content_digest("crc", "00") {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("not a signature"), "{message}");
        }
        other => panic!("expected CRC refusal, got {other:?}"),
    }
    let identity = Digest::blake3(b"production-quality-gates");
    let parsed = parse_content_digest("blake3", &identity.hex).expect("blake3 identity");
    assert_eq!(parsed, identity);

    let synthetic = labeled(TimingSource::Synthetic, 9);
    match synthetic.as_gpu_timing() {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("synthetic"), "{message}");
            assert!(message.contains("GPU timing"), "{message}");
        }
        other => panic!("expected synthetic-as-GPU refusal, got {other:?}"),
    }
}

#[test]
fn gated_search_publishes_all_attempts_and_failures() {
    let kernel = affine();
    let constraints = portable();
    let space = small_space();
    let gates = FrozenQualityGates::production();
    let expected = space.candidates(&kernel, &constraints);
    assert_eq!(expected.len(), 3);
    let fast = Schedule::from_1d(32, 1, 1);
    let evaluate = |schedule: Schedule, samples: usize| {
        if schedule == fast {
            Ok(CandidateEvaluation {
                oracle: oracle_fail(),
                timing_source: TimingSource::GpuTimestamp,
                samples_ns: vec![1; samples.max(1)],
            })
        } else {
            Ok(CandidateEvaluation {
                oracle: oracle_pass(),
                timing_source: TimingSource::GpuTimestamp,
                samples_ns: vec![200; samples.max(1)],
            })
        }
    };
    let request = gated_request(
        &kernel,
        &constraints,
        &space,
        &gates,
        None,
        SearchBudget::from_config(unit_config()),
    );
    let report = gated_search(&request, evaluate).expect("publish all");
    assert!(!report.budget_exhausted);
    assert_eq!(report.quality.search.recorded, expected.len());
    assert_eq!(report.quality.search.candidates.len(), expected.len());
    assert_eq!(report.quality.search.rejected, 1);
    assert_eq!(report.quality.search.admitted, 2);
    assert_eq!(report.quality.search.skipped, 0);
    for schedule in &expected {
        assert!(
            report
                .quality
                .search
                .candidates
                .iter()
                .any(|candidate| candidate.schedule == *schedule),
            "missing attempt {schedule:?}"
        );
    }
    let rejected = report
        .quality
        .search
        .candidates
        .iter()
        .find(|candidate| candidate.schedule == fast)
        .expect("fast failure published");
    assert_eq!(rejected.status, CandidateRecordStatus::Rejected);
    assert!(rejected.reason.is_some());
    assert_ne!(report.quality.schedule.schedule, fast);
    assert!(!report.pareto.is_empty());
    assert!(!report.learned_search);
    assert!(report.learned_search_note.contains("excluded"));
}

#[test]
fn break_even_does_not_treat_missing_cost_as_zero() {
    let baseline = PhaseCosts {
        build: measured(10),
        compile: measured(10),
        upload: measured(10),
        steady: measured(100),
    };
    let cheaper_steady = PhaseCosts {
        build: measured(100),
        compile: measured(10),
        upload: measured(10),
        steady: measured(10),
    };
    let report = break_even(&baseline, &cheaper_steady).expect("labeled break-even");
    match report.status {
        BreakEvenStatus::CandidateWins { from_n, until_n } => {
            assert_eq!(from_n, 1);
            assert_eq!(until_n, None);
        }
        other => panic!("expected candidate win from n=1, got {other:?}"),
    }
    assert_eq!(report.method, Some(TimingSource::Synthetic));
    assert!(report.note.contains("P22"));

    let missing = PhaseCosts {
        build: measured(10),
        compile: measured(10),
        upload: CostPresence::unsupported("upload cost not supplied"),
        steady: measured(100),
    };
    let inconclusive = break_even(&baseline, &missing).expect("unsupported is not an error zero");
    assert_eq!(inconclusive.status, BreakEvenStatus::Inconclusive);
    assert!(inconclusive.note.contains("unsupported"));
    assert!(inconclusive.note.contains("not zero"));
    match missing.upload.median_ns("upload") {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("unsupported"), "{message}");
            assert!(message.contains("not zero"), "{message}");
        }
        other => panic!("expected unsupported upload, got {other:?}"),
    }

    let gpu = PhaseCosts {
        build: CostPresence::Measured(labeled(TimingSource::GpuTimestamp, 10)),
        compile: CostPresence::Measured(labeled(TimingSource::GpuTimestamp, 10)),
        upload: CostPresence::Measured(labeled(TimingSource::GpuTimestamp, 10)),
        steady: CostPresence::Measured(labeled(TimingSource::GpuTimestamp, 10)),
    };
    let mixed = break_even(&baseline, &gpu).expect("mixed methods");
    assert_eq!(mixed.status, BreakEvenStatus::Inconclusive);
    assert!(mixed.note.contains("mixed"));
}
