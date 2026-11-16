use crate::autotune::{tune_with, CandidateEvaluation, TuningConfig};
use crate::cache::{TimingSource, TimingSummary};
use crate::certify::ComparisonReport;
use crate::ir::{
    ForgeError, Intrinsic, KernelSpec, Op, ScalarType, SharedLen, SharedMemorySpec,
};
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
    assert!(report.occupancy.note.contains("not a promise of measured GPU timing"));
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
            assert!(message.contains("max_workgroups_per_dimension"), "{message}");
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
            assert!(message.contains("16384") || message.contains("portable"), "{message}");
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
    assert_eq!(MAX_RECORDED_CANDIDATES, TuningConfig::default().max_candidates);

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
    assert!(count > MAX_RECORDED_CANDIDATES, "need overflow fixture, got {count}");
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
    match admit_timing(
        &synthetic_tuned.winner.timing,
        TimingSource::GpuTimestamp,
    ) {
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
