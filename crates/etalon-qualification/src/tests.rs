use super::*;

const REV_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const REV_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn fixture() -> (RunReceipt<'static>, ExpectedRun<'static>) {
    let build = BuildIdentity {
        qualia: SourceIdentity {
            revision: REV_A,
            lockfile_sha256: HASH_A,
            patch_sha256: None,
        },
        etalon: SourceIdentity {
            revision: REV_A,
            lockfile_sha256: HASH_A,
            patch_sha256: None,
        },
        toolchain: "synthetic-toolchain",
        features: &["synthetic-feature"],
    };
    let workload = WorkloadIdentity {
        fixture_sha256: HASH_A,
        model_sha256: Some(HASH_A),
        shader_sha256: Some(HASH_A),
        tokenizer_sha256: None,
        layout: "synthetic-q4k-soa-160",
        precision: "f32",
        mode: ExecutionMode::Developer,
        policy_sha256: None,
        cache_generation: 1,
        seed: 17,
    };
    let device = DeviceIdentity {
        backend: Backend::Cuda,
        device: "synthetic-device",
        driver: "synthetic-driver",
        compiler: "synthetic-compiler",
    };
    let tolerance = Tolerance {
        absolute: 0.0,
        relative: 0.0,
    };
    let receipt = RunReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        packet: "EM-01A-fixture",
        build,
        workload,
        requested_backend: Backend::Cuda,
        actual_device: device,
        execution_path: ExecutionPath::CudaGraphReplay,
        origin: EvidenceOrigin::SyntheticFixture,
        outcome: RunOutcome::Passed,
        output_kind: OutputKind::Raw,
        output_sha256: HASH_A,
        command: "synthetic test fixture; no hardware execution",
        tests: TestCounts {
            passed: 3,
            failed: 0,
            skipped: 0,
            ignored: 0,
        },
        tolerance,
        metrics: RunMetrics {
            numerical: Measurement::Measured(
                compare_f32(&[1.0, 2.0], &[1.0, 2.0], tolerance).unwrap(),
            ),
            latency_p50_ns: Measurement::Measured(10.0),
            latency_p95_ns: Measurement::Measured(12.0),
            allocations: Measurement::Measured(0),
            peak_bytes: Measurement::Measured(128),
            uploaded_bytes: Measurement::Measured(16),
            downloaded_bytes: Measurement::Measured(8),
            graph_launches: Measurement::Measured(1),
        },
        limitations: "Synthetic declarations only",
    };
    let expected = ExpectedRun {
        packet: receipt.packet,
        build,
        workload,
        device,
        execution_path: receipt.execution_path,
        expected_tests: 3,
        expected_values: 2,
        output_kind: OutputKind::Raw,
        exact_output_sha256: Some(HASH_A),
        tolerance,
        require_live: false,
        require_timing: true,
        require_resource_metrics: true,
    };
    (receipt, expected)
}

#[test]
fn synthetic_pair_consistency_is_not_live_evidence() {
    let (receipt, mut expected) = fixture();
    assert_eq!(
        validate_pair_declarations(&receipt, &expected, &receipt, &expected),
        Ok(())
    );
    expected.require_live = true;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::SyntheticEvidence)
    );
}

#[test]
fn candidate_has_independently_frozen_revision_and_cache() {
    let (baseline, old_expected) = fixture();
    let mut candidate = baseline;
    candidate.build.etalon.revision = REV_B;
    candidate.workload.cache_generation = 2;
    let mut new_expected = old_expected;
    new_expected.build.etalon.revision = REV_B;
    new_expected.workload.cache_generation = 2;
    assert_eq!(
        validate_pair_declarations(&baseline, &old_expected, &candidate, &new_expected),
        Ok(())
    );
    assert_eq!(
        validate_receipt_declarations(&baseline, &new_expected),
        Err(ReceiptError::StaleBuild)
    );
    candidate.workload.cache_generation = 1;
    assert_eq!(
        validate_receipt_declarations(&candidate, &new_expected),
        Err(ReceiptError::WrongWorkload)
    );
}

#[test]
fn detects_wrong_numerical_output_and_hash() {
    let (mut receipt, expected) = fixture();
    receipt.metrics.numerical =
        Measurement::Measured(compare_f32(&[1.0, 2.0], &[1.0, 3.0], expected.tolerance).unwrap());
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::NumericalEvidence)
    );
    let (mut receipt, expected) = fixture();
    receipt.output_sha256 = HASH_B;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::WrongOutput)
    );
}

#[test]
fn partial_numerical_comparison_is_not_complete_fixture_evidence() {
    let (mut receipt, expected) = fixture();
    receipt.metrics.numerical =
        Measurement::Measured(compare_f32(&[1.0], &[1.0], expected.tolerance).unwrap());
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::NumericalEvidence)
    );
}

#[test]
fn detects_wrong_layout_fixture_and_repaired_output() {
    let (mut receipt, expected) = fixture();
    receipt.workload.layout = "synthetic-q4k-aos-144";
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::WrongWorkload)
    );
    receipt.workload = expected.workload;
    receipt.workload.fixture_sha256 = HASH_B;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::WrongWorkload)
    );
    receipt.workload = expected.workload;
    receipt.output_kind = OutputKind::Repaired;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::WrongOutput)
    );
}

#[test]
fn cpu_substitution_fails_even_if_request_is_relabelled() {
    let (mut receipt, expected) = fixture();
    receipt.actual_device.backend = Backend::Cpu;
    receipt.execution_path = ExecutionPath::CpuReference;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::WrongExecution)
    );
    receipt.requested_backend = Backend::Cpu;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::WrongExecution)
    );
}

#[test]
fn labels_cannot_make_cpu_path_a_gpu_path() {
    let (mut receipt, mut expected) = fixture();
    receipt.execution_path = ExecutionPath::CpuReference;
    expected.execution_path = ExecutionPath::CpuReference;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::WrongExecution)
    );
}

#[test]
fn failed_skipped_unsupported_and_not_measured_remain_distinct() {
    let (mut receipt, expected) = fixture();
    for outcome in [
        RunOutcome::Failed,
        RunOutcome::Skipped,
        RunOutcome::Unsupported,
        RunOutcome::NotMeasured,
    ] {
        receipt.outcome = outcome;
        assert_eq!(
            validate_receipt_declarations(&receipt, &expected),
            Err(ReceiptError::NonPassing(outcome))
        );
    }
}

#[test]
fn rejects_empty_ignored_skipped_and_overflowed_counts() {
    let (mut receipt, mut expected) = fixture();
    for counts in [
        TestCounts {
            passed: 0,
            failed: 0,
            skipped: 0,
            ignored: 0,
        },
        TestCounts {
            passed: 0,
            failed: 0,
            skipped: 3,
            ignored: 0,
        },
        TestCounts {
            passed: 2,
            failed: 0,
            skipped: 0,
            ignored: 1,
        },
        TestCounts {
            passed: u64::MAX,
            failed: 1,
            skipped: 0,
            ignored: 0,
        },
    ] {
        receipt.tests = counts;
        assert_eq!(
            validate_receipt_declarations(&receipt, &expected),
            Err(ReceiptError::TestCounts)
        );
    }
    receipt.tests = TestCounts {
        passed: 0,
        failed: 0,
        skipped: 0,
        ignored: 0,
    };
    expected.expected_tests = 0;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::TestCounts)
    );
}

#[test]
fn estimates_and_unavailable_metrics_do_not_pass_measured_requirements() {
    let (mut receipt, expected) = fixture();
    for metric in [
        Measurement::NotMeasured,
        Measurement::Unavailable,
        Measurement::Estimated(10.0),
    ] {
        receipt.metrics.latency_p50_ns = metric;
        assert_eq!(
            validate_receipt_declarations(&receipt, &expected),
            Err(ReceiptError::TimingEvidence)
        );
    }
    let (mut receipt, expected) = fixture();
    receipt.metrics.allocations = Measurement::Estimated(0);
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::ResourceEvidence)
    );
    receipt.metrics.allocations = Measurement::Measured(0);
    receipt.metrics.graph_launches = Measurement::Measured(0);
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::GraphEvidence)
    );
}

#[test]
fn rejects_nonfinite_metrics_and_tolerance() {
    let (mut receipt, expected) = fixture();
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        receipt.metrics.latency_p50_ns = Measurement::Measured(value);
        assert_eq!(
            validate_receipt_declarations(&receipt, &expected),
            Err(ReceiptError::TimingEvidence)
        );
    }
    let (mut receipt, mut expected) = fixture();
    expected.tolerance.absolute = f64::NAN;
    receipt.tolerance = expected.tolerance;
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::NumericalEvidence)
    );
}

#[test]
fn malformed_identity_and_stale_patch_are_rejected() {
    let (mut receipt, expected) = fixture();
    receipt.build.etalon.revision = "short-sha";
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::MissingOrInvalidIdentity)
    );
    receipt.build = expected.build;
    receipt.build.etalon.patch_sha256 = Some(HASH_B);
    assert_eq!(
        validate_receipt_declarations(&receipt, &expected),
        Err(ReceiptError::StaleBuild)
    );
}

#[test]
fn pair_cannot_compare_different_layouts_even_when_individually_expected() {
    let (baseline, baseline_expected) = fixture();
    let mut candidate = baseline;
    let mut candidate_expected = baseline_expected;
    candidate.workload.layout = "different-layout";
    candidate_expected.workload = candidate.workload;
    assert_eq!(
        validate_pair_declarations(
            &baseline,
            &baseline_expected,
            &candidate,
            &candidate_expected
        ),
        Err(ReceiptError::IncomparablePair)
    );
}

#[test]
fn timing_pair_requires_identical_device_driver_and_compiler() {
    let (baseline, baseline_expected) = fixture();
    for device in [
        DeviceIdentity {
            device: "different-device",
            ..baseline.actual_device
        },
        DeviceIdentity {
            driver: "different-driver",
            ..baseline.actual_device
        },
        DeviceIdentity {
            compiler: "different-compiler",
            ..baseline.actual_device
        },
    ] {
        let mut candidate = baseline;
        let mut candidate_expected = baseline_expected;
        candidate.actual_device = device;
        candidate_expected.device = device;
        assert_eq!(
            validate_pair_declarations(
                &baseline,
                &baseline_expected,
                &candidate,
                &candidate_expected
            ),
            Err(ReceiptError::IncomparablePair)
        );
    }
}

#[test]
fn pair_requires_identical_expected_output_extent() {
    let (baseline, baseline_expected) = fixture();
    let mut candidate = baseline;
    let mut candidate_expected = baseline_expected;
    candidate_expected.expected_values = 1;
    candidate.metrics.numerical =
        Measurement::Measured(compare_f32(&[1.0], &[1.0], candidate_expected.tolerance).unwrap());
    assert_eq!(
        validate_pair_declarations(
            &baseline,
            &baseline_expected,
            &candidate,
            &candidate_expected
        ),
        Err(ReceiptError::IncomparablePair)
    );
}

#[test]
fn governed_labels_require_policy_identity_even_when_expected_agrees() {
    let (mut receipt, mut expected) = fixture();
    for mode in [ExecutionMode::Governed, ExecutionMode::IntegratedTest] {
        receipt.workload.mode = mode;
        expected.workload = receipt.workload;
        assert_eq!(
            validate_receipt_declarations(&receipt, &expected),
            Err(ReceiptError::MissingOrInvalidIdentity)
        );
    }
    receipt.workload.policy_sha256 = Some(HASH_A);
    expected.workload = receipt.workload;
    assert_eq!(validate_receipt_declarations(&receipt, &expected), Ok(()));
}

#[test]
fn numerical_comparator_checks_bounds_special_values_and_real_difference() {
    let exact = Tolerance {
        absolute: 0.0,
        relative: 0.0,
    };
    assert_eq!(
        compare_f32(&[], &[], exact),
        Err(ComparisonError::EmptyInput)
    );
    assert_eq!(
        compare_f32(&[1.0], &[], exact),
        Err(ComparisonError::LengthMismatch)
    );
    assert_eq!(
        compare_f32(&[f32::NAN], &[0.0], exact),
        Err(ComparisonError::NaN { index: 0 })
    );
    assert_eq!(
        compare_f32(&[f32::INFINITY], &[f32::NEG_INFINITY], exact),
        Err(ComparisonError::Infinity { index: 0 })
    );
    assert_eq!(
        compare_f32(&[f32::INFINITY], &[f32::INFINITY], exact),
        Err(ComparisonError::Infinity { index: 0 })
    );
    assert_eq!(
        compare_f32(&[f32::NEG_INFINITY], &[f32::NEG_INFINITY], exact),
        Err(ComparisonError::Infinity { index: 0 })
    );
    assert_eq!(
        compare_f32(
            &[f32::MAX],
            &[f32::MAX],
            Tolerance {
                absolute: f64::MAX,
                relative: f64::MAX
            }
        ),
        Err(ComparisonError::NonFiniteBound { index: 0 })
    );
    assert_eq!(compare_f32(&[1.0], &[2.0], exact).unwrap().mismatches, 1);
    assert_eq!(
        compare_f32(
            &[100.0],
            &[101.0],
            Tolerance {
                absolute: 0.0,
                relative: 0.02
            }
        )
        .unwrap()
        .mismatches,
        0
    );
    assert_eq!(
        compare_f32(
            &[0.0],
            &[0.1],
            Tolerance {
                absolute: 0.0,
                relative: 1.0
            }
        )
        .unwrap()
        .mismatches,
        1
    );
    assert_eq!(
        compare_f32(
            &[0.0],
            &[0.0],
            Tolerance {
                absolute: f64::INFINITY,
                relative: 0.0
            }
        ),
        Err(ComparisonError::InvalidTolerance)
    );
}
