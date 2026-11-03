use etalon_qualification::*;

const QUALIA: &str = "8849505a694ebd6213c6d2c95ac2a61724914f45";
const OLD: &str = "072b4d9ceeebd423234359dff1dee8fbe9e62b6e";
const NEW: &str = "a64f24b504ff42cd9a9e9062e4ff69ec01fb230e";
const LOCK: &str = "88062E10CE0D6B00F596A0F99BF48C0904C95CDDF02E89C5BE711E8351F23245";
const QLOCK: &str = "E782E0B0823A8BA21FB7A53A701AF10ADE8E18818618149A5BBC4018FCD51557";
const SYNTHETIC_HASH: &str = "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0";
const EXACT: Tolerance = Tolerance {
    absolute: 0.0,
    relative: 0.0,
};

fn expected() -> ExpectedRun<'static> {
    ExpectedRun {
        packet: "EM01A-independent-synthetic-fixture",
        build: BuildIdentity {
            qualia: SourceIdentity {
                revision: QUALIA,
                lockfile_sha256: QLOCK,
                patch_sha256: None,
            },
            etalon: SourceIdentity {
                revision: NEW,
                lockfile_sha256: LOCK,
                patch_sha256: None,
            },
            toolchain: "synthetic; not execution metadata",
            features: &[],
        },
        workload: WorkloadIdentity {
            fixture_sha256: SYNTHETIC_HASH,
            model_sha256: None,
            shader_sha256: None,
            tokenizer_sha256: None,
            layout: "q4_k_soa:type112:v1:160bytes:256weights:little-endian",
            precision: "f32",
            mode: ExecutionMode::Developer,
            policy_sha256: None,
            cache_generation: 42,
            seed: 19,
        },
        device: DeviceIdentity {
            backend: Backend::Cuda,
            device: "synthetic gpu",
            driver: "fixture-driver",
            compiler: "fixture-compiler",
        },
        execution_path: ExecutionPath::DeviceDispatch,
        expected_tests: 4,
        expected_values: 4,
        output_kind: OutputKind::Raw,
        exact_output_sha256: Some(SYNTHETIC_HASH),
        tolerance: EXACT,
        require_live: false,
        require_timing: false,
        require_resource_metrics: false,
    }
}

fn receipt() -> RunReceipt<'static> {
    let e = expected();
    RunReceipt {
        schema_version: 1, packet: e.packet, build: e.build, workload: e.workload,
        requested_backend: Backend::Cuda, actual_device: e.device,
        execution_path: ExecutionPath::DeviceDispatch, origin: EvidenceOrigin::SyntheticFixture,
        outcome: RunOutcome::Passed, output_kind: OutputKind::Raw, output_sha256: SYNTHETIC_HASH,
        command: "synthetic declaration; no GPU executed",
        tests: TestCounts { passed: 4, failed: 0, skipped: 0, ignored: 0 }, tolerance: EXACT,
        metrics: RunMetrics {
            numerical: Measurement::Measured(NumericalComparison { compared: 4, mismatches: 0, max_absolute_error: 0.0 }),
            latency_p50_ns: Measurement::NotMeasured, latency_p95_ns: Measurement::NotMeasured,
            allocations: Measurement::NotMeasured, peak_bytes: Measurement::NotMeasured,
            uploaded_bytes: Measurement::NotMeasured, downloaded_bytes: Measurement::NotMeasured,
            graph_launches: Measurement::NotMeasured,
        },
        limitations: "Synthetic declaration validation only; frozen hashes are labels, not proof of execution",
    }
}

#[test]
fn f01_f02_independent_affine_oracle_and_corruption() {
    let actual = [0.0f32, 1.0, -2.0, 3.5].map(|x| x * 2.0 + 1.0);
    let oracle = [1.0, 3.0, -3.0, 8.0];
    assert_eq!(compare_f32(&oracle, &actual, EXACT).unwrap().mismatches, 0);
    assert_eq!(
        compare_f32(&oracle, &[1.0, 3.0, -3.0, 7.0], EXACT)
            .unwrap()
            .mismatches,
        1
    );
    let mut r = receipt();
    r.metrics.numerical =
        Measurement::Measured(compare_f32(&oracle, &[1.0, 3.0, -3.0, 7.0], EXACT).unwrap());
    assert_eq!(
        validate_receipt_declarations(&r, &expected()),
        Err(ReceiptError::NumericalEvidence)
    );
}

#[test]
fn f03_empty_skipped_and_overflow_cannot_pass() {
    assert_eq!(
        compare_f32(&[], &[], EXACT),
        Err(ComparisonError::EmptyInput)
    );
    for counts in [(0, 0, 0), (0, 4, 0), (u64::MAX, 0, 1)] {
        let mut r = receipt();
        r.tests = TestCounts {
            passed: counts.0,
            skipped: counts.1,
            failed: counts.2,
            ignored: 0,
        };
        assert_eq!(
            validate_receipt_declarations(&r, &expected()),
            Err(ReceiptError::TestCounts)
        );
    }
}

#[test]
fn independently_expected_output_count_rejects_partial_or_empty_comparison() {
    for compared in [0, 1, 3, 5, usize::MAX] {
        let mut r = receipt();
        r.metrics.numerical = Measurement::Measured(NumericalComparison {
            compared,
            mismatches: 0,
            max_absolute_error: 0.0,
        });
        assert_eq!(
            validate_receipt_declarations(&r, &expected()),
            Err(ReceiptError::NumericalEvidence)
        );
    }
    let mut e = expected();
    e.expected_values = 0;
    assert_eq!(
        validate_receipt_declarations(&receipt(), &e),
        Err(ReceiptError::NumericalEvidence)
    );
    let mut partial = receipt();
    partial.metrics.numerical = Measurement::Measured(NumericalComparison {
        compared: 2,
        mismatches: 0,
        max_absolute_error: 0.0,
    });
    e.expected_values = 2;
    assert_eq!(
        validate_pair_declarations(&partial, &e, &receipt(), &expected()),
        Err(ReceiptError::IncomparablePair)
    );
}

#[test]
fn f04_equal_infinities_and_finite_bound_overflow_rejected() {
    for v in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(compare_f32(&[v], &[v], EXACT).is_err());
    }
    for tolerance in [
        Tolerance {
            absolute: -1.0,
            relative: 0.0,
        },
        Tolerance {
            absolute: f64::NAN,
            relative: 0.0,
        },
        Tolerance {
            absolute: 0.0,
            relative: f64::INFINITY,
        },
    ] {
        assert_eq!(
            compare_f32(&[1.0], &[1.0], tolerance),
            Err(ComparisonError::InvalidTolerance)
        );
    }
    assert_eq!(
        compare_f32(
            &[2.0],
            &[2.0],
            Tolerance {
                absolute: f64::MAX,
                relative: f64::MAX
            }
        ),
        Err(ComparisonError::NonFiniteBound { index: 0 })
    );
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut r = receipt();
        r.metrics.latency_p50_ns = Measurement::Estimated(value);
        assert_eq!(
            validate_receipt_declarations(&r, &expected()),
            Err(ReceiptError::TimingEvidence)
        );
    }
}

#[test]
fn f05_f06_independent_revision_and_cache_reject_matching_stale_pair() {
    let mut r = receipt();
    r.build.etalon.revision = OLD;
    assert_eq!(
        validate_pair_declarations(&r, &expected(), &r, &expected()),
        Err(ReceiptError::StaleBuild)
    );
    r = receipt();
    r.workload.cache_generation = 41;
    assert_eq!(
        validate_pair_declarations(&r, &expected(), &r, &expected()),
        Err(ReceiptError::WrongWorkload)
    );
}

#[test]
fn f07_gpu_claim_cannot_accept_cpu_path() {
    let mut r = receipt();
    r.execution_path = ExecutionPath::CpuReference;
    assert_eq!(
        validate_receipt_declarations(&r, &expected()),
        Err(ReceiptError::WrongExecution)
    );
    r.actual_device.backend = Backend::Cpu;
    r.requested_backend = Backend::Cpu;
    assert_eq!(
        validate_receipt_declarations(&r, &expected()),
        Err(ReceiptError::WrongExecution)
    );
}

#[test]
fn f09_f10_layout_packing_and_endianness_are_identity() {
    for layout in [
        "q4_k_soa:type112:v1:144bytes:256weights:little-endian",
        "q4_k_soa:type112:v1:160bytes:256weights:big-endian",
        "q4_k_soa:type112:v2:160bytes:256weights:little-endian",
    ] {
        let mut r = receipt();
        r.workload.layout = layout;
        assert_eq!(
            validate_receipt_declarations(&r, &expected()),
            Err(ReceiptError::WrongWorkload)
        );
        let mut changed_expected = expected();
        changed_expected.workload = r.workload;
        assert_eq!(
            validate_pair_declarations(&receipt(), &expected(), &r, &changed_expected),
            Err(ReceiptError::IncomparablePair)
        );
    }
}

#[test]
fn f11_synthetic_declarations_are_not_live_evidence() {
    assert_eq!(
        validate_receipt_declarations(&receipt(), &expected()),
        Ok(())
    );
    let mut e = expected();
    e.require_live = true;
    assert_eq!(
        validate_receipt_declarations(&receipt(), &e),
        Err(ReceiptError::SyntheticEvidence)
    );
}

#[test]
fn governed_and_integrated_declarations_require_policy_identity() {
    for mode in [ExecutionMode::Governed, ExecutionMode::IntegratedTest] {
        let mut r = receipt();
        let mut e = expected();
        r.workload.mode = mode;
        e.workload.mode = mode;
        assert_eq!(
            validate_receipt_declarations(&r, &e),
            Err(ReceiptError::MissingOrInvalidIdentity)
        );
        r.workload.policy_sha256 = Some(SYNTHETIC_HASH);
        e.workload.policy_sha256 = Some(SYNTHETIC_HASH);
        assert_eq!(validate_receipt_declarations(&r, &e), Ok(()));
    }
}

#[test]
fn timing_comparison_requires_same_device_driver_and_compiler() {
    let mut base = receipt();
    base.metrics.latency_p50_ns = Measurement::Measured(10.0);
    base.metrics.latency_p95_ns = Measurement::Measured(20.0);
    let mut e = expected();
    e.require_timing = true;
    for device in [
        DeviceIdentity {
            device: "other",
            ..e.device
        },
        DeviceIdentity {
            driver: "other",
            ..e.device
        },
        DeviceIdentity {
            compiler: "other",
            ..e.device
        },
    ] {
        let mut candidate = base;
        candidate.actual_device = device;
        let mut changed_expected = e;
        changed_expected.device = device;
        assert_eq!(
            validate_pair_declarations(&base, &e, &candidate, &changed_expected),
            Err(ReceiptError::IncomparablePair)
        );
    }
}
