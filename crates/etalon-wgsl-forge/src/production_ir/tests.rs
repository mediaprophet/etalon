use crate::cache::{AdapterIdentity, TimingSource, TimingSummary, ValidationLevel};
use crate::certify::{ComparisonReport, ValidationReport};
use crate::emit::GeneratedShader;
use crate::ir::{ForgeError, Op};
use crate::schedule::Schedule;

use super::*;

fn schedule() -> Schedule {
    Schedule::default()
}

fn adapter() -> AdapterIdentity {
    AdapterIdentity {
        name: "test-adapter".to_string(),
        vendor: 1,
        device: 2,
        device_type: "DiscreteGpu".to_string(),
        backend: "Vulkan".to_string(),
        driver: "test".to_string(),
        driver_info: "1".to_string(),
    }
}

fn shader_for(contract: &ProductionContract) -> GeneratedShader {
    let source = format!(
        "// {}\n@compute @workgroup_size(1) fn {}() {{}}",
        contract.kernel.id, contract.kernel.entry_point
    );
    GeneratedShader {
        kernel_id: contract.kernel.id.clone(),
        semantic_hash: contract.kernel.semantic_hash().expect("semantic hash"),
        source_hash: blake3::hash(source.as_bytes()).to_hex().to_string(),
        schedule: contract.schedule,
        source,
    }
}

fn naga_report(shader: &GeneratedShader) -> ValidationReport {
    ValidationReport {
        source_hash: shader.source_hash.clone(),
        entry_points: vec![shader.kernel_id.clone()],
        binding_count: 0,
        naga_validated: true,
        native_tool_validated: None,
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

fn gpu_timing() -> TimingSummary {
    TimingSummary::from_samples(TimingSource::GpuTimestamp, &[100, 110, 120]).unwrap()
}

fn issued(contract: &ProductionContract) -> (GeneratedShader, StageReceipt) {
    let shader = shader_for(contract);
    let options = StageOptions::wgsl(contract.arithmetic);
    let receipt = StageReceipt::issue_generated(contract, &shader, &options, None).unwrap();
    (shader, receipt)
}

fn evidence<'a>(
    shader: &'a GeneratedShader,
    validation: Option<&'a ValidationReport>,
    adapter: Option<&'a AdapterIdentity>,
    oracle: Option<&'a ComparisonReport>,
    timing: Option<&'a TimingSummary>,
    measurement: MeasurementMethod,
) -> PromotionEvidence<'a> {
    PromotionEvidence {
        shader,
        validation,
        adapter,
        oracle,
        timing,
        measurement,
    }
}

#[test]
fn reduction_sparse_and_geometry_contracts_validate() {
    let reduction = ProductionContract::reduction_sum(schedule()).unwrap();
    let sparse = ProductionContract::sparse_scatter_add(schedule()).unwrap();
    let geometry = ProductionContract::geometry_point_in_tetra(schedule()).unwrap();
    assert_eq!(reduction.family(), MathFamily::Reduction);
    assert_eq!(sparse.family(), MathFamily::SparseOperator);
    assert_eq!(geometry.family(), MathFamily::GeometrySelection);
    assert_eq!(reduction.discretization, Discretization::TreeReduceShared);
    assert_eq!(sparse.discretization, Discretization::AtomicCasBitcast);
    assert_eq!(geometry.discretization, Discretization::FilteredOrient3d);
    assert_eq!(reduction.arithmetic, ArithmeticModel::IeeeF32);
    assert_eq!(geometry.arithmetic, ArithmeticModel::FilteredPredicate);
    assert!(reduction.canonical_hash().unwrap().len() == 64);
    assert_ne!(
        reduction.canonical_hash().unwrap(),
        sparse.canonical_hash().unwrap()
    );
}

#[test]
fn invalid_dimensions_are_rejected() {
    let mut reduction = ProductionContract::reduction_sum(schedule()).unwrap();
    if let MathOp::Reduction { axis, .. } = &mut reduction.math {
        *axis = 4;
    }
    match reduction.validate() {
        Err(ForgeError::InvalidKernel(message)) => {
            assert!(message.contains("axis"), "{message}");
        }
        other => panic!("expected invalid axis, got {other:?}"),
    }

    let mut geometry = ProductionContract::geometry_point_in_tetra(schedule()).unwrap();
    geometry.tensors[0].shape.dims = vec![0, 14];
    match geometry.validate() {
        Err(ForgeError::InvalidKernel(message)) => {
            assert!(message.contains("15-wide"), "{message}");
        }
        other => panic!("expected packed-width error, got {other:?}"),
    }

    let mut sparse = ProductionContract::sparse_scatter_add(schedule()).unwrap();
    sparse.tensors[1].shape.dims = vec![8];
    match sparse.validate() {
        Err(ForgeError::InvalidKernel(message)) => {
            assert!(message.contains("identical shapes"), "{message}");
        }
        other => panic!("expected sparse shape error, got {other:?}"),
    }
}

#[test]
fn undeclared_aliasing_is_rejected() {
    let mut contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let extra = TensorRole {
        name: "values_view".to_string(),
        buffer: "input".to_string(),
        shape: contract.tensors[0].shape.clone(),
        element: contract.tensors[0].element,
        unit: PhysicalUnit::Dimensionless,
        effects: Effects::read_only(),
    };
    contract.tensors.push(extra);
    match contract.validate() {
        Err(ForgeError::InvalidKernel(message)) => {
            assert!(message.contains("undeclared aliasing"), "{message}");
        }
        other => panic!("expected undeclared aliasing, got {other:?}"),
    }

    contract.aliasing.push(AliasingDecl {
        left: "values".to_string(),
        right: "values_view".to_string(),
        kind: AliasingKind::MayAlias,
    });
    contract.validate().expect("declared MayAlias is legal");
}

#[test]
fn floating_point_rewrite_legality_is_not_real_algebra() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let (rewritten, trace) = apply_guarded(
        contract.clone(),
        &[
            RewriteRequest::CommuteAdd,
            RewriteRequest::ReassociateAdd,
            RewriteRequest::CancelDiv,
        ],
    )
    .unwrap();
    assert_eq!(trace.applied.len(), 1);
    assert_eq!(trace.applied[0].rule, RewriteRequest::CommuteAdd);
    assert_eq!(trace.applied[0].kind, RewriteKind::FloatingPoint);
    assert_eq!(trace.rejected.len(), 2);
    assert!(trace
        .rejected
        .iter()
        .any(|item| item.rule == RewriteRequest::ReassociateAdd
            && item.reason.contains("not real-algebra legal")));
    assert!(trace
        .rejected
        .iter()
        .any(|item| item.rule == RewriteRequest::CancelDiv
            && item.reason.contains("domain-invalid")));
    assert!(contains_swapped_add(&rewritten.kernel.ops));
}

fn contains_swapped_add(ops: &[Op]) -> bool {
    ops.iter().any(|op| match op {
        Op::Add { left, right, .. } => left == "x" && right == "acc",
        Op::Loop { body, .. } => contains_swapped_add(body),
        _ => false,
    })
}

#[test]
fn domain_invalid_and_illegal_rewrites_are_rejected() {
    let sparse = ProductionContract::sparse_scatter_add(schedule()).unwrap();
    let (_, trace) = apply_guarded(
        sparse,
        &[
            RewriteRequest::UnboundedCasSpin,
            RewriteRequest::PromoteDf64ToNativeF64,
        ],
    )
    .unwrap();
    assert!(trace.applied.is_empty());
    assert!(trace.rejected.iter().any(|item| {
        item.rule == RewriteRequest::UnboundedCasSpin && item.reason.contains("occupancy-bounded")
    }));
    assert!(trace.rejected.iter().any(|item| {
        item.rule == RewriteRequest::PromoteDf64ToNativeF64 && item.reason.contains("paired f32")
    }));

    let geometry = ProductionContract::geometry_point_in_tetra(schedule()).unwrap();
    let (same, trace) =
        apply_guarded(geometry.clone(), &[RewriteRequest::CollapseGeometryUncertain]).unwrap();
    assert!(trace.rejected.iter().any(|item| item
        .reason
        .contains("geometry_uncertain_policy")));
    match (&geometry.math, &same.math) {
        (
            MathOp::Geometry {
                uncertain_code: a, ..
            },
            MathOp::Geometry {
                uncertain_code: b, ..
            },
        ) => assert_eq!(a, b),
        _ => panic!("geometry math changed family"),
    }
}

#[test]
fn fma_fuse_records_assumption_trace_on_existing_ir() {
    let mut contract = ProductionContract::reduction_sum(schedule()).unwrap();
    contract.kernel.ops.insert(
        0,
        Op::Mul {
            left: "in_val".to_string(),
            right: "scale".to_string(),
            destination: "scaled".to_string(),
        },
    );
    contract.kernel.ops.insert(
        1,
        Op::Add {
            left: "scaled".to_string(),
            right: "bias".to_string(),
            destination: "out_val".to_string(),
        },
    );
    let (_, denied) = apply_guarded(contract.clone(), &[RewriteRequest::FuseFma]).unwrap();
    assert!(denied
        .rejected
        .iter()
        .any(|item| item.reason.contains("fma_contract")));

    let contract = contract
        .with_assumption(Assumption {
            id: "fma_contract".to_string(),
            kind: AssumptionKind::FloatingPoint,
            statement: "caller accepts fused multiply-add rounding".to_string(),
        })
        .unwrap();
    let (fused, trace) = apply_guarded(contract, &[RewriteRequest::FuseFma]).unwrap();
    assert_eq!(trace.applied[0].assumptions, ["fma_contract"]);
    assert!(matches!(fused.kernel.ops[0], Op::Fma { .. }));
}

#[test]
fn rewrite_engine_stays_bounded() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let requests = [RewriteRequest::MulIdentity; MAX_REWRITE_REQUESTS + 1];
    match apply_guarded(contract, &requests) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("cold-path bound"), "{message}");
        }
        other => panic!("expected bound error, got {other:?}"),
    }
}

#[test]
fn stage_receipts_use_canonical_keys_and_reuse_adapter_identity() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let (shader, generated) = issued(&contract);
    generated.replay().unwrap();
    assert_eq!(generated.validation_level, ValidationLevel::Generated);
    assert_eq!(generated.cache_key.len(), 64);

    let report = naga_report(&shader);
    let naga = generated
        .promote(
            EvidenceStage::NagaValidated,
            evidence(
                &shader,
                Some(&report),
                None,
                None,
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    assert_eq!(naga.validation_level, ValidationLevel::NagaValidated);
    assert_eq!(
        naga.certification.as_ref().unwrap().validation_level,
        ValidationLevel::NagaValidated
    );

    let identity = adapter();
    let pipeline = naga
        .promote(
            EvidenceStage::PipelineCreated,
            evidence(
                &shader,
                None,
                Some(&identity),
                None,
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    let passed = oracle_pass();
    let oracle = pipeline
        .promote(
            EvidenceStage::OracleVerified,
            evidence(
                &shader,
                None,
                Some(&identity),
                Some(&passed),
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    let timing = gpu_timing();
    let profiled = oracle
        .promote(
            EvidenceStage::Profiled,
            evidence(
                &shader,
                None,
                Some(&identity),
                None,
                Some(&timing),
                MeasurementMethod::GpuTimestamp,
            ),
        )
        .unwrap();
    let certified = profiled
        .promote(
            EvidenceStage::Certified,
            evidence(
                &shader,
                None,
                Some(&identity),
                None,
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    certified.replay().unwrap();
    let manifest = certified.certification.as_ref().unwrap();
    assert_eq!(manifest.validation_level, ValidationLevel::Certified);
    assert_eq!(
        manifest.cache_key.as_deref(),
        Some(
            identity
                .cache_key(
                    &shader.semantic_hash,
                    &shader.source_hash,
                    shader.schedule,
                    (1.0e-6, 1.0e-5),
                )
                .unwrap()
                .as_str()
        )
    );
    assert_ne!(generated.cache_key, naga.cache_key);
    assert_ne!(naga.cache_key, pipeline.cache_key);
}

#[test]
fn stale_naga_evidence_is_not_promoted() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let (shader, generated) = issued(&contract);
    let mut report = naga_report(&shader);
    report.source_hash = blake3::hash(b"stale").to_hex().to_string();
    match generated.promote(
        EvidenceStage::NagaValidated,
        evidence(
            &shader,
            Some(&report),
            None,
            None,
            None,
            MeasurementMethod::None,
        ),
    ) {
        Err(ForgeError::Validation(message)) => {
            assert!(
                message.contains("stale") || message.contains("mismatched"),
                "{message}"
            );
        }
        other => panic!("expected stale promotion refusal, got {other:?}"),
    }

    let identity = adapter();
    let fresh = naga_report(&shader);
    let passed = oracle_pass();
    match generated.promote(
        EvidenceStage::OracleVerified,
        evidence(
            &shader,
            Some(&fresh),
            Some(&identity),
            Some(&passed),
            None,
            MeasurementMethod::None,
        ),
    ) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("not skipped"), "{message}");
        }
        other => panic!("expected skip refusal, got {other:?}"),
    }
}

#[test]
fn synthetic_timing_is_not_gpu_timing_and_crc_is_not_a_signature() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let (shader, generated) = issued(&contract);
    let identity = adapter();
    let report = naga_report(&shader);
    let naga = generated
        .promote(
            EvidenceStage::NagaValidated,
            evidence(
                &shader,
                Some(&report),
                None,
                None,
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    let pipeline = naga
        .promote(
            EvidenceStage::PipelineCreated,
            evidence(
                &shader,
                None,
                Some(&identity),
                None,
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    let passed = oracle_pass();
    let oracle = pipeline
        .promote(
            EvidenceStage::OracleVerified,
            evidence(
                &shader,
                None,
                Some(&identity),
                Some(&passed),
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    let synthetic = TimingSummary::from_samples(TimingSource::Synthetic, &[1, 2, 3]).unwrap();
    match oracle.promote(
        EvidenceStage::Profiled,
        evidence(
            &shader,
            None,
            Some(&identity),
            None,
            Some(&synthetic),
            MeasurementMethod::GpuTimestamp,
        ),
    ) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("synthetic"), "{message}");
        }
        other => panic!("expected synthetic-as-GPU refusal, got {other:?}"),
    }

    match Digest::parse("crc32", "deadbeef") {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("checksum"), "{message}");
        }
        other => panic!("expected CRC refusal, got {other:?}"),
    }
}

#[test]
fn replay_rejects_missing_hashes_and_unsupported_versions() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let (_, mut receipt) = issued(&contract);
    receipt.schema_version = 99;
    match receipt.replay() {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("unsupported evidence schema"), "{message}");
        }
        other => panic!("expected unsupported version, got {other:?}"),
    }

    let (_, mut missing) = issued(&contract);
    missing.input_digest.hex.clear();
    match missing.replay() {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("missing"), "{message}");
        }
        other => panic!("expected missing hash, got {other:?}"),
    }
}

#[test]
fn transitive_invalidation_demotes_dependent_receipts() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let (shader, generated) = issued(&contract);
    let report = naga_report(&shader);
    let naga = generated
        .promote(
            EvidenceStage::NagaValidated,
            evidence(
                &shader,
                Some(&report),
                None,
                None,
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    let identity = adapter();
    let pipeline = naga
        .promote(
            EvidenceStage::PipelineCreated,
            evidence(
                &shader,
                None,
                Some(&identity),
                None,
                None,
                MeasurementMethod::None,
            ),
        )
        .unwrap();
    assert!(!naga.is_stale_child_of(&generated));
    let mut descendants = [naga.clone(), pipeline];
    let demoted = invalidate_descendants(&generated, &mut descendants);
    assert_eq!(demoted, 2);
    assert_eq!(descendants[0].stage, EvidenceStage::Generated);
    assert_eq!(descendants[1].stage, EvidenceStage::Generated);
    assert!(descendants[0].certification.is_none());
    assert!(descendants[1].adapter.is_none());
}

#[test]
fn precision_and_source_changes_mint_new_keys() {
    let contract = ProductionContract::reduction_sum(schedule()).unwrap();
    let (shader, generated) = issued(&contract);
    let other_options = StageOptions::wgsl(ArithmeticModel::Df64PairedF32);
    match StageReceipt::issue_generated(&contract, &shader, &other_options, None) {
        Err(ForgeError::Validation(message)) => {
            assert!(message.contains("precision"), "{message}");
        }
        other => panic!("expected precision mismatch, got {other:?}"),
    }

    let mut changed = shader.clone();
    changed.source.push_str("\n// mutated");
    changed.source_hash = blake3::hash(changed.source.as_bytes()).to_hex().to_string();
    let mutated = StageReceipt::issue_generated(
        &contract,
        &changed,
        &StageOptions::wgsl(contract.arithmetic),
        None,
    )
    .unwrap();
    assert_ne!(generated.cache_key, mutated.cache_key);
    assert_ne!(mutated.source_digest, generated.source_digest);
}
