use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptError {
    UnsupportedSchema,
    MissingOrInvalidIdentity,
    StaleBuild,
    WrongWorkload,
    WrongExecution,
    SyntheticEvidence,
    NonPassing(RunOutcome),
    TestCounts,
    WrongOutput,
    NumericalEvidence,
    TimingEvidence,
    ResourceEvidence,
    GraphEvidence,
    IncomparablePair,
}

fn digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|b| b.is_ascii_hexdigit())
        && value.bytes().any(|b| b != b'0')
}

fn optional_digest(value: Option<&str>) -> bool {
    value.is_none_or(|value| digest(value, 64))
}

fn source(value: SourceIdentity<'_>) -> bool {
    digest(value.revision, 40)
        && digest(value.lockfile_sha256, 64)
        && optional_digest(value.patch_sha256)
}

fn valid_identity(receipt: &RunReceipt<'_>) -> bool {
    let w = receipt.workload;
    source(receipt.build.qualia)
        && source(receipt.build.etalon)
        && !receipt.build.toolchain.trim().is_empty()
        && receipt.build.features.iter().all(|s| !s.trim().is_empty())
        && !receipt.packet.trim().is_empty()
        && !receipt.command.trim().is_empty()
        && !receipt.actual_device.device.trim().is_empty()
        && !receipt.actual_device.driver.trim().is_empty()
        && !receipt.actual_device.compiler.trim().is_empty()
        && digest(w.fixture_sha256, 64)
        && optional_digest(w.model_sha256)
        && optional_digest(w.shader_sha256)
        && optional_digest(w.tokenizer_sha256)
        && optional_digest(w.policy_sha256)
        && (!matches!(
            w.mode,
            ExecutionMode::Governed | ExecutionMode::IntegratedTest
        ) || w.policy_sha256.is_some())
        && !w.layout.trim().is_empty()
        && !w.precision.trim().is_empty()
        && digest(receipt.output_sha256, 64)
}

fn path_matches(backend: Backend, path: ExecutionPath) -> bool {
    match backend {
        Backend::Cpu => matches!(
            path,
            ExecutionPath::CpuReference | ExecutionPath::CpuOptimized
        ),
        Backend::Cuda => matches!(
            path,
            ExecutionPath::DeviceDispatch | ExecutionPath::CudaGraphReplay
        ),
        Backend::Npu => path == ExecutionPath::NpuDispatch,
        _ => path == ExecutionPath::DeviceDispatch,
    }
}

fn measured<T>(metric: Measurement<T>) -> bool {
    matches!(metric, Measurement::Measured(_))
}

fn valid_time(metric: Measurement<f64>) -> bool {
    match metric {
        Measurement::Measured(value) | Measurement::Estimated(value) => {
            value.is_finite() && value >= 0.0
        }
        _ => true,
    }
}

/// Checks declarations against independently frozen expectations. A successful
/// result does not establish that the declared tests or hardware execution occurred.
pub fn validate_receipt_declarations(
    receipt: &RunReceipt<'_>,
    expected: &ExpectedRun<'_>,
) -> Result<(), ReceiptError> {
    if receipt.schema_version != RECEIPT_SCHEMA_VERSION {
        return Err(ReceiptError::UnsupportedSchema);
    }
    if !valid_identity(receipt) || !optional_digest(expected.exact_output_sha256) {
        return Err(ReceiptError::MissingOrInvalidIdentity);
    }
    if receipt.build != expected.build || receipt.packet != expected.packet {
        return Err(ReceiptError::StaleBuild);
    }
    if receipt.workload != expected.workload {
        return Err(ReceiptError::WrongWorkload);
    }
    if receipt.requested_backend != expected.device.backend
        || receipt.actual_device != expected.device
        || receipt.execution_path != expected.execution_path
        || !path_matches(receipt.actual_device.backend, receipt.execution_path)
    {
        return Err(ReceiptError::WrongExecution);
    }
    if expected.require_live && receipt.origin != EvidenceOrigin::LiveRun {
        return Err(ReceiptError::SyntheticEvidence);
    }
    if receipt.outcome != RunOutcome::Passed {
        return Err(ReceiptError::NonPassing(receipt.outcome));
    }
    let t = receipt.tests;
    let total = t
        .passed
        .checked_add(t.failed)
        .and_then(|v| v.checked_add(t.skipped))
        .and_then(|v| v.checked_add(t.ignored));
    if expected.expected_tests == 0
        || total != Some(expected.expected_tests)
        || t.passed != expected.expected_tests
        || t.failed != 0
        || t.skipped != 0
        || t.ignored != 0
    {
        return Err(ReceiptError::TestCounts);
    }
    if receipt.output_kind != expected.output_kind
        || expected
            .exact_output_sha256
            .is_some_and(|hash| hash != receipt.output_sha256)
    {
        return Err(ReceiptError::WrongOutput);
    }
    if !expected.tolerance.is_valid() || receipt.tolerance != expected.tolerance {
        return Err(ReceiptError::NumericalEvidence);
    }
    match receipt.metrics.numerical {
        Measurement::Measured(result)
            if expected.expected_values > 0
                && result.compared == expected.expected_values
                && result.mismatches == 0
                && result.max_absolute_error.is_finite()
                && result.max_absolute_error >= 0.0 => {}
        _ => return Err(ReceiptError::NumericalEvidence),
    }
    let m = receipt.metrics;
    if !valid_time(m.latency_p50_ns)
        || !valid_time(m.latency_p95_ns)
        || (expected.require_timing && (!measured(m.latency_p50_ns) || !measured(m.latency_p95_ns)))
    {
        return Err(ReceiptError::TimingEvidence);
    }
    if let (Measurement::Measured(p50), Measurement::Measured(p95)) =
        (m.latency_p50_ns, m.latency_p95_ns)
    {
        if p50 > p95 {
            return Err(ReceiptError::TimingEvidence);
        }
    }
    if expected.require_resource_metrics
        && !(measured(m.allocations)
            && measured(m.peak_bytes)
            && measured(m.uploaded_bytes)
            && measured(m.downloaded_bytes))
    {
        return Err(ReceiptError::ResourceEvidence);
    }
    if receipt.execution_path == ExecutionPath::CudaGraphReplay
        && !matches!(m.graph_launches, Measurement::Measured(count) if count > 0)
    {
        return Err(ReceiptError::GraphEvidence);
    }
    Ok(())
}

/// Each side has its own source and cache expectation; self-consistency between
/// two stale receipts is insufficient. Timing pairs require the same device,
/// driver and compiler. No performance uplift is inferred here.
pub fn validate_pair_declarations(
    baseline: &RunReceipt<'_>,
    baseline_expected: &ExpectedRun<'_>,
    candidate: &RunReceipt<'_>,
    candidate_expected: &ExpectedRun<'_>,
) -> Result<(), ReceiptError> {
    validate_receipt_declarations(baseline, baseline_expected)?;
    validate_receipt_declarations(candidate, candidate_expected)?;
    let mut baseline_workload = baseline.workload;
    baseline_workload.cache_generation = candidate.workload.cache_generation;
    if baseline_workload != candidate.workload
        || baseline_expected.expected_values != candidate_expected.expected_values
        || baseline.requested_backend != candidate.requested_backend
        || baseline.execution_path != candidate.execution_path
        || baseline.output_kind != candidate.output_kind
        || baseline.tolerance != candidate.tolerance
        || ((baseline_expected.require_timing || candidate_expected.require_timing)
            && baseline.actual_device != candidate.actual_device)
    {
        return Err(ReceiptError::IncomparablePair);
    }
    Ok(())
}
