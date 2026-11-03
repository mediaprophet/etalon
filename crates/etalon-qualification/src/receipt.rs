use crate::{NumericalComparison, Tolerance};

pub const RECEIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceIdentity<'a> {
    pub revision: &'a str,
    pub lockfile_sha256: &'a str,
    pub patch_sha256: Option<&'a str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildIdentity<'a> {
    pub qualia: SourceIdentity<'a>,
    pub etalon: SourceIdentity<'a>,
    pub toolchain: &'a str,
    pub features: &'a [&'a str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Developer,
    Governed,
    SimulatedTest,
    IntegratedTest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkloadIdentity<'a> {
    pub fixture_sha256: &'a str,
    pub model_sha256: Option<&'a str>,
    pub shader_sha256: Option<&'a str>,
    pub tokenizer_sha256: Option<&'a str>,
    pub layout: &'a str,
    pub precision: &'a str,
    pub mode: ExecutionMode,
    pub policy_sha256: Option<&'a str>,
    pub cache_generation: u64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Vulkan,
    Dx12,
    Metal,
    BrowserWebGpu,
    Cuda,
    DirectMl,
    Npu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionPath {
    CpuReference,
    CpuOptimized,
    DeviceDispatch,
    CudaGraphReplay,
    NpuDispatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceIdentity<'a> {
    pub backend: Backend,
    pub device: &'a str,
    pub driver: &'a str,
    pub compiler: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceOrigin {
    SyntheticFixture,
    LiveRun,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    Passed,
    Failed,
    Skipped,
    Unsupported,
    NotMeasured,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    Raw,
    Repaired,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Measurement<T> {
    NotMeasured,
    Unavailable,
    Estimated(T),
    Measured(T),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TestCounts {
    pub passed: u64,
    pub failed: u64,
    pub skipped: u64,
    pub ignored: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunMetrics {
    pub numerical: Measurement<NumericalComparison>,
    pub latency_p50_ns: Measurement<f64>,
    pub latency_p95_ns: Measurement<f64>,
    pub allocations: Measurement<u64>,
    pub peak_bytes: Measurement<u64>,
    pub uploaded_bytes: Measurement<u64>,
    pub downloaded_bytes: Measurement<u64>,
    pub graph_launches: Measurement<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunReceipt<'a> {
    pub schema_version: u32,
    pub packet: &'a str,
    pub build: BuildIdentity<'a>,
    pub workload: WorkloadIdentity<'a>,
    pub requested_backend: Backend,
    pub actual_device: DeviceIdentity<'a>,
    pub execution_path: ExecutionPath,
    pub origin: EvidenceOrigin,
    pub outcome: RunOutcome,
    pub output_kind: OutputKind,
    pub output_sha256: &'a str,
    pub command: &'a str,
    pub tests: TestCounts,
    pub tolerance: Tolerance,
    pub metrics: RunMetrics,
    pub limitations: &'a str,
}

/// Frozen separately from the producer's receipt. This is an expectation,
/// not a cryptographic trust root or a supervisor decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpectedRun<'a> {
    pub packet: &'a str,
    pub build: BuildIdentity<'a>,
    pub workload: WorkloadIdentity<'a>,
    pub device: DeviceIdentity<'a>,
    pub execution_path: ExecutionPath,
    pub expected_tests: u64,
    pub expected_values: usize,
    pub output_kind: OutputKind,
    pub exact_output_sha256: Option<&'a str>,
    pub tolerance: Tolerance,
    pub require_live: bool,
    pub require_timing: bool,
    pub require_resource_metrics: bool,
}
