use crate::config_space::{Configuration, ConfigurationSpace};
pub use crate::evidence::{append_experiment_jsonl, load_experiment_log};
use etalon_inference::runtime::receipt::execution::ExecutionReceipt;
use serde::{Deserialize, Serialize};
/// Evaluation mode distinguishing live measurements from simulation fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationMode {
    /// Measured execution against a real backend and model weights.
    MeasuredLiveBackend,
    /// Synthetic or simulated test fixture (cannot be promoted to production lock-in).
    SimulationFixture,
}

impl Default for EvaluationMode {
    fn default() -> Self {
        Self::MeasuredLiveBackend
    }
}

/// Thermal reading before and after an experiment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct ThermalSnapshot {
    pub before_temp_c: u32,
    pub before_power_w: f64,
    pub after_temp_c: u32,
    pub after_power_w: f64,
    /// Energy estimate in joules: avg_power_w * duration_s.
    pub energy_j: f64,
}

/// Serializable subset of `BenchResult` (the original only derives `Serialize`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchResultSerde {
    pub label: String,
    pub quantization: String,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub cold_ttft_ms: f64,
    pub cold_total_ms: f64,
    pub warm_ttft_ms: f64,
    pub warm_total_ms: f64,
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub prefill_tok_s: f64,
    pub decode_ms: f64,
    pub decode_tok_s: f64,
    pub mapped_bytes: u64,
    pub kv_cache_bytes: u64,
}

/// The full result of one experiment run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// FNV-1a hash of the configuration CBOR (dedup key).
    pub config_hash: u64,
    /// Execution mode (F14 gate: simulation cannot masquerade as measured promotion).
    #[serde(default)]
    pub evaluation_mode: EvaluationMode,
    /// Schema 3 execution receipt (F8/F14).
    #[serde(default)]
    pub receipt: Option<ExecutionReceipt>,
    /// Links to the hypothesis backlog (if any).
    pub hypothesis_id: Option<String>,
    /// The benchmark result (latency, tok/s, phase split).
    pub bench: Option<BenchResultSerde>,
    /// Intra-decode phase breakdown.
    pub phase: PhaseSnapshotSerde,
    /// Post-turn quality checks.
    pub quality: QualityScore,
    /// GPU thermal before/after.
    pub thermal: ThermalSnapshot,
    /// VRAM used (bytes) from the GPU context ledger.
    pub vram_used: u64,
    /// CBOR config snapshot for reproduction.
    pub config_cbor: Vec<u8>,
    /// Unix epoch timestamp (ms).
    pub timestamp_ms: u64,
    /// Sampler seed for reproducibility.
    pub seed: u64,
    /// Error message if the experiment failed.
    pub error: Option<String>,
    #[serde(default)]
    pub evidence: TrialEvidence,
}

/// Serializable phase snapshot (the atomic counters are copied into plain u64s).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct PhaseSnapshotSerde {
    pub load_ns: u64,
    pub prefill_ns: u64,
    pub prefill_tokens: u64,
    pub decode_ns: u64,
    pub decode_tokens: u64,
    pub decode_forward_ns: u64,
    pub decode_output_ns: u64,
}

/// Quality score derived from `VerifiedTurn` checks.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityScore {
    /// Fraction of checks that passed (0.0 to 1.0).
    pub pass_rate: f64,
    /// Total number of checks.
    pub total_checks: usize,
    /// Number of checks that passed.
    pub passed: usize,
    /// Whether the draft was repaired by the graph.
    pub repaired: bool,
    /// Final text length (chars).
    pub text_len: usize,
}

impl QualityScore {
    /// A composite quality score in [0, 1]: pass_rate weighted by whether repair was needed.
    pub fn composite(&self) -> f64 {
        let base = self.pass_rate;
        if self.repaired {
            base * 0.9
        } else {
            base
        }
    }
}

/// Configuration for one experiment run.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// The configuration space this experiment belongs to.
    pub space: ConfigurationSpace,
    /// The concrete configuration to test.
    pub config: Configuration,
    /// Model path for the benchmark.
    pub model_path: String,
    /// Quantization label.
    pub quantization: String,
    /// Prompt to run.
    pub prompt: String,
    /// Decode token budget (0 = production default).
    pub decode_tokens: u32,
    /// Warm repeats.
    pub warm_repeats: u32,
    /// Sampler seed.
    pub seed: u64,
    /// Linked hypothesis ID.
    pub hypothesis_id: Option<String>,
    /// Evaluation mode for this run (defaults to MeasuredLiveBackend).
    pub evaluation_mode: EvaluationMode,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrialEvidence {
    #[serde(default)]
    pub telemetry_before: Option<crate::runner::TelemetrySample>,
    #[serde(default)]
    pub telemetry_after: Option<crate::runner::TelemetrySample>,
    pub schema_version: u16,
    pub model_identity: Option<String>,
    pub prompt_identity: Option<String>,
    pub runtime_identity: Option<String>,
    pub device_identity: Option<String>,
    pub policy_identity: Option<String>,
    pub consumed_seed: Option<u64>,
    pub requested_token_budget: Option<u32>,
    pub actual_token_budget: Option<u32>,
    pub raw_output: Option<String>,
    pub repaired_output: Option<String>,
    pub raw_quality: Option<QualityScore>,
    pub repaired_quality: Option<QualityScore>,
    pub quality_is_token_proxy: bool,
    pub first_token_ms: crate::runner::MetricState,
    pub peak_memory_bytes: crate::runner::MetricState,
    pub energy_j: crate::runner::MetricState,
    pub repair_duration_ms: crate::runner::MetricState,
}
impl ExperimentResult {
    pub fn qualification_gaps(&self) -> Vec<&'static str> {
        let mut gaps = Vec::new();
        let e = &self.evidence;
        if self.evaluation_mode == EvaluationMode::SimulationFixture {
            gaps.push("simulation fixture");
        }
        if self
            .receipt
            .as_ref()
            .map(|r| {
                r.executed_backend
                    == etalon_inference::runtime::receipt::execution::BackendKind::Unknown
            })
            .unwrap_or(true)
        {
            gaps.push("missing actual executor receipt");
        }
        if e.schema_version == 0 {
            gaps.push("legacy incomplete evidence");
        }
        if e.consumed_seed.is_none() {
            gaps.push("seed consumption unavailable");
        }
        if e.raw_quality.is_none() || e.quality_is_token_proxy {
            gaps.push("raw quality unavailable or proxy");
        }
        if e.model_identity.is_none()
            || e.prompt_identity.is_none()
            || e.runtime_identity.is_none()
            || e.device_identity.is_none()
            || e.policy_identity.is_none()
        {
            gaps.push("incomplete trial identity");
        }
        gaps
    }
}

impl ExperimentConfig {
    /// Build a `BenchConfig` from this experiment config.
    pub fn to_bench_config(&self) -> crate::benchmark::BenchConfig {
        crate::benchmark::BenchConfig {
            label: format!("{}_h{:016x}", self.space.name, self.config.hash()),
            model_path: self.model_path.clone(),
            quantization: self.quantization.clone(),
            prompt: self.prompt.clone(),
            decode_tokens: self.decode_tokens,
            warm_repeats: self.warm_repeats,
        }
    }
}

pub trait VerifiedQuality {
    fn quality_score(&self) -> QualityScore;
}
impl QualityScore {
    pub fn from_verified(v: &impl VerifiedQuality) -> Self {
        v.quality_score()
    }
}
