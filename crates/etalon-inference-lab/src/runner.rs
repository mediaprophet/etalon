use crate::experiment::{ExperimentConfig, ExperimentResult, QualityScore};

pub trait ExperimentExecutor {
    fn cancelled(&self) -> bool {
        false
    }
    fn resume_compatible(&self, _config: &ExperimentConfig, _result: &ExperimentResult) -> bool {
        false
    }
    fn execute(&mut self, config: &ExperimentConfig, with_quality: bool) -> ExperimentResult;
}
impl<F: FnMut(&ExperimentConfig, bool) -> ExperimentResult> ExperimentExecutor for F {
    fn execute(&mut self, config: &ExperimentConfig, quality: bool) -> ExperimentResult {
        self(config, quality)
    }
}
pub trait Cancellation {
    fn cancelled(&self) -> bool;
}
impl Cancellation for std::sync::atomic::AtomicBool {
    fn cancelled(&self) -> bool {
        self.load(std::sync::atomic::Ordering::Acquire)
    }
}
pub struct NeverCancel;
impl Cancellation for NeverCancel {
    fn cancelled(&self) -> bool {
        false
    }
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum MetricState {
    Measured(f64),
    Estimated(f64),
    Unavailable,
}
impl Default for MetricState {
    fn default() -> Self {
        Self::Unavailable
    }
}
pub trait Telemetry {
    fn sample(&mut self) -> TelemetrySample;
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TelemetrySample {
    pub temperature_c: MetricState,
    pub power_w: MetricState,
    pub peak_memory_bytes: MetricState,
}
pub trait QualityEvaluator {
    fn evaluate(&mut self, prompt: &str, raw_output: &str) -> Result<QualityEvaluation, String>;
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityEvaluation {
    pub raw_quality: QualityScore,
    pub repaired_quality: Option<QualityScore>,
    pub raw_output: String,
    pub repaired_output: Option<String>,
    pub repair_duration_ms: MetricState,
}
pub trait ArtifactSink {
    fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), String>;
}
