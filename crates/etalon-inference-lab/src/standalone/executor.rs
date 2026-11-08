use super::{model_run, BoundedArtifacts};
use crate::experiment::*;
use crate::runner::{ExperimentExecutor, MetricState, QualityEvaluator, Telemetry};
use etalon_inference::runtime::receipt::execution::{
    BackendKind, ExecutionReceipt, COUNTER_DECODE_STEPS,
};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub struct StandaloneOptions {
    pub model: PathBuf,
    pub context: usize,
    /// Model + metadata + bindings + caller buffers; not process RSS.
    pub byte_budget: usize,
    pub max_tensor_visits: u64,
    pub cancellation: Arc<AtomicBool>,
    pub deadline: Option<Instant>,
}
impl StandaloneOptions {
    pub fn cancelled(&self) -> bool {
        self.cancellation.load(Ordering::Acquire)
            || self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }
}

pub struct StandaloneCpuExecutor {
    options: StandaloneOptions,
    bytes: Vec<u8>,
    model_identity: String,
    runtime_identity: String,
    model_read_ms: f64,
    artifacts: BoundedArtifacts,
    quality: Option<Box<dyn QualityEvaluator>>,
    telemetry: Option<Box<dyn Telemetry>>,
    trial: u64,
}
impl StandaloneCpuExecutor {
    pub fn new(options: StandaloneOptions, artifacts: BoundedArtifacts) -> Result<Self, String> {
        if options.context == 0 || options.context > 1_000_000 || options.cancelled() {
            return Err("invalid context or cancelled".into());
        }
        let started = Instant::now();
        let mut file = std::fs::File::open(&options.model).map_err(|e| e.to_string())?;
        let length = usize::try_from(file.metadata().map_err(|e| e.to_string())?.len())
            .map_err(|e| e.to_string())?;
        if length > options.byte_budget.saturating_sub(131_072) {
            return Err("model exceeds caller buffer byte budget".into());
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|e| e.to_string())?;
        if bytes.capacity() > options.byte_budget.saturating_sub(131_072) {
            return Err("model allocated capacity exceeds budget".into());
        }
        bytes.resize(length, 0);
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let mut extra = [0];
        if file.read(&mut extra).map_err(|e| e.to_string())? != 0 {
            return Err("model changed length during read".into());
        }
        let model_identity = format!("p64:sha256:{:x}", Sha256::digest(&bytes));
        Ok(Self {
            options,
            bytes,
            model_identity,
            runtime_identity: super::runtime_identity(),
            model_read_ms: started.elapsed().as_secs_f64() * 1000.0,
            artifacts,
            quality: None,
            telemetry: None,
            trial: 0,
        })
    }
    pub fn with_quality(mut self, quality: Box<dyn QualityEvaluator>) -> Self {
        self.quality = Some(quality);
        self
    }
    pub fn with_telemetry(mut self, telemetry: Box<dyn Telemetry>) -> Self {
        self.telemetry = Some(telemetry);
        self
    }
    pub fn artifacts_mut(&mut self) -> &mut BoundedArtifacts {
        &mut self.artifacts
    }
    pub fn finish(self) -> BoundedArtifacts {
        self.artifacts
    }

    fn run(
        &mut self,
        config: &ExperimentConfig,
        result: &mut ExperimentResult,
        with_quality: bool,
    ) -> Result<(), String> {
        if self.cancelled() {
            return Err("cancelled".into());
        }
        if config.evaluation_mode != EvaluationMode::MeasuredLiveBackend
            || PathBuf::from(&config.model_path) != self.options.model
            || config.quantization != "native-p64"
        {
            return Err(
                "executor requires matching native P64 model and measured-live configuration"
                    .into(),
            );
        }
        if with_quality && self.quality.is_none() {
            return Err(
                "quality evaluation requested but no independent evaluator was supplied".into(),
            );
        }
        let thermal_before = self.telemetry.as_mut().map(|t| t.sample());
        let measured = model_run::measure(
            &self.bytes,
            self.bytes.capacity(),
            &self.options,
            config,
            &mut self.artifacts,
            self.trial,
        )?;
        let thermal_after = self.telemetry.as_mut().map(|t| t.sample());
        result.evidence.telemetry_before = thermal_before.clone();
        result.evidence.telemetry_after = thermal_after.clone();
        let mut receipt = ExecutionReceipt::new(
            BackendKind::Cpu,
            BackendKind::Cpu,
            self.model_identity.clone(),
            format!(
                "{}:config:{:016x}",
                self.runtime_identity, result.config_hash
            ),
        );
        receipt.tuning_profile = "p64-dense-cpu-v1".into();
        receipt.stop_reason = "requested-token-budget".into();
        receipt.counter_coverage = COUNTER_DECODE_STEPS;
        receipt.counters.decode_steps = measured.decode_steps;
        result.receipt = Some(receipt);
        result.phase = measured.phase;
        result.bench = Some(measured.bench);
        result.evidence.first_token_ms = MetricState::Measured(measured.first_token_ms);
        result.evidence.actual_token_budget = Some(config.decode_tokens);
        result.evidence.raw_output = Some(measured.raw);
        if with_quality {
            let evaluated = self
                .quality
                .as_mut()
                .ok_or("quality evaluator unavailable")?
                .evaluate(
                    &config.prompt,
                    result.evidence.raw_output.as_deref().unwrap_or_default(),
                )?;
            result.quality = evaluated.raw_quality.clone();
            result.evidence.raw_quality = Some(evaluated.raw_quality);
            result.evidence.repaired_quality = evaluated.repaired_quality;
            result.evidence.repaired_output = evaluated.repaired_output;
            result.evidence.repair_duration_ms = evaluated.repair_duration_ms;
        }
        #[derive(serde::Serialize)]
        struct Metadata<'a> {
            model: &'a str,
            runtime: &'a str,
            backend: &'a str,
            mode: &'a str,
            model_read_ms: f64,
            charged_buffer_bytes: usize,
            buffer_byte_budget: usize,
            thermal_before: Option<crate::runner::TelemetrySample>,
            thermal_after: Option<crate::runner::TelemetrySample>,
            energy_j: MetricState,
            quality: &'a Option<QualityScore>,
            sampler: &'a str,
        }
        self.artifacts.write_json(
            &format!("trial-{}-metadata.json", self.trial),
            &Metadata {
                model: &self.model_identity,
                runtime: &self.runtime_identity,
                backend: "native-scalar-cpu",
                mode: "standalone-developer",
                model_read_ms: self.model_read_ms,
                charged_buffer_bytes: measured.charged_bytes,
                buffer_byte_budget: self.options.byte_budget,
                thermal_before,
                thermal_after,
                energy_j: MetricState::Unavailable,
                quality: &result.evidence.raw_quality,
                sampler: "greedy-argmax-no-rng-seed-consumed",
            },
        )?;
        Ok(())
    }
}
impl ExperimentExecutor for StandaloneCpuExecutor {
    fn cancelled(&self) -> bool {
        self.options.cancelled()
    }
    fn execute(&mut self, config: &ExperimentConfig, with_quality: bool) -> ExperimentResult {
        let mut result = ExperimentResult {
            config_hash: config.config.hash(),
            evaluation_mode: EvaluationMode::MeasuredLiveBackend,
            receipt: None,
            hypothesis_id: config.hypothesis_id.clone(),
            bench: None,
            phase: PhaseSnapshotSerde::default(),
            quality: QualityScore::default(),
            thermal: ThermalSnapshot::default(),
            vram_used: 0,
            config_cbor: config.config.to_cbor(),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                .min(u64::MAX as u128) as u64,
            seed: config.seed,
            error: None,
            evidence: TrialEvidence {
                schema_version: 1,
                model_identity: Some(self.model_identity.clone()),
                prompt_identity: Some(format!(
                    "pretokenized-csv:sha256:{:x}",
                    Sha256::digest(config.prompt.as_bytes())
                )),
                runtime_identity: Some(self.runtime_identity.clone()),
                device_identity: Some(format!(
                    "native-scalar-cpu:{}:{}",
                    std::env::consts::ARCH,
                    std::env::consts::OS
                )),
                policy_identity: Some("standalone-developer:no-qualia-authority".into()),
                requested_token_budget: Some(config.decode_tokens),
                ..Default::default()
            },
        };
        if let Err(error) = self.run(config, &mut result, with_quality) {
            result.error = Some(error);
        }
        if let Err(error) = self
            .artifacts
            .write_json(&format!("trial-{}-result.json", self.trial), &result)
        {
            result.error = Some(format!("artifact persistence failed: {error}"));
        }
        self.trial = self.trial.saturating_add(1);
        result
    }
}
