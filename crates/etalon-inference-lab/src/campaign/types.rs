//! End-to-end optimization campaign orchestration.
//!
//! Ties together the five lab layers into a single pipeline:
//! 1. `SearchEngine` proposes configurations (Sobol + Bayesian EI)
//! 2. `run_experiment_with_quality()` executes each trial
//! 3. `ParetoFrontier` computes the 6-dimensional non-dominated set
//! 4. `BeliefGraph` records verdicts and cascades confidence
//! 5. Results are persisted to JSONL for reproducibility
//!
//! The campaign runs in batches: after each batch, the frontier is recomputed
//! and the belief graph is updated. The search engine adapts based on all
//! observations so far.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config_space::ConfigurationSpace;
use crate::experiment::ExperimentResult;
use crate::hypothesis::BeliefGraph;
use crate::pareto::{ApplicationProfileWeight, ParetoFrontier};

/// Configuration for an optimization campaign.
#[derive(Debug, Clone)]
pub struct CampaignConfig {
    /// The configuration space to search over.
    pub space: ConfigurationSpace,
    /// Model path for benchmarking.
    pub model_path: String,
    /// Quantization label.
    pub quantization: String,
    /// Prompt for quality evaluation.
    pub prompt: String,
    /// Decode token budget (0 = production default).
    pub decode_tokens: u32,
    /// Warm repeats for benchmark.
    pub warm_repeats: u32,
    /// Total evaluation budget (number of trials).
    pub budget: usize,
    /// Batch size: after each batch, recompute frontier + update beliefs.
    pub batch_size: usize,
    /// Application profile for best-config selection.
    pub profile: ApplicationProfileWeight,
    /// Improvement threshold for verdict evaluation (e.g. 0.20 = 20%).
    pub improvement_threshold: f64,
    /// Optional path to persist JSONL experiment log.
    pub jsonl_path: Option<PathBuf>,
    /// Optional path to persist belief graph JSON.
    pub belief_path: Option<PathBuf>,
    /// Use quality verification (decode_with_metrics + verify_and_heal_turn).
    /// If false, uses run_bench only (faster but no quality score).
    pub with_quality: bool,
    /// Max wall-clock duration for the campaign.
    pub max_duration: Option<Duration>,
}

impl CampaignConfig {
    /// Create a campaign config with sensible defaults.
    pub fn new(space: ConfigurationSpace, model_path: impl Into<String>, budget: usize) -> Self {
        Self {
            space,
            model_path: model_path.into(),
            quantization: "q4_k".into(),
            prompt: "What is the capital of France?".into(),
            decode_tokens: 32,
            warm_repeats: 3,
            budget,
            batch_size: 5,
            profile: ApplicationProfileWeight::Interactive,
            improvement_threshold: 0.20,
            jsonl_path: None,
            belief_path: None,
            with_quality: true,
            max_duration: None,
        }
    }

    /// Builder: sets the campaign prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Builder: sets the application-profile weighting.
    pub fn with_profile(mut self, profile: ApplicationProfileWeight) -> Self {
        self.profile = profile;
        self
    }

    /// Builder: toggles quality scoring.
    pub fn with_quality(mut self, quality: bool) -> Self {
        self.with_quality = quality;
        self
    }

    /// Builder: writes results to a JSONL log at `path`.
    pub fn with_jsonl_log(mut self, path: impl Into<PathBuf>) -> Self {
        self.jsonl_path = Some(path.into());
        self
    }

    /// Builder: writes belief traces to `path`.
    pub fn with_belief_log(mut self, path: impl Into<PathBuf>) -> Self {
        self.belief_path = Some(path.into());
        self
    }

    /// Builder: sets the batch size.
    pub fn with_batch_size(mut self, batch: usize) -> Self {
        self.batch_size = batch.max(1);
        self
    }

    /// Builder: caps campaign duration.
    pub fn with_max_duration(mut self, dur: Duration) -> Self {
        self.max_duration = Some(dur);
        self
    }
}

/// The result of an optimization campaign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignReport {
    /// All experiment results from the campaign.
    pub results: Vec<ExperimentResult>,
    /// The Pareto frontier (indices into `results`).
    pub frontier: ParetoFrontier,
    /// The belief graph after the campaign.
    pub beliefs: BeliefGraph,
    /// Index of the best result (selected by profile).
    pub best_index: Option<usize>,
    /// Number of trials actually run (may be less than budget if time-limited).
    pub trials_run: usize,
    /// Wall-clock duration of the campaign in seconds.
    pub elapsed_s: f64,
    /// Baseline throughput (tok/s) for comparison.
    pub baseline_tok_s: Option<f64>,
    /// Best throughput achieved (tok/s).
    pub best_tok_s: Option<f64>,
    /// Summary message for logging.
    pub summary: String,
}

impl CampaignReport {
    /// The best experiment result (selected by application profile).
    pub fn best_result(&self) -> Option<&ExperimentResult> {
        self.best_index.and_then(|i| self.results.get(i))
    }

    /// Improvement ratio: best / baseline.
    pub fn improvement_ratio(&self) -> Option<f64> {
        match (self.best_tok_s, self.baseline_tok_s) {
            (Some(best), Some(baseline)) if baseline > 0.0 => Some(best / baseline),
            _ => None,
        }
    }

    /// Serialize the report to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}
