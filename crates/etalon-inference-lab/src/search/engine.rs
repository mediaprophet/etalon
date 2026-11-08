use super::{expected_improvement, KnnSurrogate, SobolSequence};
use crate::config_space::{Configuration, ConfigurationSpace};
use crate::experiment::ExperimentResult;
use std::collections::HashMap;
/// The search engine: combines Sobol initial design with k-NN + EI Bayesian optimization.
pub struct SearchEngine {
    objective: SearchObjective,
    space: ConfigurationSpace,
    sobol: SobolSequence,
    surrogate: KnnSurrogate,
    /// Number of initial Sobol points before switching to EI.
    initial_design: usize,
    /// Observed results: (normalized input, objective value, config hash).
    pub(super) observations: Vec<(Vec<f64>, f64, u64)>,
    /// Best objective value seen so far.
    best_objective: f64,
    /// Budget (max number of trials).
    budget: usize,
    /// Set of already-tried config hashes (dedup).
    pub(super) tried: HashMap<u64, ()>,
    seen: Vec<Configuration>,
}

#[derive(Clone, Copy, Debug)]
pub enum SearchObjective {
    ThroughputTimesQuality,
    /// Exploratory measurement ranking; does not establish quality or promotion.
    MeasuredDecodeThroughput,
}

impl SearchEngine {
    /// Creates the component.
    pub fn new(space: ConfigurationSpace, budget: usize) -> Self {
        Self::with_objective(space, budget, SearchObjective::ThroughputTimesQuality)
    }

    pub fn with_objective(
        space: ConfigurationSpace,
        budget: usize,
        objective: SearchObjective,
    ) -> Self {
        let dim = space.dims();
        let initial = (budget / 5).clamp(5, 20).min(budget);
        Self {
            objective,
            space,
            sobol: SobolSequence::new(dim),
            surrogate: KnnSurrogate::new(5),
            initial_design: initial,
            observations: Vec::new(),
            best_objective: f64::NEG_INFINITY,
            budget,
            tried: HashMap::new(),
            seen: Vec::new(),
        }
    }

    /// Number of trials so far.
    pub fn trials(&self) -> usize {
        self.observations.len()
    }

    /// Best objective value seen.
    pub fn best_value(&self) -> f64 {
        self.best_objective
    }

    /// Ask for the next configuration to try.
    /// Returns `None` if the budget is exhausted.
    pub fn ask(&mut self) -> Option<Configuration> {
        if self.observations.len() >= self.budget {
            return None;
        }

        if self.space.validate().is_err() {
            return None;
        }
        if self.observations.len() < self.initial_design {
            for _ in 0..1024 {
                let cfg = self.space.build_from_normalized(&self.sobol.next());
                if !self.seen.contains(&cfg) {
                    return Some(cfg);
                }
            }
            return None;
        }

        // Phase 2: EI-guided search.
        // Generate a batch of random candidates and pick the one with highest EI.
        let n_candidates = 100;
        let mut best_ei = f64::NEG_INFINITY;
        let mut best_cfg = None;

        for _ in 0..n_candidates {
            let t = self.sobol.next();
            let cfg = self.space.build_from_normalized(&t);
            if self.seen.contains(&cfg) {
                continue;
            }
            let (mu, sigma) = self.surrogate.predict(&t);
            let ei = expected_improvement(mu, sigma, self.best_objective);
            if ei > best_ei {
                best_ei = ei;
                best_cfg = Some(cfg);
            }
        }

        // If all candidates were already tried, fall back to a random one.
        if best_cfg.is_none() {
            for _ in 0..10 {
                let t = self.sobol.next();
                let cfg = self.space.build_from_normalized(&t);
                if !self.seen.contains(&cfg) {
                    return Some(cfg);
                }
            }
            return None;
        }

        best_cfg
    }

    /// Tell the engine the result of an experiment.
    pub fn tell(&mut self, cfg: &Configuration, result: &ExperimentResult) {
        let h = cfg.hash();
        if self.space.validate_config(cfg).is_err() || self.seen.contains(cfg) {
            return;
        }
        self.seen.push(cfg.clone());
        self.tried.insert(h, ());

        // Objective: combine throughput and quality (higher is better).
        // This is a scalarization for the surrogate; the Pareto frontier uses all 6 dims.
        let objective = match self.objective {
            SearchObjective::ThroughputTimesQuality => compute_objective(result),
            SearchObjective::MeasuredDecodeThroughput => {
                if result.error.is_none()
                    && result.evaluation_mode == crate::experiment::EvaluationMode::MeasuredLiveBackend
                    && result.receipt.as_ref().is_some_and(|r| r.executed_backend != etalon_inference::runtime::receipt::execution::BackendKind::Unknown)
                {
                    result.bench.as_ref().map(|b| b.decode_tok_s)
                        .filter(|v| v.is_finite() && *v > 0.0).unwrap_or(f64::NEG_INFINITY)
                } else { f64::NEG_INFINITY }
            }
        };

        let normalized = self.space.normalize_config(cfg);
        if objective.is_finite() {
            self.surrogate.add(normalized.clone(), objective);
        }
        self.observations.push((normalized, objective, h));

        if objective > self.best_objective {
            self.best_objective = objective;
        }
    }

    /// Directly add an observation to the surrogate (for resuming from a log).
    /// The normalized vector and config hash must be pre-computed.
    pub fn surrogate_add(&mut self, normalized: Vec<f64>, objective: f64, config_hash: u64) {
        if self.tried.contains_key(&config_hash) {
            return;
        }
        self.tried.insert(config_hash, ());
        if objective.is_finite() {
            self.surrogate.add(normalized.clone(), objective);
        }
        self.observations.push((normalized, objective, config_hash));
        if objective > self.best_objective {
            self.best_objective = objective;
        }
    }

    /// Get the best configuration found so far (by scalar objective).
    pub fn best_config(&self) -> Option<u64> {
        self.observations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, _, h)| *h)
    }
}

/// Scalar objective for the surrogate model: throughput × quality (higher is better).
/// Failed experiments get -infinity.
fn compute_objective(r: &ExperimentResult) -> f64 {
    if r.error.is_some() {
        return f64::NEG_INFINITY;
    }
    let tok_s = r.bench.as_ref().map(|b| b.decode_tok_s).unwrap_or(0.0);
    let quality = r.quality.composite();
    if tok_s.is_finite() && tok_s >= 0.0 && quality.is_finite() {
        tok_s * quality
    } else {
        f64::NEG_INFINITY
    }
}

// ── Track-and-Stop bandit for A/B testing ─────────────────────────────────────
