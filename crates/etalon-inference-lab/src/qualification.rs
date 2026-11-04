use crate::experiment::{ExperimentConfig, ExperimentResult};
use crate::runner::ExperimentExecutor;
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TechnicalQualification {
    pub baseline_mean_tok_s: f64,
    pub candidate_mean_tok_s: f64,
    pub baseline_trials: Vec<ExperimentResult>,
    pub candidate_trials: Vec<ExperimentResult>,
}
pub fn qualify_candidate(
    baseline: &ExperimentConfig,
    candidate: &ExperimentConfig,
    held_out_prompt: &str,
    pairs: usize,
    min_gain: f64,
    runner: &mut dyn ExperimentExecutor,
) -> Result<TechnicalQualification, String> {
    if pairs < 3 || pairs > 100 || !min_gain.is_finite() || min_gain < 0.0 {
        return Err("invalid confirmation limits".into());
    }
    if held_out_prompt.is_empty() || held_out_prompt == baseline.prompt {
        return Err("independent held-out prompt required".into());
    }
    if baseline.model_path != candidate.model_path
        || baseline.quantization != candidate.quantization
        || baseline.decode_tokens != candidate.decode_tokens
        || baseline.decode_tokens == 0
    {
        return Err("confirmation model, precision or work budget differs".into());
    }
    for name in [
        "governance_profile",
        "intervention_cadence",
        "p64_integrity",
        "output_policy",
        "fidelity",
        "work_limit",
    ] {
        if baseline.config.get(name) != candidate.config.get(name) {
            return Err(format!("protected comparison dimension differs: {name}"));
        }
    }
    let mut baseline_trials = Vec::new();
    let mut candidate_trials = Vec::new();
    for i in 0..pairs {
        if runner.cancelled() {
            return Err("qualification cancelled".into());
        }
        let mut b = baseline.clone();
        let mut c = candidate.clone();
        b.prompt = held_out_prompt.into();
        c.prompt = held_out_prompt.into();
        b.seed = i as u64;
        c.seed = i as u64;
        let (br, cr) = if i % 2 == 0 {
            (runner.execute(&b, true), runner.execute(&c, true))
        } else {
            let cr = runner.execute(&c, true);
            (runner.execute(&b, true), cr)
        };
        for r in [&br, &cr] {
            let gaps = r.qualification_gaps();
            if !gaps.is_empty() {
                return Err(format!(
                    "confirmation evidence incomplete: {}",
                    gaps.join(", ")
                ));
            }
            if r.error.is_some()
                || r.evidence.consumed_seed != Some(i as u64)
                || r.evidence.actual_token_budget != Some(baseline.decode_tokens)
            {
                return Err("confirmation failed or seed/work budget not consumed".into());
            }
        }
        let be = &br.evidence;
        let ce = &cr.evidence;
        if be.model_identity != ce.model_identity
            || be.prompt_identity != ce.prompt_identity
            || be.runtime_identity != ce.runtime_identity
            || be.device_identity != ce.device_identity
            || be.policy_identity != ce.policy_identity
        {
            return Err("paired confirmation identities differ".into());
        }
        baseline_trials.push(br);
        candidate_trials.push(cr);
    }
    let mean = |rows: &[ExperimentResult]| -> Result<f64, String> {
        let mut total = 0.0;
        for r in rows {
            let t = r
                .bench
                .as_ref()
                .ok_or("confirmation benchmark absent")?
                .decode_tok_s;
            if !t.is_finite() || t <= 0.0 {
                return Err("invalid confirmation throughput".into());
            }
            total += t;
        }
        Ok(total / rows.len() as f64)
    };
    let baseline_mean_tok_s = mean(&baseline_trials)?;
    let candidate_mean_tok_s = mean(&candidate_trials)?;
    if candidate_mean_tok_s < baseline_mean_tok_s * (1.0 + min_gain) {
        return Err("independent confirmation did not meet frozen improvement threshold".into());
    }
    Ok(TechnicalQualification {
        baseline_mean_tok_s,
        candidate_mean_tok_s,
        baseline_trials,
        candidate_trials,
    })
}
