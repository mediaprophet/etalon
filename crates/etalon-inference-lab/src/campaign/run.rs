use super::*;
use crate::config_space::Configuration;
use crate::experiment::{append_experiment_jsonl, load_experiment_log, EvaluationMode};
use crate::experiment::{ExperimentConfig, ExperimentResult};
use crate::hypothesis::BeliefGraph;
use crate::hypothesis::{evaluate_verdict, Hypothesis};
use crate::pareto::ParetoFrontier;
use crate::search::SearchEngine;
use std::time::Instant;
/// Run an optimization campaign end-to-end.
///
/// This is the main entry point for the AI Inference Optimization Lab.
/// It runs `budget` trials, computing the Pareto frontier and updating the
/// belief graph after each batch.
pub fn try_run_optimization_campaign(
    cfg: &CampaignConfig,
    runner: &mut dyn crate::runner::ExperimentExecutor,
) -> Result<CampaignReport, String> {
    try_run_optimization_campaign_with_objective(cfg, runner, crate::search::SearchObjective::ThroughputTimesQuality)
}

pub fn try_run_optimization_campaign_with_objective(
    cfg: &CampaignConfig,
    runner: &mut dyn crate::runner::ExperimentExecutor,
    objective: crate::search::SearchObjective,
) -> Result<CampaignReport, String> {
    cfg.space.validate()?;
    if cfg.batch_size == 0 {
        return Err("campaign batch size cannot be zero".into());
    }
    let t_start = Instant::now();
    let mut engine = SearchEngine::with_objective(cfg.space.clone(), cfg.budget, objective);
    let mut beliefs = BeliefGraph::new();
    let mut results: Vec<ExperimentResult> = Vec::new();

    if let Some(ref jsonl) = cfg.jsonl_path {
        let existing = load_experiment_log(jsonl)?;
        for r in &existing {
            let config = Configuration::from_cbor(&r.config_cbor)?;
            cfg.space.validate_config(&config)?;
            let expected = ExperimentConfig {
                space: cfg.space.clone(),
                config: config.clone(),
                model_path: cfg.model_path.clone(),
                quantization: cfg.quantization.clone(),
                prompt: cfg.prompt.clone(),
                decode_tokens: cfg.decode_tokens,
                warm_repeats: cfg.warm_repeats,
                seed: r.seed,
                hypothesis_id: r.hypothesis_id.clone(),
                evaluation_mode: EvaluationMode::MeasuredLiveBackend,
            };
            if !runner.resume_compatible(&expected, r) {
                return Err("resume history identity is unverified or incompatible".into());
            }
            engine.tell(&config, r);
        }
        results = existing;
    }

    // Run a baseline measurement if we have no results yet.
    let baseline_tok_s = if results.is_empty() {
        let baseline_cfg = cfg.space.default_config();
        let exp_cfg = ExperimentConfig {
            space: cfg.space.clone(),
            config: baseline_cfg,
            model_path: cfg.model_path.clone(),
            quantization: cfg.quantization.clone(),
            prompt: cfg.prompt.clone(),
            decode_tokens: cfg.decode_tokens,
            warm_repeats: cfg.warm_repeats,
            seed: 0,
            hypothesis_id: None,
            evaluation_mode: EvaluationMode::MeasuredLiveBackend,
        };
        let baseline_result = if cfg.with_quality {
            runner.execute(&exp_cfg, true)
        } else {
            runner.execute(&exp_cfg, false)
        };
        let bt = baseline_result.bench.as_ref().map(|b| b.decode_tok_s);
        results.push(baseline_result);
        bt
    } else {
        results[0].bench.as_ref().map(|b| b.decode_tok_s)
    };

    // Main search loop.
    let mut trials_run = 0;
    while trials_run < cfg.budget {
        if runner.cancelled() {
            break;
        }
        // Check time budget.
        if let Some(max_dur) = cfg.max_duration {
            if t_start.elapsed() >= max_dur {
                log::info!(
                    "campaign|time_budget_exhausted|{}s",
                    t_start.elapsed().as_secs_f64()
                );
                break;
            }
        }

        // Ask the search engine for the next configuration.
        let config = match engine.ask() {
            Some(c) => c,
            None => break,
        };

        // Build the experiment config.
        let exp_cfg = ExperimentConfig {
            space: cfg.space.clone(),
            config: config.clone(),
            model_path: cfg.model_path.clone(),
            quantization: cfg.quantization.clone(),
            prompt: cfg.prompt.clone(),
            decode_tokens: cfg.decode_tokens,
            warm_repeats: cfg.warm_repeats,
            seed: trials_run as u64,
            hypothesis_id: None,
            evaluation_mode: EvaluationMode::MeasuredLiveBackend,
        };

        // Run the experiment.
        let result = if cfg.with_quality {
            runner.execute(&exp_cfg, true)
        } else {
            runner.execute(&exp_cfg, false)
        };

        // Tell the search engine the outcome.
        engine.tell(&config, &result);

        // Persist to JSONL.
        if let Some(ref jsonl) = cfg.jsonl_path {
            append_experiment_jsonl(jsonl, &result)?;
        }

        results.push(result);
        trials_run += 1;

        // After each batch: recompute frontier + update beliefs.
        if trials_run % cfg.batch_size == 0 {
            update_beliefs(
                &mut beliefs,
                &results,
                baseline_tok_s,
                cfg.improvement_threshold,
            );
            log::info!(
                "campaign|batch_complete|trials={}|frontier={}|beliefs={}",
                trials_run,
                ParetoFrontier::compute(&results).frontier_size(),
                beliefs.hypotheses.len(),
            );
        }
    }

    // Final frontier computation.
    let frontier = ParetoFrontier::compute(&results);

    // Final belief update.
    update_beliefs(
        &mut beliefs,
        &results,
        baseline_tok_s,
        cfg.improvement_threshold,
    );

    // Select best result by application profile.
    let best_index = cfg
        .profile
        .select_best(&results, &frontier)
        .and_then(|r| results.iter().position(|x| x.config_hash == r.config_hash));

    let best_tok_s = best_index
        .and_then(|i| results[i].bench.as_ref())
        .map(|b| b.decode_tok_s);

    // Persist belief graph.
    if let Some(ref belief_path) = cfg.belief_path {
        beliefs.save(belief_path)?;
    }

    let elapsed_s = t_start.elapsed().as_secs_f64();

    // Build summary.
    let improvement = match (best_tok_s, baseline_tok_s) {
        (Some(best), Some(base)) if base > 0.0 => {
            format!("{:.1}% improvement", (best / base - 1.0) * 100.0)
        }
        _ => "no baseline".to_string(),
    };
    let summary = format!(
        "trials={}, frontier={}, beliefs={}, {}",
        trials_run,
        frontier.frontier_size(),
        beliefs.hypotheses.len(),
        improvement,
    );

    Ok(CampaignReport {
        results,
        frontier,
        beliefs,
        best_index,
        trials_run,
        elapsed_s,
        baseline_tok_s,
        best_tok_s,
        summary,
    })
}

/// Update the belief graph from experiment results.
/// For each result, evaluate the verdict against the baseline and record it.
fn update_beliefs(
    beliefs: &mut BeliefGraph,
    results: &[ExperimentResult],
    baseline_tok_s: Option<f64>,
    threshold: f64,
) {
    // Create a default hypothesis if none exist.
    if beliefs.hypotheses.is_empty() {
        let h = Hypothesis::new(
            "H-default",
            "Search finds configurations that improve decode throughput",
            "campaign",
        );
        beliefs.add_hypothesis(h);
    }

    let baseline = match baseline_tok_s {
        Some(b) if b > 0.0 => b,
        _ => return,
    };

    // Evaluate each result (skip the baseline, which is results[0]).
    for (i, r) in results.iter().enumerate().skip(1) {
        if r.error.is_some() {
            continue;
        }
        let tok_s = match r.bench.as_ref() {
            Some(b) => b.decode_tok_s,
            None => continue,
        };
        if tok_s <= 0.0 {
            continue;
        }

        // Check if we already recorded this experiment.
        let exp_id = format!("E-campaign-{i}");
        if beliefs.experiments.contains_key(&exp_id) {
            continue;
        }

        let h = beliefs.hypotheses.get("H-default").unwrap();
        let verdict = evaluate_verdict(h, tok_s, baseline, threshold);
        let weight = 0.5 + (tok_s / baseline).min(2.0) * 0.25; // weight scales with improvement
        beliefs.record_experiment(exp_id, "H-default", r.config_hash, verdict, weight);
    }
}

pub fn run_optimization_campaign(
    cfg: &CampaignConfig,
    runner: &mut dyn crate::runner::ExperimentExecutor,
) -> CampaignReport {
    match try_run_optimization_campaign(cfg, runner) {
        Ok(report) => report,
        Err(error) => CampaignReport {
            results: Vec::new(),
            frontier: ParetoFrontier::default(),
            beliefs: BeliefGraph::new(),
            best_index: None,
            trials_run: 0,
            elapsed_s: 0.0,
            baseline_tok_s: None,
            best_tok_s: None,
            summary: format!("campaign failed; evidence incomplete: {error}"),
        },
    }
}
