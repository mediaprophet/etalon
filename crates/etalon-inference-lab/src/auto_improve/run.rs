use super::*;
use crate::experiment_log::{append_run_csv, ExperimentRun};
use std::{
    fs::{create_dir_all, File},
    io::Write,
    time::Instant,
};
/// Run the recursive lab program; write lock-in package under `cfg.out_dir`.
pub fn run_auto_improve<M: LabMode>(
    cfg: &AutoImproveConfig,
    host: &mut dyn AutoImproveHost<M>,
) -> Result<LockInPackage<M>, String> {
    if !cfg.model.is_file() {
        return Err(format!("model not found: {}", cfg.model.display()));
    }
    create_dir_all(&cfg.out_dir).map_err(|e| e.to_string())?;
    let csv_path = cfg.out_dir.join("runs.csv");
    let log_path = cfg.out_dir.join("auto_improve.log");
    let mut log = File::create(&log_path).map_err(|e| e.to_string())?;
    let mut log_bytes = 0u64;

    macro_rules! logln {
        ($($t:tt)*) => {{
            let line = format!($($t)*);
            log_bytes=log_bytes.saturating_add(line.len() as u64).saturating_add(1);
            if log_bytes>crate::evidence::DEFAULT_ARTIFACT_BUDGET{return Err("auto-improve log budget exhausted".into());}
            writeln!(log,"{line}").map_err(|e|e.to_string())?;
            log.flush().map_err(|e|e.to_string())?;
            log::info!("lab_auto|{line}");
            eprintln!("lab_auto|{line}");
        }};
    }

    logln!(
        "start model={} tokens={} budget_s={:.0}",
        cfg.model.display(),
        cfg.tokens,
        cfg.max_duration.as_secs_f64()
    );

    let audit = host.instruments()?;
    crate::evidence::atomic_write(
        &cfg.out_dir.join("audit_path.txt"),
        audit.audit.as_bytes(),
        crate::evidence::DEFAULT_ARTIFACT_BUDGET,
    )?;
    crate::evidence::atomic_write(
        &cfg.out_dir.join("device_roof.txt"),
        audit.roof.as_bytes(),
        crate::evidence::DEFAULT_ARTIFACT_BUDGET,
    )?;
    crate::evidence::atomic_write(
        &cfg.out_dir.join("micro_q4k.txt"),
        audit.micro.as_bytes(),
        crate::evidence::DEFAULT_ARTIFACT_BUDGET,
    )?;
    if let Some(m) = cfg.ollama_model.as_ref() {
        logln!("ollama_probe model={m}");
    }
    let ollama_tok_s = cfg
        .ollama_model
        .as_ref()
        .and_then(|m| host.external_comparison(&cfg.ollama_url, m, cfg.tokens.max(16)));
    if let Some(o) = ollama_tok_s {
        logln!("ollama_decode_tok_s={o:.3}");
    } else {
        logln!("ollama_probe skipped or failed");
    }

    let space = default_search_space::<M>();
    logln!("search_space_size={}", space.len());

    let t_start = Instant::now();
    let mut all_trials: Vec<TrialResult<M>> = Vec::new();
    let mut best: Option<TrialResult<M>> = None;
    let mut gens_without_improve = 0u32;
    let mut generation = 0u32;

    let model_id = cfg
        .model
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let passport_key = audit.passport_key.clone();

    // Generation 0: full space (or truncated if huge — still fine)
    let mut queue: Vec<LabConfig<M>> = space;
    while generation < cfg.max_generations && t_start.elapsed() < cfg.max_duration {
        generation += 1;
        logln!(
            "generation={generation} queue={} elapsed_s={:.0}",
            queue.len(),
            t_start.elapsed().as_secs_f64()
        );

        let gen_best_before = best.as_ref().and_then(|b| b.tok_s).unwrap_or(0.0);

        for cfg_point in queue.drain(..) {
            if t_start.elapsed() >= cfg.max_duration {
                logln!("budget_exhausted mid-generation");
                break;
            }
            let mut trial = host.measure(&cfg.model, cfg.tokens, &cfg_point);
            trial.generation = generation;
            logln!(
                "trial gen={generation} label={} tok_s={:?} wall_ms={:.0}",
                trial.config.label,
                trial.tok_s,
                trial.wall_ms
            );

            // CSV
            let mut run = ExperimentRun {
                run_id: ExperimentRun::new_id(),
                utc: ExperimentRun::utc_now(),
                git_sha: String::new(),
                host_passport_key: passport_key.clone(),
                adapter: audit.best_label.clone(),
                backend: trial.config.backend.unwrap_or("auto").to_string(),
                model_id: model_id.clone(),
                model_hash: String::new(),
                layout: if model_id.contains("soa") {
                    "soa".into()
                } else if model_id.contains("f16") {
                    "f16".into()
                } else {
                    "verbatim".into()
                },
                mode: trial.config.mode.as_str().into(),
                profile: "lab-auto".into(),
                toggles_json: trial.config.toggles_json(),
                qualia_decode_tok_s: trial.tok_s,
                ollama_decode_tok_s: ollama_tok_s,
                a_gap: None,
                prefill_tok_s: None,
                phase_ns_json: "{}".into(),
                n_ulp_max: if audit.micro_cuda_ok {
                    Some(audit.micro_max_ulp)
                } else {
                    None
                },
                c_score: None,
                notes: format!("gen={generation}"),
            };
            run.compute_a_gap();
            append_run_csv(&csv_path, &run)?;

            all_trials.push(trial);
            best = aggregate_trials(&all_trials).into_iter().next();
        }
        let ranked = aggregate_trials(&all_trials);

        let gen_best = best.as_ref().and_then(|b| b.tok_s).unwrap_or(0.0);
        if gen_best > gen_best_before * (1.0 + cfg.plateau_rel) {
            gens_without_improve = 0;
        } else {
            gens_without_improve += 1;
        }
        logln!(
            "generation_end best={:.4} plateau_gens={gens_without_improve}",
            gen_best
        );
        if gens_without_improve >= cfg.plateau_gens && generation >= 2 {
            logln!("plateau_stop");
            break;
        }

        // Next queue: re-measure top elites (variance) + flip one bit neighbors of best
        queue.clear();
        for elite in ranked.iter().take(cfg.elite_resample) {
            queue.push(elite.config.clone());
        }
        if let Some(b) = best.as_ref() {
            for neighbor in neighbors(&b.config) {
                if !queue.iter().any(|c| c == &neighbor) {
                    queue.push(neighbor);
                }
            }
        }
        if queue.is_empty() {
            break;
        }
    }

    let a_gap = match (best.as_ref().and_then(|b| b.tok_s), ollama_tok_s) {
        (Some(q), Some(o)) if q > 0.0 => Some(o / q),
        _ => None,
    };

    let methodology =
        build_methodology(&best, ollama_tok_s, a_gap, &audit, &all_trials, generation);
    crate::evidence::atomic_write(
        &cfg.out_dir.join("METHODOLOGY.md"),
        methodology.as_bytes(),
        crate::evidence::DEFAULT_ARTIFACT_BUDGET,
    )?;

    if let Some(ref b) = best {
        bounded_artifact(
            cfg.out_dir.join("BEST_CONFIG.json"),
            format!(
                "{{\n  \"label\": \"{}\",\n  \"tok_s\": {},\n  \"toggles\": {},\n  \"mode\": \"{}\",\n  \"backend\": \"{}\",\n  \"resident\": {},\n  \"coop\": {},\n  \"kv_int8\": {}\n}}\n",
                b.config.label,
                b.tok_s.unwrap_or(0.0),
                b.config.toggles_json(),
                b.config.mode.as_str(),
                b.config.backend.unwrap_or("auto"),
                b.config.resident,
                b.config.coop,
                b.config.kv_int8
            ),
        )?;
        let backend_line = b
            .config
            .backend
            .map(|be| format!("$env:QUALIA_WGPU_BACKEND='{be}'\n"))
            .unwrap_or_default();
        let apply = format!(
            "# Auto-generated by lab auto-improve — apply winning config\n\
             $env:QUALIA_INFERENCE_MODE='{}'\n\
             {backend_line}\
             $env:QUALIA_LLM_RESIDENT_DECODE='{}'\n\
             $env:QUALIA_LLM_COOP_GEMV='{}'\n\
             $env:QUALIA_LLM_KV_INT8='{}'\n\
             # tok_s measured ≈ {:?}\n",
            b.config.mode.as_str(),
            if b.config.resident { "1" } else { "0" },
            if b.config.coop { "1" } else { "0" },
            if b.config.kv_int8 { "1" } else { "0" },
            b.tok_s
        );
        bounded_artifact(cfg.out_dir.join("apply-best.ps1"), apply)?;
    }

    let pkg = LockInPackage {
        best,
        ollama_tok_s,
        a_gap,
        trials: all_trials.len(),
        generations: generation,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
        out_dir: cfg.out_dir.clone(),
        methodology,
    };
    let summary = format_lockin_summary(&pkg);
    bounded_artifact(cfg.out_dir.join("LOCKIN_SUMMARY.txt"), &summary)?;
    logln!("done\n{summary}");
    Ok(pkg)
}

fn bounded_artifact(path: std::path::PathBuf, value: impl AsRef<[u8]>) -> Result<(), String> {
    crate::evidence::atomic_write(
        &path,
        value.as_ref(),
        crate::evidence::DEFAULT_ARTIFACT_BUDGET,
    )
}
