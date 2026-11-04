use etalon_inference_lab::{
    campaign::{CampaignConfig, try_run_optimization_campaign_with_objective},
    config_space::ParameterValue,
    experiment::{EvaluationMode, ExperimentConfig, ExperimentResult},
    runner::ExperimentExecutor,
    search::{SearchEngine, SearchObjective},
    standalone::{BoundedArtifacts, StandaloneCpuExecutor, StandaloneOptions, dense_cpu_space},
};
use std::{collections::BTreeMap, path::PathBuf, sync::{Arc, atomic::AtomicBool}, time::{Duration, Instant}};

const USAGE: &str = "Required: --mode standalone-developer --profile dense-p64-cpu --workflow benchmark|search|campaign|ablate --model MODEL.p64 --prompt-ids CSV --context N --decode-tokens N --buffer-bytes N --artifact-bytes N --scratch-parent DIRECTORY\nOptional: --warm-repeats N (default 1, max 16) --trials N (default 2, max 64) --timeout-seconds N (default 3600) --retain NEW_DIRECTORY\nDense Llama-compatible P64 + native scalar CPU only. MoE, FTW and Qwen P64 sequence profiles are unsupported; no silent backend fallback. Pretokenized IDs and greedy argmax; no tokenizer or seed consumption. No quality/thermal/energy provider in this CLI. No qualified promotion is performed.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run().map_err(Into::into)
}
fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut flags = BTreeMap::new();
    while let Some(flag) = args.next() {
        if flag == "--help" { println!("{USAGE}"); return Ok(()); }
        if !["--mode", "--profile", "--workflow", "--model", "--prompt-ids", "--context",
            "--decode-tokens", "--buffer-bytes", "--artifact-bytes", "--scratch-parent",
            "--warm-repeats", "--trials", "--timeout-seconds", "--retain"].contains(&flag.as_str()) {
            return Err(format!("unknown flag {flag}\n{USAGE}"));
        }
        let value = args.next().ok_or_else(|| format!("missing value for {flag}"))?;
        if value.len() > 65_536 || flags.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate or oversized argument {flag}"));
        }
    }
    let required = |key: &str| flags.get(key).cloned().ok_or_else(|| format!("{key} required\n{USAGE}"));
    if required("--mode")? != "standalone-developer" { return Err("explicit standalone-developer mode required".into()); }
    if required("--profile")? != "dense-p64-cpu" { return Err("unsupported model/backend profile; only dense-p64-cpu is implemented".into()); }
    let workflow = required("--workflow")?;
    if !["benchmark", "search", "campaign", "ablate"].contains(&workflow.as_str()) { return Err("unsupported workflow".into()); }
    let model = required("--model")?;
    let prompt = required("--prompt-ids")?;
    let context: usize = number(&required("--context")?)?;
    let decode_tokens: u32 = number(&required("--decode-tokens")?)?;
    let buffer_bytes: usize = number(&required("--buffer-bytes")?)?;
    let artifact_bytes: u64 = number(&required("--artifact-bytes")?)?;
    let warm_repeats: u32 = number(flags.get("--warm-repeats").map(String::as_str).unwrap_or("1"))?;
    let trials: usize = number(flags.get("--trials").map(String::as_str).unwrap_or("2"))?;
    let timeout: u64 = number(flags.get("--timeout-seconds").map(String::as_str).unwrap_or("3600"))?;
    if trials == 0 || trials > 64 || decode_tokens == 0 || decode_tokens > 10_000
        || warm_repeats == 0 || warm_repeats > 16 || timeout == 0 || timeout > 86_400 {
        return Err("trial/decode/repeat/timeout bounds exceeded".into());
    }
    let parent = PathBuf::from(required("--scratch-parent")?);
    let artifacts = BoundedArtifacts::new(&parent, artifact_bytes)?;
    let options = StandaloneOptions {
        model: PathBuf::from(&model), context, byte_budget: buffer_bytes,
        max_tensor_visits: 100_000_000_000,
        cancellation: Arc::new(AtomicBool::new(false)),
        deadline: Instant::now().checked_add(Duration::from_secs(timeout)),
    };
    let mut executor = StandaloneCpuExecutor::new(options, artifacts)?;
    let space = dense_cpu_space();
    let template = ExperimentConfig {
        config: space.default_config(), space: space.clone(), model_path: model.clone(),
        quantization: "native-p64".into(), prompt: prompt.clone(), decode_tokens,
        warm_repeats, seed: 0, hypothesis_id: None,
        evaluation_mode: EvaluationMode::MeasuredLiveBackend,
    };
    println!("mode=standalone-developer profile=dense-p64-cpu backend=native-scalar-cpu workflow={workflow}");
    println!("raw output is token IDs; thermal/power/energy/semantic quality unavailable; seed unused");
    let mut failed = false;
    match workflow.as_str() {
        "campaign" => {
            let mut config = CampaignConfig::new(space, model, trials).with_prompt(prompt).with_quality(false);
            config.quantization = "native-p64".into();
            config.decode_tokens = decode_tokens;
            config.warm_repeats = warm_repeats;
            config.max_duration = Some(Duration::from_secs(timeout));
            let report = try_run_optimization_campaign_with_objective(&config, &mut executor,
                SearchObjective::MeasuredDecodeThroughput)?;
            for result in &report.results { display(result); failed |= result.error.is_some(); }
            println!("qualified_frontier={} (missing independent quality/energy evidence prevents promotion)", report.frontier.frontier_size());
            executor.artifacts_mut().write_json("campaign.json", &report)?;
        }
        "search" => {
            let mut search = SearchEngine::with_objective(space, trials, SearchObjective::MeasuredDecodeThroughput);
            for _ in 0..trials {
                if executor.cancelled() { failed = true; break; }
                let Some(config) = search.ask() else { break; };
                let mut trial = template.clone(); trial.config = config;
                let result = executor.execute(&trial, false);
                search.tell(&trial.config, &result);
                display(&result); failed |= result.error.is_some();
            }
            println!("exploratory_throughput_best_config={:?}; independent quality/energy qualification remains open", search.best_config());
        }
        "ablate" => {
            let mut baseline = None;
            for reversible in [true, false] {
                if executor.cancelled() { failed = true; break; }
                let mut trial = template.clone();
                trial.config.values.insert("reversible_decode".into(), ParameterValue::Bool(reversible));
                let result = executor.execute(&trial, false);
                display(&result); failed |= result.error.is_some();
                let tokens = result.evidence.raw_output.clone();
                if reversible { baseline = tokens; }
                else { println!("ablation_output_ids_equal={}", baseline.is_some() && baseline == tokens); }
            }
        }
        _ => { let result = executor.execute(&template, false); display(&result); failed = result.error.is_some(); }
    }
    let artifacts = executor.finish();
    println!("artifact_bytes={}", artifacts.written_bytes());
    if let Some(destination) = flags.get("--retain") {
        let retained = artifacts.retain(&PathBuf::from(destination))?;
        println!("retained_artifacts={}", retained.display());
    } else {
        drop(artifacts);
        println!("temporary artifacts removed; use --retain NEW_DIRECTORY for explicit retention");
    }
    if failed { Err("one or more trials failed or were cancelled; inspect retained trial evidence".into()) } else { Ok(()) }
}
fn number<T: std::str::FromStr>(value: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("invalid numeric argument {value}"))
}
fn display(result: &ExperimentResult) {
    println!("config={:016x} error={:?} warm_decode_tok_s={:?} raw_quality_available={} qualification_gaps={:?}",
        result.config_hash, result.error, result.bench.as_ref().map(|b| b.decode_tok_s),
        result.evidence.raw_quality.is_some(), result.qualification_gaps());
}
