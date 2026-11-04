use super::{BoundedArtifacts, StandaloneOptions};
use crate::experiment::{BenchResultSerde, ExperimentConfig, PhaseSnapshotSerde};
use etalon_inference::cpu_model::{empty_layer, prepare_dense_p64_into, BindingLimits};
use etalon_inference::cpu_model::sequence::{run_sequence, TokenMeasurement};
use etalon_inference_kernel::dense_cpu::{self as dense};
use etalon_model::{inference::{ModelPreparationBudget, PreparedP64, TensorInfo}, p64::IntegrityMode};
use std::{fmt::Write, time::Instant};

pub(super) struct Measured {
    pub bench: BenchResultSerde,
    pub phase: PhaseSnapshotSerde,
    pub raw: String,
    pub first_token_ms: f64,
    pub charged_bytes: usize,
    pub decode_steps: u64,
}

pub(super) fn measure(
    bytes: &[u8], source_capacity: usize, options: &StandaloneOptions, config: &ExperimentConfig,
    artifacts: &mut BoundedArtifacts, trial: u64,
) -> Result<Measured, String> {
    if config.decode_tokens == 0 || config.decode_tokens > 10_000
        || config.warm_repeats == 0 || config.warm_repeats > 16
        || config.prompt.is_empty() || config.prompt.len() > 65_536 {
        return Err("prompt/decode/repeat limit exceeded (decode<=10000, warm repeats 1..16)".into());
    }
    super::dense_cpu_space().validate_config(&config.config)?;
    if config.space.name != "p64-dense-cpu-v1" || config.space.params.len() != 1
        || !matches!(config.space.params.get("reversible_decode"), Some(crate::config_space::ParameterDef::Bool)) {
        return Err("unsupported CPU configuration space".into());
    }
    let reversible = config.config.get_bool("reversible_decode").ok_or("reversible_decode required")?;
    let steps = config.decode_tokens as usize;
    let count = config.prompt.split(',').count();
    if count.checked_add(steps - 1).is_none_or(|n| n > options.context) {
        return Err("prompt plus output exceeds caller context".into());
    }
    let mut charged = source_capacity.checked_add(131_072)
        .and_then(|n| n.checked_add(count * 4))
        .and_then(|n| n.checked_add(steps * (std::mem::size_of::<TokenMeasurement>() + 11)))
        .ok_or("buffer budget overflow")?;
    admit(charged, options.byte_budget)?;
    let mut prompt = reserve(count, 0u32)?;
    for (slot, token) in prompt.iter_mut().zip(config.prompt.split(',')) {
        *slot = token.parse().map_err(|_| "prompt must contain u32 token IDs in CSV form")?;
    }
    let load = Instant::now();
    let prepared: PreparedP64<'_> = PreparedP64::parse(bytes, IntegrityMode::Full,
        ModelPreparationBudget { metadata_bytes: options.byte_budget - charged, tensor_count: 1_000_000 }, None)
        .map_err(|e| e.to_string())?;
    charged = charged.checked_add(prepared.index().entries.capacity()
        .checked_mul(std::mem::size_of::<(u64, TensorInfo)>()).ok_or("index overflow")?)
        .ok_or("index budget overflow")?;
    admit(charged, options.byte_budget)?;
    if prompt.iter().any(|&id| id as usize >= prepared.index().vocab_dim()) {
        return Err("prompt token outside actual model vocabulary".into());
    }
    let layer_count = prepared.index().hyperparams.n_layer as usize;
    let binding_bytes = layer_count.checked_mul(std::mem::size_of_val(&empty_layer())).ok_or("binding overflow")?;
    charged = charged.checked_add(binding_bytes).ok_or("binding overflow")?;
    admit(charged, options.byte_budget)?;
    let mut layers = reserve(layer_count, empty_layer())?;
    let layers_capacity = layers.capacity();
    let model = prepare_dense_p64_into(&prepared, options.context, &mut layers,
        BindingLimits { byte_budget: binding_bytes, max_tensor_visits: options.max_tensor_visits })
        .map_err(|e| format!("unsupported or invalid native P64 dense profile: {e}"))?;
    let divisor = if reversible { 2 } else { 1 };
    let plan = dense::DensePlan::new(model, (options.byte_budget - charged) / divisor)
        .map_err(|e| format!("plan: {e:?}"))?;
    let r = plan.requirements();
    charged = charged.checked_add(r.bytes.checked_mul(divisor).ok_or("workspace overflow")?)
        .ok_or("workspace overflow")?;
    admit(charged, options.byte_budget)?;
    let mut state = reserve(r.bytes / 4, 0f32)?;
    let mut undo = reserve(if reversible { r.bytes / 4 } else { 0 }, 0f32)?;
    let mut samples = reserve(steps, TokenMeasurement::default())?;
    // Charge actual capacities rather than only requested lengths.
    charged = charged.checked_add((prompt.capacity() - count) * 4)
        .and_then(|n| n.checked_add((layers_capacity - layer_count) * std::mem::size_of_val(&empty_layer())))
        .and_then(|n| n.checked_add((state.capacity() - state.len() + undo.capacity() - undo.len()) * 4))
        .and_then(|n| n.checked_add((samples.capacity() - steps) * std::mem::size_of::<TokenMeasurement>()))
        .ok_or("capacity budget overflow")?;
    admit(charged, options.byte_budget)?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.0;
    let mut bench = BenchResultSerde { label: format!("p64-cpu-{:016x}", config.config.hash()),
        quantization: "native-p64-mixed-tensor-codecs".into(), prompt_tokens: count as u64,
        output_tokens: steps as u64, load_ms, mapped_bytes: bytes.len() as u64,
        kv_cache_bytes: (r.kv * 4) as u64, ..Default::default() };
    let mut phase = PhaseSnapshotSerde::default();
    let mut first_token_ms = 0.0;
    let mut raw = String::new();
    raw.try_reserve_exact(steps * 11).map_err(|e| e.to_string())?;
    charged = charged.checked_add(raw.capacity().saturating_sub(steps * 11))
        .ok_or("raw output capacity overflow")?;
    admit(charged, options.byte_budget)?;
    for repeat in 0..=config.warm_repeats {
        state.fill(0.0);
        let measurement = run_sequence(&plan, &mut state,
            if reversible { Some(undo.as_mut_slice()) } else { None }, &prompt, &mut samples,
            &|| options.cancelled()).map_err(str::to_owned)?;
        artifacts.write_json(&format!("trial-{trial}-repeat-{repeat}.json"), &RawRepeat {
            config_hash: config.config.hash(), repeat, timing: measurement,
            tokens: &samples, charged_buffer_bytes: charged,
            thermal: crate::runner::MetricState::Unavailable,
            power: crate::runner::MetricState::Unavailable,
            quality: None,
        })?;
        let total = (measurement.prefill_ns as f64 + measurement.decode_ns as f64) / 1e6;
        let ttft = measurement.first_token_ns as f64 / 1e6;
        if repeat == 0 {
            bench.cold_total_ms = total; bench.cold_ttft_ms = ttft; first_token_ms = ttft;
        } else {
            bench.warm_total_ms += total; bench.warm_ttft_ms += ttft;
            bench.prefill_ms += measurement.prefill_ns as f64 / 1e6;
            bench.decode_ms += measurement.decode_ns as f64 / 1e6;
            phase.prefill_ns = phase.prefill_ns.saturating_add(measurement.prefill_ns);
            phase.decode_ns = phase.decode_ns.saturating_add(measurement.decode_ns);
        }
    }
    for (i, sample) in samples.iter().enumerate() {
        if i != 0 { raw.push(','); }
        write!(&mut raw, "{}", sample.token).map_err(|e| e.to_string())?;
    }
    let repeats = config.warm_repeats as f64;
    bench.warm_total_ms /= repeats; bench.warm_ttft_ms /= repeats;
    bench.prefill_ms /= repeats; bench.decode_ms /= repeats;
    bench.decode_tok_s = steps as f64 * 1000.0 / bench.decode_ms;
    bench.prefill_tok_s = if count > 1 { (count - 1) as f64 * 1000.0 / bench.prefill_ms } else { 0.0 };
    phase.prefill_tokens = (count - 1) as u64 * config.warm_repeats as u64;
    phase.decode_tokens = steps as u64 * config.warm_repeats as u64;
    phase.decode_forward_ns = phase.decode_ns;
    phase.load_ns = (load_ms * 1e6) as u64;
    Ok(Measured { bench, phase, raw, first_token_ms, charged_bytes: charged,
        decode_steps: steps as u64 * (config.warm_repeats as u64 + 1) })
}
#[derive(serde::Serialize)]
struct RawRepeat<'a> {
    config_hash: u64, repeat: u32,
    timing: etalon_inference::cpu_model::sequence::SequenceMeasurement,
    tokens: &'a [TokenMeasurement], charged_buffer_bytes: usize,
    thermal: crate::runner::MetricState, power: crate::runner::MetricState,
    quality: Option<crate::experiment::QualityScore>,
}
fn admit(used: usize, budget: usize) -> Result<(), String> {
    if used > budget { Err("caller buffer byte budget exhausted".into()) } else { Ok(()) }
}
fn reserve<T: Clone>(count: usize, value: T) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|e| e.to_string())?;
    out.resize(count, value);
    Ok(out)
}
