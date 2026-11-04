#[cfg(feature = "cpu-inference")]
mod runner {

    use etalon_inference::cpu_model::{
        empty_layer, prepare_dense_p64_into, BindingLimits, GgmlCpuKernels,
    };
    use etalon_inference::runtime::dense_transaction::propose_dense;
    use etalon_inference::cpu_model::sequence::workspace;
    use etalon_inference_kernel::dense_cpu::{self as dense};
    use etalon_model::{
        inference::{ModelPreparationBudget, PreparedP64, TensorInfo},
        p64::IntegrityMode,
    };
    use sha2::{Digest, Sha256};
    use std::io::Read;

    fn report(buffer: &[f32], stage: &str, position: usize) {
        let mut digest = Sha256::new();
        for value in buffer {
            digest.update(value.to_bits().to_le_bytes());
        }
        println!(
            "workspace stage={stage} position={position} compared_f32={} sha256={:x}",
            buffer.len(),
            digest.finalize()
        );
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args().skip(1);
        if args.next().as_deref() != Some("--mode")
            || args.next().as_deref() != Some("standalone-developer")
        {
            return Err(
            "required: --mode standalone-developer MODEL TOKEN_IDS_CSV CONTEXT STEPS BYTE_BUDGET"
                .into(),
        );
        }
        let path = args.next().ok_or("model path required")?;
        let csv = args.next().ok_or("pretokenized prompt CSV required")?;
        if csv.is_empty() || csv.len() > 65_536 {
            return Err("prompt CSV exceeds byte limit".into());
        }
        let context: usize = args.next().ok_or("context required")?.parse()?;
        let steps: usize = args.next().ok_or("steps required")?.parse()?;
        let budget: usize = args.next().ok_or("byte budget required")?.parse()?;
        if args.next().is_some() || context == 0 || context > 4096 || steps == 0 || steps > 4096 {
            return Err("invalid arguments or context/step limit".into());
        }
        let count = csv.split(',').count();
        if count.checked_add(steps - 1).is_none_or(|n| n > context) {
            return Err("prompt and decode exceed context".into());
        }
        let prompt_bytes = count.checked_mul(4).ok_or("prompt size overflow")?;
        // Argument storage and fixed kernel stack are charged before model allocation.
        let fixed_bytes = prompt_bytes
            .checked_add(csv.capacity())
            .and_then(|n| n.checked_add(path.capacity()))
            .and_then(|n| n.checked_add(40_960))
            .ok_or("argument budget overflow")?;
        if fixed_bytes > budget {
            return Err("argument budget exceeded".into());
        }
        let mut prompt = Vec::new();
        prompt.try_reserve_exact(count)?;
        for token in csv.split(',') {
            prompt.push(token.parse::<u32>()?);
        }
        let fixed_bytes = fixed_bytes - prompt_bytes + prompt.capacity() * 4;
        let mut file = std::fs::File::open(path.as_str())?;
        let length = usize::try_from(file.metadata()?.len())?;
        if length > budget.saturating_sub(fixed_bytes) {
            return Err("model exceeds byte budget".into());
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length)?;
        if bytes.capacity() > budget.saturating_sub(fixed_bytes) {
            return Err("allocated model capacity exceeds budget".into());
        }
        bytes.resize(length, 0);
        file.read_exact(&mut bytes)?;
        let mut extra = [0u8; 1];
        if file.read(&mut extra)? != 0 {
            return Err("model grew while reading".into());
        }
        let config: PreparedP64<'_> = PreparedP64::parse(
            &bytes,
            IntegrityMode::Full,
            ModelPreparationBudget {
                metadata_bytes: budget.saturating_sub(fixed_bytes + bytes.capacity()),
                tensor_count: 1_000_000,
            },
            None,
        )?;
        let metadata_bytes = config
            .index()
            .entries
            .capacity()
            .checked_mul(std::mem::size_of::<(u64, TensorInfo)>())
            .and_then(|n| n.checked_add(std::mem::size_of_val(&config)))
            .ok_or("metadata size overflow")?;
        let layer_count = config.index().hyperparams.n_layer as usize;
        let vocabulary = config.index().vocab_dim();
        if prompt.iter().any(|&t| t as usize >= vocabulary) {
            return Err("prompt token outside vocabulary".into());
        }
        let binding_bytes = layer_count
            .checked_mul(std::mem::size_of_val(&empty_layer()))
            .ok_or("binding size overflow")?;
        let mut used = fixed_bytes
            .checked_add(bytes.capacity())
            .and_then(|n| n.checked_add(metadata_bytes))
            .and_then(|n| n.checked_add(binding_bytes))
            .ok_or("budget overflow")?;
        if used > budget {
            return Err("binding byte budget exceeded".into());
        }
        let mut layers = Vec::new();
        layers.try_reserve_exact(layer_count)?;
        layers.resize(layer_count, empty_layer());
        used = used - binding_bytes + layers.capacity() * std::mem::size_of_val(&empty_layer());
        if used > budget {
            return Err("allocated binding capacity exceeds budget".into());
        }
        let model = prepare_dense_p64_into(
            &config,
            context,
            &mut layers,
            BindingLimits {
                byte_budget: binding_bytes,
                max_tensor_visits: 100_000_000_000,
            },
        )?;
        let plan = dense::DensePlan::new(model, (budget - used) / 2)?;
        let r = plan.requirements();
        let mut state = Vec::new();
        let mut undo = Vec::new();
        state.try_reserve_exact(r.bytes / 4)?;
        used = used
            .checked_add(state.capacity() * 4)
            .ok_or("workspace budget overflow")?;
        if used > budget {
            return Err("allocated workspace capacity exceeds budget".into());
        }
        undo.try_reserve_exact(r.bytes / 4)?;
        used = used
            .checked_add(undo.capacity() * 4)
            .ok_or("undo budget overflow")?;
        if used > budget {
            return Err("allocated undo capacity exceeds budget".into());
        }
        state.resize(r.bytes / 4, 0.0);
        undo.resize(r.bytes / 4, 0.0);
        println!("mode=standalone-developer backend=native-cpu container=p64 governance=none pretokenized=true");
        println!("model_sha256={:x} model_bytes={} charged_buffer_and_argument_bytes={used} byte_budget={budget} kernel_stack_bytes=40960", Sha256::digest(&bytes), bytes.len());
        for (name, source) in [
            ("native.rs", include_str!("../src/cpu_model/native.rs")),
            ("bindings.rs", include_str!("../src/cpu_model/bindings.rs")),
            ("kernels.rs", include_str!("../src/cpu_model/kernels.rs")),
            (
                "dense_p64_developer.rs",
                include_str!("dense_p64_developer.rs"),
            ),
        ] {
            println!(
                "source={name} sha256={:x}",
                Sha256::digest(source.as_bytes())
            );
        }
        let mut kernels = GgmlCpuKernels;
        for (position, &token) in prompt[..count - 1].iter().enumerate() {
            dense::ingest_token(
                &plan,
                &mut kernels,
                workspace(&mut state, r),
                token,
                position,
                r.bytes,
            )?;
            report(&state, "prefill", position);
        }
        let mut token = prompt[count - 1];
        for step in 0..steps {
            let position = count - 1 + step;
            let pending = propose_dense(
                &plan,
                &mut kernels,
                workspace(&mut state, r),
                &mut undo,
                token,
                position,
                r.bytes * 2,
            )
            .map_err(|e| format!("numerical proposal failed: {e:?}"))?;
            let accepted = pending.commit();
            token = accepted.token_id;
            println!(
                "developer_step={step} token={token} max_logit_bits={:08x}",
                accepted.max_logit.to_bits()
            );
            report(&state, "decode", position);
        }
        dense::reset_kv(&plan, &mut state[..r.kv])?;
        report(&state, "reset", 0);
        println!("prompt_tokens_consumed={count} generated_tokens={steps}");
        Ok(())
    }
}
#[cfg(feature = "cpu-inference")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}
#[cfg(not(feature = "cpu-inference"))]
fn main() {
    eprintln!("enable cpu-inference for the native P64 developer runner");
}
