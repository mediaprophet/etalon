use super::{GgmlCpuKernels, PackedTensor};
use crate::runtime::dense_transaction::propose_dense;
use etalon_inference_kernel::dense_cpu::{
    self as dense, DensePlan, DenseRequirements, DenseWorkspace,
};

pub fn workspace(mut buffer: &mut [f32], r: DenseRequirements) -> DenseWorkspace<'_> {
    let mut take = |length| {
        let (part, rest) = std::mem::take(&mut buffer).split_at_mut(length);
        buffer = rest;
        part
    };
    DenseWorkspace {
        kv: take(r.kv),
        hidden: take(r.hidden),
        normed: take(r.normed),
        norm_weight: take(r.norm_weight),
        q: take(r.q),
        k: take(r.k),
        v: take(r.v),
        attention: take(r.attention),
        projection: take(r.projection),
        gate: take(r.gate),
        up: take(r.up),
        scores: take(r.scores),
        logits: take(r.logits),
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct TokenMeasurement {
    pub token: u32,
    pub max_logit_bits: u32,
    pub elapsed_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct SequenceMeasurement {
    pub prefill_ns: u64,
    pub decode_ns: u64,
    pub first_token_ns: u64,
}

/// Developer-only greedy sequence. Cancellation is checked before every token.
/// `undo=None` measures direct execution; `Some` uses the reversible proposal path.
pub fn run_sequence(
    plan: &DensePlan<'_, PackedTensor<'_>>,
    state: &mut [f32],
    mut undo: Option<&mut [f32]>,
    prompt: &[u32],
    output: &mut [TokenMeasurement],
    cancelled: &dyn Fn() -> bool,
) -> Result<SequenceMeasurement, &'static str> {
    let r = plan.requirements();
    if prompt.is_empty()
        || output.is_empty()
        || state.len() < r.bytes / 4
        || undo.as_ref().is_some_and(|s| s.len() < r.bytes / 4)
    {
        return Err("invalid sequence buffers");
    }
    let mut kernels = GgmlCpuKernels;
    let start = std::time::Instant::now();
    for (position, &token) in prompt[..prompt.len() - 1].iter().enumerate() {
        if cancelled() {
            return Err("cancelled");
        }
        dense::ingest_token(
            plan,
            &mut kernels,
            workspace(state, r),
            token,
            position,
            r.bytes,
        )
        .map_err(|_| "prefill failed")?;
    }
    let prefill_ns = nanos(start.elapsed());
    let decode_start = std::time::Instant::now();
    let mut first_token_ns = 0;
    let mut token = prompt[prompt.len() - 1];
    for (step, sample) in output.iter_mut().enumerate() {
        if cancelled() {
            return Err("cancelled");
        }
        let step_start = std::time::Instant::now();
        let position = prompt.len() - 1 + step;
        let (next, logit) = if let Some(snapshot) = undo.as_deref_mut() {
            let pending = propose_dense(
                plan,
                &mut kernels,
                workspace(state, r),
                snapshot,
                token,
                position,
                r.bytes.checked_mul(2).ok_or("budget overflow")?,
            )
            .map_err(|_| "reversible decode failed")?;
            let accepted = pending.commit();
            (accepted.token_id, accepted.max_logit)
        } else {
            let accepted = dense::run_token(
                plan,
                &mut kernels,
                workspace(state, r),
                token,
                position,
                r.bytes,
            )
            .map_err(|_| "direct decode failed")?;
            (accepted.token_id, accepted.max_logit)
        };
        token = next;
        *sample = TokenMeasurement {
            token,
            max_logit_bits: logit.to_bits(),
            elapsed_ns: nanos(step_start.elapsed()),
        };
        if step == 0 {
            first_token_ns = nanos(start.elapsed());
        }
    }
    Ok(SequenceMeasurement {
        prefill_ns,
        decode_ns: nanos(decode_start.elapsed()),
        first_token_ns,
    })
}

fn nanos(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
