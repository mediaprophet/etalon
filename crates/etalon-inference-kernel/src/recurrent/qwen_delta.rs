//! Explicit Qwen decay-before-prediction recurrence, independent of the legacy API.
// Numerical reference: transformers f2754311602414ed561cdf8ed9d852390f514404,
// modeling_qwen3_5_moe.py::torch_recurrent_gated_delta_rule.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenDeltaError {
    DimensionMismatch,
    SizeOverflow,
    NonFinite,
    InvalidGate,
    InvalidEpsilon,
}

impl std::fmt::Display for QwenDeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Qwen numerical admission failed: {self:?}")
    }
}
impl std::error::Error for QwenDeltaError {}

pub(super) fn finite(values: &[f32]) -> Result<(), QwenDeltaError> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(QwenDeltaError::NonFinite)
    }
}

/// One rectangular head: `state[key_lane * value_dim + value_lane]`.
/// Q/K are already L2-normalized and Q is already scaled by `key_dim^-0.5`.
/// Decay and beta are explicit scalars in [0,1]; decay is `exp(g)`, not `g`.
/// Every slice must have its exact logical length. Any error leaves all buffers unchanged.
#[allow(clippy::too_many_arguments)]
pub fn step_qwen_delta(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    key_dim: usize,
    value_dim: usize,
    out: &mut [f32],
) -> Result<(), QwenDeltaError> {
    let count = key_dim
        .checked_mul(value_dim)
        .ok_or(QwenDeltaError::SizeOverflow)?;
    if key_dim == 0
        || value_dim == 0
        || state.len() != count
        || q.len() != key_dim
        || k.len() != key_dim
        || v.len() != value_dim
        || out.len() != value_dim
    {
        return Err(QwenDeltaError::DimensionMismatch);
    }
    for values in [&*state, q, k, v] {
        finite(values)?;
    }
    finite(&[decay, beta])?;
    if !(0.0..=1.0).contains(&decay) || !(0.0..=1.0).contains(&beta) {
        return Err(QwenDeltaError::InvalidGate);
    }
    // Dry-run all independent columns before committing any of them.
    for (h, &value) in v.iter().enumerate() {
        column(state, q, k, value, decay, beta, value_dim, h)?;
    }
    for (h, &value) in v.iter().enumerate() {
        let (delta, y) = column(state, q, k, value, decay, beta, value_dim, h)?;
        for (s, &key) in k.iter().enumerate() {
            let index = s * value_dim + h;
            state[index] = state[index] * decay + key * delta;
        }
        out[h] = y;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn column(
    state: &[f32],
    q: &[f32],
    k: &[f32],
    value: f32,
    decay: f32,
    beta: f32,
    width: usize,
    h: usize,
) -> Result<(f32, f32), QwenDeltaError> {
    let mut prediction = 0.0;
    for s in 0..k.len() {
        prediction += (state[s * width + h] * decay) * k[s];
    }
    let delta = (value - prediction) * beta;
    finite(&[prediction, delta])?;
    let mut y = 0.0;
    for s in 0..k.len() {
        let next = state[s * width + h] * decay + k[s] * delta;
        finite(&[next])?;
        y += next * q[s];
    }
    finite(&[y])?;
    Ok((delta, y))
}
