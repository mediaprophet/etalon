//! Caller-buffered Qwen projection ordering and per-head numerical transforms.
// Reference: transformers f2754311602414ed561cdf8ed9d852390f514404 (Qwen3.5 MoE);
// grouped ordering: 08810b1e278938278c50153ee1edfd7a20a759da (Qwen3Next).
// These layouts describe admitted input contracts, not local converter provenance.

use super::qwen_delta::{finite, QwenDeltaError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenHeadLayout {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenProjectionPacking {
    /// Per key head: Q[K],K[K],V[R*V],Z[R*V]; BA: B[R],A[R], R=Hv/Hk.
    GroupedByKeyHead,
    /// Qkvz: all Q,all K,all V,all Z; BA: all B,all A (each in head-major order).
    GlobalQkvzBa,
}

impl QwenHeadLayout {
    fn extents(self) -> Result<(usize, usize, usize, usize), QwenDeltaError> {
        if self.key_heads == 0
            || self.value_heads == 0
            || self.key_dim == 0
            || self.value_dim == 0
            || !self.value_heads.is_multiple_of(self.key_heads)
        {
            return Err(QwenDeltaError::DimensionMismatch);
        }
        let key = self
            .key_heads
            .checked_mul(self.key_dim)
            .ok_or(QwenDeltaError::SizeOverflow)?;
        let value = self
            .value_heads
            .checked_mul(self.value_dim)
            .ok_or(QwenDeltaError::SizeOverflow)?;
        let qkv = key
            .checked_mul(2)
            .and_then(|x| x.checked_add(value))
            .ok_or(QwenDeltaError::SizeOverflow)?;
        let qkvz = qkv.checked_add(value).ok_or(QwenDeltaError::SizeOverflow)?;
        self.value_heads
            .checked_mul(2)
            .ok_or(QwenDeltaError::SizeOverflow)?;
        self.value_heads
            .checked_mul(self.key_dim)
            .ok_or(QwenDeltaError::SizeOverflow)?;
        Ok((key, value, qkv, qkvz))
    }
}

/// Unpack one token into global Q/K/V channels and head-major Z/B/A.
/// Exact extents, finite inputs, and packing are required before any write.
#[allow(clippy::too_many_arguments)]
pub fn unpack_qwen_projections(
    layout: QwenHeadLayout,
    packing: QwenProjectionPacking,
    qkvz: &[f32],
    ba: &[f32],
    qkv: &mut [f32],
    z: &mut [f32],
    b: &mut [f32],
    a: &mut [f32],
) -> Result<(), QwenDeltaError> {
    let (key, value, qkv_len, qkvz_len) = layout.extents()?;
    if qkvz.len() != qkvz_len
        || ba.len() != 2 * layout.value_heads
        || qkv.len() != qkv_len
        || z.len() != value
        || b.len() != layout.value_heads
        || a.len() != layout.value_heads
    {
        return Err(QwenDeltaError::DimensionMismatch);
    }
    finite(qkvz)?;
    finite(ba)?;
    match packing {
        QwenProjectionPacking::GlobalQkvzBa => {
            qkv.copy_from_slice(&qkvz[..qkv_len]);
            z.copy_from_slice(&qkvz[qkv_len..]);
            b.copy_from_slice(&ba[..layout.value_heads]);
            a.copy_from_slice(&ba[layout.value_heads..]);
        }
        QwenProjectionPacking::GroupedByKeyHead => {
            let ratio = layout.value_heads / layout.key_heads;
            let group_value = ratio * layout.value_dim;
            let group = 2 * layout.key_dim + 2 * group_value;
            for h in 0..layout.key_heads {
                let src = h * group;
                let dst = h * layout.key_dim;
                qkv[dst..dst + layout.key_dim].copy_from_slice(&qkvz[src..src + layout.key_dim]);
                qkv[key + dst..key + dst + layout.key_dim]
                    .copy_from_slice(&qkvz[src + layout.key_dim..src + 2 * layout.key_dim]);
                let src_v = src + 2 * layout.key_dim;
                let dst_v = h * group_value;
                qkv[2 * key + dst_v..2 * key + dst_v + group_value]
                    .copy_from_slice(&qkvz[src_v..src_v + group_value]);
                z[dst_v..dst_v + group_value]
                    .copy_from_slice(&qkvz[src_v + group_value..src_v + 2 * group_value]);
                b[h * ratio..(h + 1) * ratio]
                    .copy_from_slice(&ba[h * 2 * ratio..h * 2 * ratio + ratio]);
                a[h * ratio..(h + 1) * ratio]
                    .copy_from_slice(&ba[h * 2 * ratio + ratio..(h + 1) * 2 * ratio]);
            }
        }
    }
    Ok(())
}

/// L2 normalize global Q/K with sqrt(sum(x*x)+1e-6), repeat each key head
/// Hv/Hk consecutive times, and scale Q by K^-0.5. Outputs have [Hv,K] shape.
/// Every error leaves both output buffers unchanged.
pub fn normalize_qwen_qk(
    layout: QwenHeadLayout,
    qkv: &[f32],
    q: &mut [f32],
    k: &mut [f32],
) -> Result<(), QwenDeltaError> {
    let (key, _, qkv_len, _) = layout.extents()?;
    let expanded = layout.value_heads * layout.key_dim;
    if qkv.len() != qkv_len || q.len() != expanded || k.len() != expanded {
        return Err(QwenDeltaError::DimensionMismatch);
    }
    finite(qkv)?;
    for h in 0..layout.key_heads {
        norm_scale(&qkv[h * layout.key_dim..(h + 1) * layout.key_dim], 1.0e-6)?;
        norm_scale(
            &qkv[key + h * layout.key_dim..key + (h + 1) * layout.key_dim],
            1.0e-6,
        )?;
    }
    let ratio = layout.value_heads / layout.key_heads;
    let query_scale = 1.0 / (layout.key_dim as f32).sqrt();
    for h in 0..layout.value_heads {
        let src = (h / ratio) * layout.key_dim;
        let qs = norm_scale(&qkv[src..src + layout.key_dim], 1.0e-6)?;
        let ks = norm_scale(&qkv[key + src..key + src + layout.key_dim], 1.0e-6)?;
        for i in 0..layout.key_dim {
            q[h * layout.key_dim + i] = (qkv[src + i] * qs) * query_scale;
            k[h * layout.key_dim + i] = qkv[key + src + i] * ks;
        }
    }
    Ok(())
}

fn norm_scale(values: &[f32], epsilon: f32) -> Result<f32, QwenDeltaError> {
    let sum: f32 = values.iter().map(|x| x * x).sum();
    finite(&[sum])?;
    Ok(1.0 / (sum + epsilon).sqrt())
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn gates(a: f32, b: f32, a_log: f32, dt: f32) -> Result<(f32, f32), QwenDeltaError> {
    let x = a + dt;
    let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
    let rate = a_log.exp();
    let g = -rate * softplus;
    finite(&[x, softplus, rate, g])?;
    Ok((g.exp(), sigmoid(b)))
}

/// Produce explicit per-value-head decay=exp(-exp(A_log)*softplus(a+dt_bias))
/// and beta=sigmoid(b). Stable gates; any admission/arithmetic error is atomic.
pub fn qwen_gates(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    decay: &mut [f32],
    beta: &mut [f32],
) -> Result<(), QwenDeltaError> {
    let count = a.len();
    if count == 0
        || [b.len(), a_log.len(), dt_bias.len(), decay.len(), beta.len()]
            .iter()
            .any(|&n| n != count)
    {
        return Err(QwenDeltaError::DimensionMismatch);
    }
    for values in [a, b, a_log, dt_bias] {
        finite(values)?;
    }
    for h in 0..count {
        gates(a[h], b[h], a_log[h], dt_bias[h])?;
    }
    for h in 0..count {
        (decay[h], beta[h]) = gates(a[h], b[h], a_log[h], dt_bias[h])?;
    }
    Ok(())
}

/// Stable SiLU over one token's channels; errors leave output unchanged.
pub fn silu_into(input: &[f32], out: &mut [f32]) -> Result<(), QwenDeltaError> {
    if input.len() != out.len() {
        return Err(QwenDeltaError::DimensionMismatch);
    }
    finite(input)?;
    for (dst, &x) in out.iter_mut().zip(input) {
        *dst = x * sigmoid(x);
    }
    Ok(())
}

/// One head: RMS(input)*stored_weight*SiLU(z), with no zero-centered weight offset.
/// Exact nonempty slices and positive finite epsilon; every error is atomic.
pub fn gated_rms_norm(
    input: &[f32],
    z: &[f32],
    weight: &[f32],
    epsilon: f32,
    out: &mut [f32],
) -> Result<(), QwenDeltaError> {
    let n = input.len();
    if n == 0 || z.len() != n || weight.len() != n || out.len() != n {
        return Err(QwenDeltaError::DimensionMismatch);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(QwenDeltaError::InvalidEpsilon);
    }
    for values in [input, z, weight] {
        finite(values)?;
    }
    let sum: f32 = input.iter().map(|x| x * x).sum();
    let denom = sum / n as f32 + epsilon;
    finite(&[sum, denom])?;
    let scale = 1.0 / denom.sqrt();
    for i in 0..n {
        finite(&[(input[i] * scale * weight[i]) * (z[i] * sigmoid(z[i]))])?;
    }
    for i in 0..n {
        out[i] = (input[i] * scale * weight[i]) * (z[i] * sigmoid(z[i]));
    }
    Ok(())
}
