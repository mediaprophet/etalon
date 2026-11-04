use super::super::{finite, workspace::take, Error, Host};
use super::{math, TextRope, WorkspaceLayout};
use etalon_model::inference::qwen_attention::{QkNorm, Qwen35AttentionRoles};

/// One causal text token with a separate normalized input and residual hidden.
/// Cache is token-major [position, kv_head, lane], without sliding or padding.
/// Errors preserve hidden, both complete cache buffers and cursor bit-for-bit.
/// Workspace is disposable, including on errors. No allocation or I/O occurs.
#[allow(clippy::too_many_arguments)]
pub fn execute_normalized(
    host: &impl Host,
    roles: &Qwen35AttentionRoles,
    normalized: &[f32],
    hidden: &mut [f32],
    keys: &mut [f32],
    values: &mut [f32],
    cursor: &mut usize,
    rope: TextRope<'_>,
    epsilon: f32,
    workspace: &mut [f32],
) -> Result<(), Error> {
    roles.validate().map_err(Error::Roles)?;
    let g = roles.geometry;
    let kv_width = g.kv_heads * g.head_dim;
    if keys.len() != values.len()
        || keys.len() % kv_width != 0
        || hidden.len() != g.hidden
        || normalized.len() != g.hidden
    {
        return Err(Error::Shape);
    }
    let l = WorkspaceLayout::checked(g, keys.len() / kv_width)?;
    // FP32 positions must remain exact; no implicit interpolation or M-RoPE.
    if *cursor >= l.capacity
        || rope.position != *cursor
        || rope.position > (1usize << 24)
        || rope.inv_freq.len() != g.rotary_dim / 2
    {
        return Err(Error::Shape);
    }
    if workspace.len() < l.floats {
        return Err(Error::Workspace);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::Numerical(
            etalon_inference_kernel::recurrent::QwenDeltaError::InvalidEpsilon,
        ));
    }
    finite(rope.inv_freq)?;
    if rope.inv_freq.iter().any(|&v| v <= 0.0) {
        return Err(Error::Shape);
    }
    for p in [&roles.q_gate, &roles.k, &roles.v, &roles.out] {
        if !host.admits(p) {
            return Err(Error::UnsupportedProjection);
        }
    }
    // Unused cache capacity is opaque and need not have been initialized to finite values.
    for input in [
        normalized,
        &*hidden,
        &keys[..*cursor * l.kv],
        &values[..*cursor * l.kv],
    ] {
        finite(input)?;
    }
    let mut rest = &mut workspace[..l.floats];
    let projected = take(&mut rest, 2 * l.query);
    let q = take(&mut rest, l.query);
    let k = take(&mut rest, l.kv);
    let v = take(&mut rest, l.kv);
    let q_norm = take(&mut rest, g.head_dim);
    let k_norm = take(&mut rest, g.head_dim);
    let attended = take(&mut rest, l.query);
    let output = take(&mut rest, g.hidden);
    let scores = take(&mut rest, l.capacity);
    for (info, dst) in [(&roles.q_norm, &mut *q_norm), (&roles.k_norm, &mut *k_norm)] {
        dst.fill(f32::NAN);
        host.read(info, dst)?;
        finite(dst)?;
    }
    for (p, dst) in [
        (&roles.q_gate, &mut *projected),
        (&roles.k, &mut *k),
        (&roles.v, &mut *v),
    ] {
        dst.fill(f32::NAN);
        host.gemm(p, normalized, dst)?;
        finite(dst)?;
    }
    let raw = roles.norm == QkNorm::RawZeroCentered;
    for h in 0..g.query_heads {
        let head = &mut q[h * g.head_dim..(h + 1) * g.head_dim];
        head.copy_from_slice(&projected[2 * h * g.head_dim..(2 * h + 1) * g.head_dim]);
        math::norm_inplace(head, q_norm, raw, epsilon)?;
    }
    for head in k.chunks_exact_mut(g.head_dim) {
        math::norm_inplace(head, k_norm, raw, epsilon)?;
    }
    math::rotate(q, g.head_dim, rope.inv_freq, rope.position)?;
    math::rotate(k, g.head_dim, rope.inv_freq, rope.position)?;
    math::attention(
        q,
        k,
        v,
        keys,
        values,
        *cursor,
        g.query_heads,
        g.kv_heads,
        g.head_dim,
        scores,
        attended,
    )?;
    for h in 0..g.query_heads {
        for d in 0..g.head_dim {
            let gate = projected[(2 * h + 1) * g.head_dim + d];
            // Stable sigmoid, including finite gates whose negation would overflow exp.
            let sigmoid = if gate >= 0.0 {
                1.0 / (1.0 + (-gate).exp())
            } else {
                let e = gate.exp();
                e / (1.0 + e)
            };
            attended[h * g.head_dim + d] *= sigmoid;
        }
    }
    finite(attended)?;
    output.fill(f32::NAN);
    host.gemm(&roles.out, attended, output)?;
    finite(output)?;
    for (dst, &residual) in output.iter_mut().zip(hidden.iter()) {
        *dst += residual;
    }
    finite(output)?;
    let start = *cursor * l.kv;
    keys[start..start + l.kv].copy_from_slice(k);
    values[start..start + l.kv].copy_from_slice(v);
    hidden.copy_from_slice(output);
    *cursor += 1;
    Ok(())
}
