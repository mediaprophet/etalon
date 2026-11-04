use super::{finite, workspace::take, Error, Host, WorkspaceLayout};
use etalon_inference_kernel::recurrent::{
    gated_rms_norm, normalize_qwen_qk, qwen_gates, silu_into, step_qwen_delta,
    unpack_qwen_projections, QwenHeadLayout, QwenProjectionPacking,
};
use etalon_model::inference::qwen_roles::{
    Projection, Qwen35Packing, Qwen35Projections, Qwen35Roles,
};

fn admit(host: &impl Host, p: &Projection, input: usize, output: usize) -> Result<(), Error> {
    if p.weight.n_dims != 2 || p.cols() != input || p.rows() != output {
        return Err(Error::Shape);
    }
    if !host.admits(p) {
        return Err(Error::UnsupportedProjection);
    }
    Ok(())
}

/// One token, linear attention plus residual. `normalized` is already normalized by the
/// caller; `hidden` supplies the separate residual and receives the committed result.
/// Packing is the explicit role profile, never inferred from local tensor names.
/// Errors preserve hidden, live history and live recurrent state bit-for-bit. Workspace
/// is disposable on error. This is an f32 CPU numerical path, not GPU qualification.
#[allow(clippy::too_many_arguments)]
pub fn execute_normalized(
    host: &impl Host,
    roles: &Qwen35Roles,
    normalized: &[f32],
    hidden: &mut [f32],
    history: &mut [f32],
    state: &mut [f32],
    epsilon: f32,
    workspace: &mut [f32],
) -> Result<(), Error> {
    roles.validate().map_err(Error::Roles)?;
    let g = roles.geometry;
    let l = WorkspaceLayout::checked(g)?;
    if normalized.len() != g.hidden
        || hidden.len() != g.hidden
        || history.len() != l.history
        || state.len() != l.recurrent
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
    match &roles.projections {
        Qwen35Projections::Separate { qkv, z, a, b } => {
            admit(host, qkv, g.hidden, l.channels)?;
            admit(host, z, g.hidden, l.value)?;
            admit(host, a, g.hidden, g.value_heads)?;
            admit(host, b, g.hidden, g.value_heads)?;
        }
        Qwen35Projections::Fused { qkvz, ba, packing } => {
            if *packing == Qwen35Packing::Separate {
                return Err(Error::Shape);
            }
            admit(host, qkvz, g.hidden, l.projection)?;
            admit(host, ba, g.hidden, 2 * g.value_heads)?;
        }
    }
    admit(host, &roles.out, l.value, g.hidden)?;
    for values in [normalized, &*hidden, &*history, &*state] {
        finite(values)?;
    }
    let mut rest = &mut workspace[..l.floats];
    let next_history = take(&mut rest, l.history);
    let next_state = take(&mut rest, l.recurrent);
    let conv_weights = take(&mut rest, l.conv_weights);
    let a_log = take(&mut rest, g.value_heads);
    let dt_bias = take(&mut rest, g.value_heads);
    let norm = take(&mut rest, g.value_dim);
    let projected = take(&mut rest, l.projection);
    let ba = take(&mut rest, 2 * g.value_heads);
    let qkv = take(&mut rest, l.channels);
    let z = take(&mut rest, l.value);
    let b = take(&mut rest, g.value_heads);
    let a = take(&mut rest, g.value_heads);
    let convolved = take(&mut rest, l.channels);
    let q = take(&mut rest, l.expanded_key);
    let k = take(&mut rest, l.expanded_key);
    let decay = take(&mut rest, g.value_heads);
    let beta = take(&mut rest, g.value_heads);
    let y = take(&mut rest, l.value);
    let gated = take(&mut rest, l.value);
    let output = take(&mut rest, g.hidden);
    next_history.copy_from_slice(history);
    next_state.copy_from_slice(state);
    // Poison every provider destination so incomplete successful writes fail admission.
    for (info, dst) in [
        (&roles.conv, &mut *conv_weights),
        (&roles.a_log, &mut *a_log),
        (&roles.dt_bias, &mut *dt_bias),
        (&roles.norm, &mut *norm),
    ] {
        dst.fill(f32::NAN);
        host.read(info, dst)?;
        finite(dst)?;
    }
    projected.fill(f32::NAN);
    ba.fill(f32::NAN);
    let packing = match &roles.projections {
        Qwen35Projections::Separate { qkv, z, a, b } => {
            host.gemm(qkv, normalized, &mut projected[..l.channels])?;
            host.gemm(z, normalized, &mut projected[l.channels..])?;
            host.gemm(b, normalized, &mut ba[..g.value_heads])?;
            host.gemm(a, normalized, &mut ba[g.value_heads..])?;
            QwenProjectionPacking::GlobalQkvzBa
        }
        Qwen35Projections::Fused {
            qkvz,
            ba: ba_role,
            packing,
        } => {
            host.gemm(qkvz, normalized, projected)?;
            host.gemm(ba_role, normalized, ba)?;
            match packing {
                Qwen35Packing::FusedGlobal => QwenProjectionPacking::GlobalQkvzBa,
                Qwen35Packing::FusedKeyHeadGrouped => QwenProjectionPacking::GroupedByKeyHead,
                Qwen35Packing::Separate => return Err(Error::Shape),
            }
        }
    };
    let heads = QwenHeadLayout {
        key_heads: g.key_heads,
        value_heads: g.value_heads,
        key_dim: g.key_dim,
        value_dim: g.value_dim,
    };
    unpack_qwen_projections(heads, packing, projected, ba, qkv, z, b, a)?;
    for c in 0..l.channels {
        let start = c * (g.kernel - 1);
        let weights = &conv_weights[c * g.kernel..(c + 1) * g.kernel];
        let past = &history[start..start + g.kernel - 1];
        let mut sum = 0.0;
        for j in 0..g.kernel - 1 {
            sum += past[j] * weights[j];
        }
        sum += qkv[c] * weights[g.kernel - 1];
        convolved[c] = sum;
        if g.kernel > 1 {
            next_history[start..start + g.kernel - 2].copy_from_slice(&past[1..]);
            next_history[start + g.kernel - 2] = qkv[c];
        }
    }
    silu_into(convolved, qkv)?;
    normalize_qwen_qk(heads, qkv, q, k)?;
    qwen_gates(a, b, a_log, dt_bias, decay, beta)?;
    let head_state = g.key_dim * g.value_dim;
    for h in 0..g.value_heads {
        let keys = h * g.key_dim..(h + 1) * g.key_dim;
        let values = h * g.value_dim..(h + 1) * g.value_dim;
        step_qwen_delta(
            &mut next_state[h * head_state..(h + 1) * head_state],
            &q[keys.clone()],
            &k[keys],
            &qkv[2 * g.qk_width + values.start..2 * g.qk_width + values.end],
            decay[h],
            beta[h],
            g.key_dim,
            g.value_dim,
            &mut y[values.clone()],
        )?;
        gated_rms_norm(
            &y[values.clone()],
            &z[values.clone()],
            norm,
            epsilon,
            &mut gated[values],
        )?;
    }
    output.fill(f32::NAN);
    host.gemm(&roles.out, gated, output)?;
    finite(output)?;
    for (value, residual) in output.iter_mut().zip(hidden.iter()) {
        *value += residual;
    }
    finite(output)?;
    history.copy_from_slice(next_history);
    state.copy_from_slice(next_state);
    hidden.copy_from_slice(output);
    Ok(())
}

/// Pinned HF ordinary RMSNorm representation: stored weight has a +1 offset.
/// Use only for explicitly qualified zero-centered weights, never infer from names.
pub fn zero_centered_rms_norm(
    input: &[f32],
    stored_weight: &[f32],
    epsilon: f32,
    out: &mut [f32],
) -> Result<(), Error> {
    ordinary_rms_norm(input, stored_weight, true, epsilon, out)
}

/// Ordinary RMSNorm with an explicit checkpoint weight convention.
pub fn ordinary_rms_norm(
    input: &[f32],
    stored_weight: &[f32],
    zero_centered: bool,
    epsilon: f32,
    out: &mut [f32],
) -> Result<(), Error> {
    if input.is_empty() || input.len() != stored_weight.len() || out.len() != input.len() {
        return Err(Error::Shape);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::Numerical(
            etalon_inference_kernel::recurrent::QwenDeltaError::InvalidEpsilon,
        ));
    }
    finite(input)?;
    finite(stored_weight)?;
    let sum: f32 = input.iter().map(|x| x * x).sum();
    finite(&[sum])?;
    let denominator = sum / input.len() as f32 + epsilon;
    finite(&[denominator])?;
    let scale = 1.0 / denominator.sqrt();
    finite(&[scale])?;
    for i in 0..input.len() {
        let weight = if zero_centered {
            1.0 + stored_weight[i]
        } else {
            stored_weight[i]
        };
        finite(&[input[i] * scale * weight])?;
    }
    for i in 0..input.len() {
        let weight = if zero_centered {
            1.0 + stored_weight[i]
        } else {
            stored_weight[i]
        };
        out[i] = input[i] * scale * weight;
    }
    Ok(())
}
