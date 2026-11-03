use super::math::{rms_norm_inplace, rope_inplace};
use super::*;

fn missing<H: Copy, E>(
    tensor: Option<TensorBinding<H>>,
    layer: Option<usize>,
    role: DenseRole,
) -> Result<TensorBinding<H>, DenseError<E>> {
    tensor.ok_or(DenseError::MissingTensor { layer, role })
}

fn gemv<H: Copy, K: TensorKernels<H>>(
    kernels: &mut K,
    tensor: TensorBinding<H>,
    input: &[f32],
    output: &mut [f32],
    layer: Option<usize>,
    role: DenseRole,
) -> Result<usize, DenseError<K::Error>> {
    if tensor.input_width > input.len() || tensor.output_rows > output.len() {
        return Err(DenseError::Kernel { layer, role });
    }
    kernels
        .gemv_into(tensor, input, output)
        .map_err(|source| DenseError::Tensor {
            layer,
            role,
            source,
        })?;
    Ok(tensor.output_rows)
}

fn norm_hidden<H: Copy, K: TensorKernels<H>>(
    kernels: &mut K,
    w: &mut DenseWorkspace<'_>,
    tensor: Option<TensorBinding<H>>,
    layer: usize,
    role: DenseRole,
) -> Result<(), DenseError<K::Error>> {
    w.normed.copy_from_slice(w.hidden);
    let Some(tensor) = tensor else {
        return Ok(());
    };
    let actual = kernels
        .decode_row_into(tensor, 0, w.norm_weight)
        .map_err(|source| DenseError::Tensor {
            layer: Some(layer),
            role,
            source,
        })?;
    if actual < w.hidden.len() {
        return Err(DenseError::DecodeCount {
            layer: Some(layer),
            role,
            expected: w.hidden.len(),
            actual,
        });
    }
    rms_norm_inplace(w.normed, w.norm_weight);
    Ok(())
}

fn forward_layer<H: Copy, K: TensorKernels<H>>(
    plan: &DensePlan<'_, H>,
    kernels: &mut K,
    w: &mut DenseWorkspace<'_>,
    layer: usize,
    position: usize,
) -> Result<(), DenseError<K::Error>> {
    let tensors = plan.model.layers[layer];
    let c = plan.model.config;
    let q = missing(tensors.q, Some(layer), DenseRole::Q)?;
    let k = missing(tensors.k, Some(layer), DenseRole::K)?;
    let v = missing(tensors.v, Some(layer), DenseRole::V)?;
    let output = missing(tensors.output, Some(layer), DenseRole::AttentionOutput)?;
    norm_hidden(
        kernels,
        w,
        tensors.attn_norm,
        layer,
        DenseRole::AttentionNorm,
    )?;
    let qn = gemv(kernels, q, w.normed, w.q, Some(layer), DenseRole::Q)?;
    let kn = gemv(kernels, k, w.normed, w.k, Some(layer), DenseRole::K)?;
    let vn = gemv(kernels, v, w.normed, w.v, Some(layer), DenseRole::V)?;
    if qn != c.heads * c.head_dim || kn != c.kv_heads * c.head_dim || vn != kn {
        return Err(DenseError::Kernel {
            layer: Some(layer),
            role: DenseRole::AttentionDimensions,
        });
    }
    rope_inplace(
        &mut w.q[..qn],
        c.heads,
        c.head_dim,
        position as u32,
        c.rope_base,
        c.rope_scale,
    );
    rope_inplace(
        &mut w.k[..kn],
        c.kv_heads,
        c.head_dim,
        position as u32,
        c.rope_base,
        c.rope_scale,
    );
    super::attention::attention(c, plan.required.kv_plane, layer, position, w)?;
    let on = gemv(
        kernels,
        output,
        w.attention,
        w.projection,
        Some(layer),
        DenseRole::AttentionOutput,
    )?;
    if on < c.embedding {
        return Err(DenseError::Kernel {
            layer: Some(layer),
            role: DenseRole::AttentionOutputDimensions,
        });
    }
    for (hidden, projection) in w.hidden.iter_mut().zip(w.projection.iter()) {
        *hidden += *projection;
    }

    norm_hidden(kernels, w, tensors.ffn_norm, layer, DenseRole::FfnNorm)?;
    let gate = missing(tensors.gate, Some(layer), DenseRole::Gate)?;
    let up = missing(tensors.up, Some(layer), DenseRole::Up)?;
    let down = missing(tensors.down, Some(layer), DenseRole::Down)?;
    let gate_n = gemv(
        kernels,
        gate,
        w.normed,
        w.gate,
        Some(layer),
        DenseRole::Gate,
    )?;
    let up_n = gemv(kernels, up, w.normed, w.up, Some(layer), DenseRole::Up)?;
    if gate_n != c.ffn || up_n != gate_n {
        return Err(DenseError::Kernel {
            layer: Some(layer),
            role: DenseRole::FfnDimensions,
        });
    }
    for (gate, up) in w.gate.iter_mut().zip(w.up.iter()) {
        let g = *gate;
        *gate = (g / (1.0 + (-g).exp())) * *up;
    }
    let dn = gemv(
        kernels,
        down,
        w.gate,
        w.projection,
        Some(layer),
        DenseRole::Down,
    )?;
    if dn < c.embedding {
        return Err(DenseError::Kernel {
            layer: Some(layer),
            role: DenseRole::DownDimensions,
        });
    }
    for (hidden, projection) in w.hidden.iter_mut().zip(w.projection.iter()) {
        *hidden += *projection;
    }
    Ok(())
}

fn forward_hidden<H: Copy, K: TensorKernels<H>>(
    plan: &DensePlan<'_, H>,
    kernels: &mut K,
    w: &mut DenseWorkspace<'_>,
    token: u32,
    position: usize,
) -> Result<(), DenseError<K::Error>> {
    let actual = kernels
        .decode_row_into(plan.model.embedding, token as usize, w.hidden)
        .map_err(|source| DenseError::Tensor {
            layer: None,
            role: DenseRole::Embedding,
            source,
        })?;
    if actual != plan.model.config.embedding {
        return Err(DenseError::DecodeCount {
            layer: None,
            role: DenseRole::Embedding,
            expected: plan.model.config.embedding,
            actual,
        });
    }
    for layer in 0..plan.model.layers.len() {
        forward_layer(plan, kernels, w, layer, position)?;
    }
    Ok(())
}

fn admitted<'a, H: Copy, E>(
    plan: &DensePlan<'_, H>,
    workspace: DenseWorkspace<'a>,
    position: usize,
    byte_budget: usize,
) -> Result<DenseWorkspace<'a>, DenseError<E>> {
    if position >= plan.model.config.max_context {
        return Err(DenseError::ContextExceeded {
            position,
            max_context: plan.model.config.max_context,
        });
    }
    workspace
        .admit(plan.required, byte_budget)
        .map_err(DenseError::Workspace)
}

/// Executes all dense layers and updates KV without a vocabulary projection.
pub fn ingest_token<H: Copy, K: TensorKernels<H>>(
    plan: &DensePlan<'_, H>,
    kernels: &mut K,
    workspace: DenseWorkspace<'_>,
    token: u32,
    position: usize,
    byte_budget: usize,
) -> Result<(), DenseError<K::Error>> {
    let mut w = admitted(plan, workspace, position, byte_budget)?;
    forward_hidden(plan, kernels, &mut w, token, position)
}

/// Executes a complete dense token and finite greedy argmax; failure can leave
/// partial KV/scratch updates. Caller-owned logits remain available for inspection.
pub fn run_token<H: Copy, K: TensorKernels<H>>(
    plan: &DensePlan<'_, H>,
    kernels: &mut K,
    workspace: DenseWorkspace<'_>,
    token: u32,
    position: usize,
    byte_budget: usize,
) -> Result<DenseStep, DenseError<K::Error>> {
    let mut w = admitted(plan, workspace, position, byte_budget)?;
    forward_hidden(plan, kernels, &mut w, token, position)?;
    if let Some(norm) = plan.model.output_norm {
        let actual = kernels
            .decode_row_into(norm, 0, w.norm_weight)
            .map_err(|source| DenseError::Tensor {
                layer: None,
                role: DenseRole::OutputNorm,
                source,
            })?;
        if actual < w.hidden.len() {
            return Err(DenseError::DecodeCount {
                layer: None,
                role: DenseRole::OutputNorm,
                expected: w.hidden.len(),
                actual,
            });
        }
        rms_norm_inplace(w.hidden, w.norm_weight);
    }
    let output = missing(plan.model.output, None, DenseRole::Output)?;
    let vocabulary = gemv(kernels, output, w.hidden, w.logits, None, DenseRole::Output)?;
    let mut token_id = 0u32;
    let mut max_logit = f32::NEG_INFINITY;
    for (token, &score) in w.logits[..vocabulary].iter().enumerate() {
        if score.is_finite()
            && (score > max_logit || (score == max_logit && token < token_id as usize))
        {
            max_logit = score;
            token_id = token as u32;
        }
    }
    if max_logit == f32::NEG_INFINITY {
        return Err(DenseError::Kernel {
            layer: None,
            role: DenseRole::OutputArgmax,
        });
    }
    Ok(DenseStep {
        token_id,
        max_logit,
    })
}

/// Clears only used KV. This neither restores a checkpoint nor resets scratch/RNG.
pub fn reset_kv<H: Copy>(
    plan: &DensePlan<'_, H>,
    kv: &mut [f32],
) -> Result<(), DenseWorkspaceError> {
    if kv.len() < plan.required.kv {
        return Err(DenseWorkspaceError::Capacity {
            buffer: DenseBuffer::Kv,
            required: plan.required.kv,
            available: kv.len(),
        });
    }
    kv[..plan.required.kv].fill(0.0);
    Ok(())
}
