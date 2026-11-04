use super::super::nvfp4::{nvfp4_gemv_zero_heap, Nvfp4Error};

/// SiLU (Swish) activation function: x * sigmoid(x).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Description of one expert's raw NVFP4 matrices.
pub struct ExpertWeightView<'a> {
    /// Gate + Up packed matrix (NVFP4, 2 nibbles per byte)
    pub gate_up_packed: &'a [u8],
    /// Per-16 FP8 block scales
    pub gate_up_scale: &'a [u8],
    pub gate_up_global: f32,
    /// Down packed matrix (NVFP4)
    pub down_packed: &'a [u8],
    pub down_scale: &'a [u8],
    pub down_global: f32,
    pub intermediate_dim: usize,
    pub emb_dim: usize,
}

/// Zero-heap evaluation of one SwiGLU expert:
/// y = (SiLU(x * W_gate) * (x * W_up)) * W_down
pub fn evaluate_swiglu_expert_nvfp4(
    input: &[f32],
    weights: &ExpertWeightView<'_>,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    swiglu_buf: &mut [f32],
    out: &mut [f32],
) -> Result<(), Nvfp4Error> {
    let emb_dim = weights.emb_dim;
    let inter_dim = weights.intermediate_dim;

    if input.len() < emb_dim || out.len() < emb_dim {
        return Err(Nvfp4Error::BufferTooSmall);
    }
    if gate_buf.len() < inter_dim || up_buf.len() < inter_dim || swiglu_buf.len() < inter_dim {
        return Err(Nvfp4Error::BufferTooSmall);
    }

    // In packed gate_up, gate is rows 0..inter_dim, up is rows inter_dim..2*inter_dim
    let bytes_per_row = emb_dim / 2;
    let blocks_per_row = emb_dim / super::super::nvfp4::NVFP4_BLOCK_SIZE;

    let gate_packed = &weights.gate_up_packed[..inter_dim * bytes_per_row];
    let gate_scales = &weights.gate_up_scale[..inter_dim * blocks_per_row];
    nvfp4_gemv_zero_heap(
        input,
        gate_packed,
        gate_scales,
        weights.gate_up_global,
        inter_dim,
        emb_dim,
        &mut gate_buf[..inter_dim],
    )?;

    let up_packed =
        &weights.gate_up_packed[inter_dim * bytes_per_row..2 * inter_dim * bytes_per_row];
    let up_scales =
        &weights.gate_up_scale[inter_dim * blocks_per_row..2 * inter_dim * blocks_per_row];
    nvfp4_gemv_zero_heap(
        input,
        up_packed,
        up_scales,
        weights.gate_up_global,
        inter_dim,
        emb_dim,
        &mut up_buf[..inter_dim],
    )?;

    // SwiGLU activation: silu(gate) * up
    for i in 0..inter_dim {
        swiglu_buf[i] = silu(gate_buf[i]) * up_buf[i];
    }

    // Down projection: swiglu * W_down -> out (emb_dim)
    nvfp4_gemv_zero_heap(
        &swiglu_buf[..inter_dim],
        weights.down_packed,
        weights.down_scale,
        weights.down_global,
        emb_dim,
        inter_dim,
        &mut out[..emb_dim],
    )?;

    Ok(())
}
