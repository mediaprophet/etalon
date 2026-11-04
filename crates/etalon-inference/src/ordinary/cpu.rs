use super::RMS_NORM_EPS;
use etalon_model::inference::TensorInfo as GgufTensorInfo;
#[inline]
/// In-place ReLU.
pub fn relu_inplace(buf: &mut [f32], n: usize) {
    for v in buf.iter_mut().take(n) {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// SiLU (Swish): x * sigmoid(x) = x / (1 + e^{-x}). Llama/SmolLM2 SwiGLU gate activation.
#[inline]
pub fn silu_inplace(x: &mut [f32], n: usize) {
    for v in x.iter_mut().take(n) {
        *v = *v / (1.0 + (-*v).exp());
    }
}

#[inline]
/// In-place residual add.
pub fn add_residual_inplace(dst: &mut [f32], src: &[f32], n: usize) {
    for i in 0..n.min(dst.len()).min(src.len()) {
        dst[i] += src[i];
    }
}

#[inline]
/// In-place RMSNorm.
pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len().min(weight.len());
    if n == 0 {
        return;
    }
    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    ss /= n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

/// Dequantize a 1-D norm weight row (`attn_norm` / `ffn_norm` / `output_norm`) into `out`.
pub fn dequant_norm_row_into(
    mmap: &[u8],
    tensor_data_start: u64,
    info: &GgufTensorInfo,
    out: &mut [f32],
) -> usize {
    let n = info.dims[0] as usize;
    if n == 0 || n > out.len() {
        return 0;
    }
    let raw = match etalon_model::ggml::fetch_tensor_bytes(mmap, tensor_data_start, info) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    etalon_model::ggml::dequantize_row_into(raw, info.ggml_type, n, &mut out[..n]).unwrap_or(0)
}

/// Dequantize a 1-D norm weight row directly from raw bytes into `out`.
pub fn dequant_norm_from_raw(raw: &[u8], info: &GgufTensorInfo, out: &mut [f32]) -> usize {
    let n = info.dims[0] as usize;
    if n == 0 || n > out.len() {
        return 0;
    }
    etalon_model::ggml::dequantize_row_into(raw, info.ggml_type, n, &mut out[..n]).unwrap_or(0)
}

/// Pre-norm: copy `hidden` into `h_norm`, apply RMSNorm with `norm_info` weights; return slice to use.
pub fn prepare_pre_norm_input<'a>(
    hidden: &'a [f32],
    emb_dim: usize,
    norm_info: Option<&GgufTensorInfo>,
    mmap: Option<&[u8]>,
    tensor_data_start: u64,
    h_norm: &'a mut [f32],
    norm_w: &mut [f32],
) -> &'a [f32] {
    let n_embd = emb_dim.min(hidden.len()).min(h_norm.len());
    if let (Some(mmap), Some(info)) = (mmap, norm_info) {
        if dequant_norm_row_into(mmap, tensor_data_start, info, norm_w) >= n_embd {
            h_norm[..n_embd].copy_from_slice(&hidden[..n_embd]);
            rms_norm_inplace(&mut h_norm[..n_embd], &norm_w[..n_embd], RMS_NORM_EPS);
            return &h_norm[..n_embd];
        }
    }
    &hidden[..n_embd]
}
