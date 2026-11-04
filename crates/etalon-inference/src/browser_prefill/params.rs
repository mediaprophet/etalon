use crate::ordinary::gpu_params::{ElemGpuParams, GemmGpuParams};
use crate::ordinary::RMS_NORM_EPS;
use etalon_model::inference::TensorInfo as GgufTensorInfo;
pub fn mc8_gemm_params(
    info: &GgufTensorInfo,
    raw_len: usize,
    n_in: usize,
    n_out: usize,
    n_batch: u32,
    in_row_stride: u32,
    out_row_stride: u32,
) -> GemmGpuParams {
    GemmGpuParams {
        n_in: n_in as u32,
        n_out: n_out as u32,
        weight_ggml_type: info.ggml_type,
        weight_row_elems: info.dims[0] as u32,
        weight_byte_len: raw_len as u32,
        n_batch: n_batch.max(1),
        in_row_stride,
        out_row_stride,
    }
}
pub fn mc8_elem_params(
    op: u32,
    n: u32,
    batch: u32,
    a_row_stride: u32,
    b_row_stride: u32,
    out_row_stride: u32,
    a_slot: u32,
    b_slot: u32,
    out_slot: u32,
) -> ElemGpuParams {
    ElemGpuParams {
        n,
        batch: batch.max(1),
        op,
        eps: RMS_NORM_EPS,
        a_row_stride,
        b_row_stride,
        out_row_stride,
        a_slot,
        b_slot,
        out_slot,
        _pad: 0,
    }
}
pub fn mc8_prefill_row_off(t: u32, row_stride_floats: usize) -> wgpu::BufferAddress {
    (t as usize * row_stride_floats * 4) as wgpu::BufferAddress
}
pub fn mc8_emb_off(t: u32, n_embd: usize) -> wgpu::BufferAddress {
    (t as usize * n_embd * 4) as wgpu::BufferAddress
}
pub fn mc8_prefill_row_stride(n_embd: usize, n_ffn_est: usize, gemm_max_out_dim: u32) -> usize {
    (n_embd + 2 * n_ffn_est + n_embd)
        .max(gemm_max_out_dim as usize * 2)
        .max(n_embd)
}
