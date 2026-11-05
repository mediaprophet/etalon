//! Numerical projections; model custody, execution permission and token filtering belong to the host.
use crate::ordinary::cpu::{dequant_norm_from_raw, dequant_norm_row_into, rms_norm_inplace};
use crate::ordinary::RMS_NORM_EPS;
use etalon_model::inference::TensorInfo as GgufTensorInfo;
#[allow(dead_code)]
const MAX_HIDDEN_DIM: usize = 4096;
#[allow(dead_code)]
const MAX_STACK_GEMM_IN: usize = 10240;
pub const VOCAB_CHUNK_ROWS: usize = 10240;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingArgmaxResult {
    pub best_token_id: u32,
    pub max_logit: f32,
}
pub trait ProjectionProvider {
    type Index;
    type Mask;
    fn logits(index: &Self::Index) -> Option<GgufTensorInfo>;
    fn output_norm(index: &Self::Index) -> Option<GgufTensorInfo>;
    fn data_start(index: &Self::Index) -> u64;
    fn embedding_dim(index: &Self::Index) -> usize;
    fn model_bytes(&self) -> Option<&[u8]>;
    fn nvfp4_output(&self) -> Option<(&[u8], &[u8], &[u8])>;
    fn fetch_tensor_raw_bytes(&self, info: &GgufTensorInfo) -> Option<&[u8]>;
    fn gemm(
        &self,
        info: &GgufTensorInfo,
        raw: &[u8],
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
    fn allows(mask: &Self::Mask, token: u32) -> bool;
    fn scrub(&self, values: &mut [f32], n: usize);
    fn update_argmax(
        &self,
        logits: &[f32],
        rows: usize,
        chunk: usize,
        mask: Option<&Self::Mask>,
        best: &mut u32,
        max: &mut f32,
    ) {
        for (local, &value) in logits.iter().take(rows).enumerate() {
            let token = (chunk * VOCAB_CHUNK_ROWS + local) as u32;
            let score = if mask.map(|m| Self::allows(m, token)).unwrap_or(true) {
                value
            } else {
                f32::NEG_INFINITY
            };
            if score > *max {
                *max = score;
                *best = token;
            }
        }
    }
    fn update_argmax_unmasked(
        &self,
        logits: &[f32],
        rows: usize,
        chunk: usize,
        best: &mut u32,
        max: &mut f32,
    ) {
        self.update_argmax(logits, rows, chunk, None, best, max)
    }
}
pub struct ProjectionEngine<'a, P> {
    pub host: &'a P,
}
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
mod gpu;
mod output;
pub mod topk;
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
pub use gpu::*;
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
mod gemm;
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
mod top1;
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
mod topk_dispatch;
#[cfg(feature = "gpu-attention")]
pub mod topk_resources;

#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
mod embedding;
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
mod fused_block;

#[cfg(feature = "gpu-attention")]
pub mod weight_resources;
pub mod weights;

#[cfg(all(not(target_arch = "wasm32"), feature = "gpu-attention"))]
pub mod readback;

#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "gpu-attention",
    any(feature = "native-directml", feature = "native-accelerate")
))]
pub mod backend_q4;
