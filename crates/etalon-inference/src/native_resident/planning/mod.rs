//! Cold resident plans; the embedding supplies model access, admitted resources and policy.
use crate::ordinary::{gpu_params::*, matmul_dims, quant_support::*, RMS_NORM_EPS};
use etalon_model::inference::{LayerTensors, ModelHyperparams, TensorInfo as GgufTensorInfo};
mod fused_quant;
use fused_quant::fused_ffn_quant_supported;
mod bindings;
mod decode_layers;
mod decode_qkv;
mod prefill_arena;
mod prefill_fusions;
mod resident_decode;
mod resources;
mod verify_arena;
use bindings::{ubind, Bindings};
pub use resources::ResidentPlanResources;

const MAX_HIDDEN_DIM: usize = 4096;
const MAX_STACK_GEMM_DIM: usize = 10240;
const PREFILL_CHUNK_SIZE: usize = 64;
const VOCAB_CHUNK_ROWS: usize = 10240;
use crate::native_resident::verify_arena_plan::VERIFY_BATCH_MAX;

pub struct ResidentPlanner<'a, P> {
    pub host: &'a P,
}

#[derive(Clone, Copy)]
pub struct ResidentPlanOptions {
    pub coop_gemv: bool,
    pub ffn_fusion: bool,
    pub triple_qkv: bool,
    pub multirow: bool,
    pub warp_gemv: bool,
    pub ffn_warp: bool,
    pub ffn_mr: bool,
    pub topk_block_size: usize,
}

pub enum PlanEvent {
    FfnF16(u32, u32, u32),
    DecodeBuilt {
        layers: u32,
        passes: usize,
        fused: bool,
        triple: bool,
        dual: bool,
        ffn_mr: bool,
        multirow: bool,
        ffn_warp: bool,
    },
    PrefillBuilt {
        layers: u32,
        passes: usize,
        fused: bool,
        dual: bool,
    },
    VerifyBuilt {
        layers: u32,
        chunks: usize,
        batch: usize,
    },
}

pub trait ResidentPlanProvider {
    type Index;
    type ModelLease: AsRef<[u8]>;
    type Layout;
    fn model_lease(&self) -> Option<Self::ModelLease>;
    fn hyperparams(index: &Self::Index) -> ModelHyperparams;
    fn data_start(index: &Self::Index) -> u64;
    fn tensors(index: &Self::Index, layer: u32) -> LayerTensors;
    fn logits_info(index: &Self::Index) -> Option<GgufTensorInfo>;
    fn output_norm_info(index: &Self::Index) -> Option<GgufTensorInfo>;
    fn tensor_bytes<'a>(bytes: &'a [u8], start: u64, info: &GgufTensorInfo) -> Option<&'a [u8]>;
    fn norm_row(bytes: &[u8], start: u64, info: &GgufTensorInfo, out: &mut [f32]) -> usize;
    fn plan_layout(&self) -> Option<Self::Layout>;
    fn max_context(layout: &Self::Layout) -> u32;
    fn layer_stride(layout: &Self::Layout) -> u32;
    fn plan_options(&self) -> ResidentPlanOptions;
    fn plan_resources(&self) -> ResidentPlanResources<'_>;
    fn gpu_device(&self) -> &wgpu::Device;
    fn gpu_queue(&self) -> &wgpu::Queue;
    fn native_gemm_bind_layout(&self, coop: bool) -> &wgpu::BindGroupLayout;
    fn resident_weight_buffer(&self, key: u64, raw: &[u8]) -> Option<wgpu::Buffer>;
    fn promote_matrix_to_f16_resident(
        &self,
        info: &GgufTensorInfo,
        raw: &[u8],
    ) -> Option<(wgpu::Buffer, u32, u32, u32)>;
    fn prewarm_layer(&self, layer: u32, matrices: &[(&GgufTensorInfo, &[u8], usize, usize)]);
    fn set_ffn_fusion_in_resident(&self, fused: bool);
    fn event(&self, event: PlanEvent);
    fn attention_gpu_params(
        h: &ModelHyperparams,
        layout: &Self::Layout,
        layer: u32,
        token_idx: u32,
        info: &GgufTensorInfo,
        raw_len: usize,
        proj_kind: u32,
        num_tokens_in_batch: u32,
        batch_start_token_idx: u32,
        mask_active: u32,
        mask_word_count: u32,
        out_stride_elems: u32,
    ) -> AttentionGpuParams;
}

fn topk_params_bytes_with_base(rows: u32, k: u32, block: u32, base: u32) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    for (chunk, value) in bytes.chunks_exact_mut(4).zip([rows, k, block, base]) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}
