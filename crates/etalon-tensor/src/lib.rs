//! Prepared tensor execution, unified forward arena, and fused attention.

pub mod attention;
pub mod cuda_mega;
pub mod dml_lane;
pub mod forward_arena;
pub mod gpu_sieve_topk;
mod kernel_limits;
pub mod kv_cache;
pub use kernel_limits::TensorPipelineError;
pub mod view;

pub use attention::{
    online_softmax_rescale, pack_attention_weights_and_mask, packed_fused_attention_wgsl,
    AttentionParams, FusedAttentionParams, FusedAttentionPipeline, PackedAttentionError,
    FUSED_ATTENTION_WGSL,
};
pub use cuda_mega::{
    build_parameter_packs, collect_and_validate_layer_keys, weight_fingerprint,
    wmma_usable_for_batch, CapturedCudaMegaPass, CudaMegaPassExecutor, LayerWeightKeys,
    MegaPassError, MegaPassExecutionPath, MegaPassLayerDims, MegaPassLayerWeights,
    MegaPassParameterPacks, MegaPassPlan, MegaPassPlanView, MegaPassStepDescriptor,
    MegaPassWeightLayout, CUDA_SOA_SLAB_BYTES, MAX_CUDA_CONTEXT_WINDOW, MAX_MEGA_PASS_LAYERS,
};
pub use dml_lane::{
    dequantize_q4_k_block, dequantize_q4_k_tensor, dequantize_q8_0_block, dequantize_q8_0_tensor,
    f16_bits_to_f32, AdapterMemoryInfo, DirectMlComputeLane, DirectStorageValidator, DmlError,
    DmlGemmOp, DmlGemmParams, DmlLane, DmlStatus, DmlTensorDesc2D, DIRECTSTORAGE_PAGE_ALIGN,
    DML_GEMM_WGSL, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE,
};
pub use forward_arena::{
    ForwardArena, ForwardMode, TensorArenaError, TokenReadbackError, DEFAULT_FORWARD_ARENA_BUDGET,
};
pub use gpu_sieve_topk::{LogitMaskHeader, TopkParams, TopkReductionPipeline, TOPK_REDUCTION_WGSL};
pub use kv_cache::{KvCacheLayout, PagedKvBlockTable};
pub use view::{QuantScheme, TensorView};
