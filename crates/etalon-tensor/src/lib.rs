//! Prepared tensor execution, unified forward arena, and fused attention.

pub mod view;
pub mod kv_cache;
pub mod forward_arena;
pub mod attention;
pub mod gpu_sieve_topk;
pub mod cuda_mega;
pub mod dml_lane;

pub use view::{QuantScheme, TensorView};
pub use kv_cache::{KvCacheLayout, PagedKvBlockTable};
pub use forward_arena::{ForwardArena, ForwardMode};
pub use attention::{
    online_softmax_rescale, AttentionParams, FusedAttentionPipeline, FUSED_ATTENTION_WGSL,
};
pub use gpu_sieve_topk::{
    LogitMaskHeader, TopkParams, TopkReductionPipeline, TOPK_REDUCTION_WGSL,
};
