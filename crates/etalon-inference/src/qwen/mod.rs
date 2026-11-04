//! Qwen hybrid architecture execution and state management (Work Package F11).
//!
//! Implements GatedDeltaNet convolution, linear-attention recurrent updates,
//! and dual-state checkpointing for Qwen3.6 MoE on the RTX A2000.

/// Qwen4Exp MoE top-k width — shared by residency metadata (wasm-safe) and streamed MoE (native).
pub const QWEN4EXP_TOP_EXPERTS: usize = 10;
/// Qwen4Exp routed expert bank size (native streamed MoE geometry).
pub const QWEN4EXP_EXPERT_COUNT: usize = 512;

pub mod checkpoint;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod decode;
pub mod expert_residency;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod expert_tile;
pub mod gated_deltanet;
#[cfg(feature = "qwen-metadata")]
pub mod graph_contract;
pub mod hyper_connection;
pub mod numerics;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod ple_block;
#[cfg(feature = "qwen-metadata")]
pub mod ple_ngram;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod ple_nvme;
#[cfg(feature = "qwen-metadata")]
pub mod storage_plan;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod streamed_gdn;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod streamed_hyper;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod streamed_layer;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod streamed_moe;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod streamed_qsa;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod trunk_nvme;

pub use checkpoint::{QwenCheckpointRegistry, QwenDualCheckpoint};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use decode::{
    decode_step, decode_tokens, Qwen4ExpDecodeError, Qwen4ExpDecodeReceipt, Qwen4ExpDecodeScratch,
    Qwen4ExpSession, Qwen4ExpTraceBuffer, Qwen4ExpTraceRecord, Qwen4ExpTraceSink,
    Qwen4ExpTraceSummary, QWEN4EXP_TRACE_META_LAYER,
};
pub use expert_residency::{
    QwenExpertAccess, QwenExpertKey, QwenExpertResidency, MAX_QWEN_EXPERT_SLOTS,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use expert_tile::{
    QwenExpertPlane, QwenExpertTileAdmission, QwenExpertTileDescriptor, QwenExpertTileError,
    QwenExpertTilePlaneKind, QwenExpertTileReader, QWEN_EXPERT_TILE_FORMAT,
    QWEN_EXPERT_TILE_SUFFIX,
};
pub use gated_deltanet::{
    step_causal_conv1d, step_gated_deltanet, QwenStateError, CAUSAL_CONV_KERNEL,
};
#[cfg(feature = "qwen-metadata")]
pub use graph_contract::{
    validate_qwen4exp_graph, Qwen4ExpGraphError, Qwen4ExpGraphReport, Qwen4ExpLayerContract,
    QWEN4EXP_LAYER_GDN, QWEN4EXP_LAYER_HC_ATTN, QWEN4EXP_LAYER_HC_FFN, QWEN4EXP_LAYER_MOE,
    QWEN4EXP_LAYER_PLE, QWEN4EXP_LAYER_QSA,
};
pub use hyper_connection::{
    group_rms_norm_in_place, group_rms_norm_into, inject_stream_update, mix_streams_into,
};
pub use numerics::{add_assign, add_into, rms_norm_into, QwenNumericError};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use ple_block::{
    execute_ple_block, PleBlockBuffers, PleBlockError, QWEN4EXP_HYPER_STREAMS, QWEN4EXP_PLE_HISTORY,
};
#[cfg(feature = "qwen-metadata")]
pub use ple_ngram::{
    select_rows as select_ple_rows, sort_dedup_rows, PleNgramError, PleTokenHistory,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use ple_nvme::{PleGatherReceipt, PleIoStats, PleNvmeError, PleNvmeReader};
#[cfg(feature = "qwen-metadata")]
pub use storage_plan::{
    plan_qwen4exp_storage, Qwen4ExpStorageError, Qwen4ExpStoragePlan,
    DEFAULT_QWEN4EXP_HOST_OS_FLOOR,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use streamed_gdn::{
    execute_streamed_gated_delta, GatedDeltaBuffers, GatedDeltaError, GatedDeltaState,
    QWEN4EXP_GDN_HEADS, QWEN4EXP_GDN_HEAD_DIM, QWEN4EXP_GDN_KEY_HEADS,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use streamed_hyper::{
    execute_streamed_final_hyper_connection, execute_streamed_hyper_connection,
    inject_hyper_update, mix_hyper_streams, StreamedHyperBuffers, StreamedHyperError,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use streamed_layer::{
    execute_streamed_gdn_moe_layer, execute_streamed_qsa_moe_layer, StreamedGdnMoeLayerBuffers,
    StreamedLayerError, StreamedMoeBuffers, StreamedQsaMoeLayerBuffers,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use streamed_moe::{
    execute_streamed_moe, execute_streamed_moe_with_tiles, normalize_top_experts,
    select_top_experts, StreamedMoeError,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use streamed_qsa::{
    execute_streamed_qsa, QsaState, StreamedQsaBuffers, StreamedQsaError,
    QWEN4EXP_QSA_COMPRESS_RATIO, QWEN4EXP_QSA_HEAD_DIM, QWEN4EXP_QSA_INDEXER_HEADS,
    QWEN4EXP_QSA_INDEXER_HEAD_DIM, QWEN4EXP_QSA_KV_HEADS, QWEN4EXP_QSA_MAX_SELECTED,
    QWEN4EXP_QSA_QUERY_HEADS, QWEN4EXP_QSA_ROTARY_DIM, QWEN4EXP_QSA_TOP_BLOCKS, QWEN4EXP_QSA_TOP_K,
};
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use trunk_nvme::{StreamedArgmax, TrunkIoStats, TrunkNvmeError, TrunkNvmeReader};

#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
mod runtime;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use runtime::*;

#[cfg(feature = "qwen-metadata")]
pub use etalon_model::inference::HyperparamsSource as QwenHyperparams;

#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub mod p64_runtime;
#[cfg(all(feature = "qwen-native", not(target_arch = "wasm32")))]
pub use p64_runtime::*;
