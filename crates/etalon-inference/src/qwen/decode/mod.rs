//! Persistent Qwen4Exp decode session: per-layer recurrent state and the
//! complete 48-layer token step.
//!
//! A session owns every piece of mutable state the trained graph needs —
//! 36 GatedDeltaNet convolutions and delta-rule matrices, 12 QSA K/V and
//! raw indexer caches, the PLE n-gram context and dilated-conv history, and
//! the four-stream residual.  All weight bytes are streamed row-wise from
//! the E: trunk and C: PLE payload; nothing here maps or stages the model.

use std::time::Instant;

use etalon_model::inference::TensorIndex;

use super::{
    execute_ple_block, execute_streamed_final_hyper_connection, execute_streamed_gdn_moe_layer,
    execute_streamed_qsa_moe_layer, GatedDeltaBuffers, GatedDeltaError, GatedDeltaState,
    PleBlockBuffers, PleBlockError, PleTokenHistory, QsaState, Qwen4ExpRuntime, StreamedArgmax,
    StreamedGdnMoeLayerBuffers, StreamedHyperBuffers, StreamedHyperError, StreamedLayerError,
    StreamedMoeBuffers, StreamedQsaBuffers, StreamedQsaMoeLayerBuffers, TrunkNvmeError,
    QWEN4EXP_GDN_HEADS, QWEN4EXP_GDN_HEAD_DIM, QWEN4EXP_GDN_KEY_HEADS, QWEN4EXP_HYPER_STREAMS,
    QWEN4EXP_PLE_HISTORY, QWEN4EXP_QSA_HEAD_DIM, QWEN4EXP_QSA_INDEXER_HEADS,
    QWEN4EXP_QSA_INDEXER_HEAD_DIM, QWEN4EXP_QSA_KV_HEADS, QWEN4EXP_QSA_MAX_SELECTED,
    QWEN4EXP_QSA_QUERY_HEADS, QWEN4EXP_QSA_TOP_BLOCKS,
};

const GDN_CONV_WIDTH: usize =
    QWEN4EXP_GDN_KEY_HEADS * QWEN4EXP_GDN_HEAD_DIM * 2 + QWEN4EXP_GDN_HEADS * QWEN4EXP_GDN_HEAD_DIM;
const GDN_CONV_HISTORY: usize = 3;
const GDN_DELTA_WIDTH: usize = QWEN4EXP_GDN_HEADS * QWEN4EXP_GDN_HEAD_DIM * QWEN4EXP_GDN_HEAD_DIM;
const QSA_KV_WIDTH: usize = QWEN4EXP_QSA_KV_HEADS * QWEN4EXP_QSA_HEAD_DIM;
const QSA_INDEX_QUERY_WIDTH: usize = QWEN4EXP_QSA_INDEXER_HEADS * QWEN4EXP_QSA_INDEXER_HEAD_DIM;
const HYPER_LOW_RANK: usize = 320;
const MOE_INTERMEDIATE: usize = 640;
const ROUTER_EXPERTS: usize = 512;
const RAW_ROW_BYTES: usize = 65_536;

#[derive(Debug)]
pub enum Qwen4ExpDecodeError {
    MissingTensor,
    PublicationDenied,
    OutputTooSmall,
    SizeOverflow,
    AllocationFailed,
    BudgetExceeded { required: u64, limit: u64 },
    MissingTokenMixer { layer: u32 },
    Ple(PleBlockError),
    Layer(StreamedLayerError),
    Hyper(StreamedHyperError),
    Trunk(TrunkNvmeError),
}

impl core::fmt::Display for Qwen4ExpDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PublicationDenied => write!(
                f,
                "Qwen candidate publication refused; discard numerical state"
            ),
            Self::OutputTooSmall => write!(f, "Qwen token output buffer is too small"),
            Self::SizeOverflow => write!(f, "Qwen state size overflowed"),
            Self::AllocationFailed => write!(f, "Qwen state allocation failed"),
            Self::BudgetExceeded { required, limit } => {
                write!(f, "Qwen state requires {required} bytes, budget is {limit}")
            }
            Self::MissingTensor => {
                write!(f, "Qwen4Exp decode is missing a required trunk tensor")
            }
            Self::MissingTokenMixer { layer } => {
                write!(f, "Qwen4Exp layer {layer} has no executable token mixer")
            }
            Self::Ple(error) => error.fmt(f),
            Self::Layer(error) => error.fmt(f),
            Self::Hyper(error) => error.fmt(f),
            Self::Trunk(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Qwen4ExpDecodeError {}
impl From<PleBlockError> for Qwen4ExpDecodeError {
    fn from(value: PleBlockError) -> Self {
        Self::Ple(value)
    }
}
impl From<StreamedLayerError> for Qwen4ExpDecodeError {
    fn from(value: StreamedLayerError) -> Self {
        Self::Layer(value)
    }
}
impl From<StreamedHyperError> for Qwen4ExpDecodeError {
    fn from(value: StreamedHyperError) -> Self {
        Self::Hyper(value)
    }
}
impl From<TrunkNvmeError> for Qwen4ExpDecodeError {
    fn from(value: TrunkNvmeError) -> Self {
        Self::Trunk(value)
    }
}
impl From<GatedDeltaError> for Qwen4ExpDecodeError {
    fn from(value: GatedDeltaError) -> Self {
        Self::Layer(value.into())
    }
}

mod driver;
mod scratch;
mod state;
mod step;
mod trace;
pub use driver::*;
pub use scratch::*;
pub use state::*;
pub use step::*;
pub use trace::*;

mod allocation;
