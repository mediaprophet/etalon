//! Caller-buffered convolution and recurrent state updates.
// Preserve the existing caller-buffer API and numerical accumulation order.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub mod gated_deltanet;

pub mod qwen_delta;
pub mod qwen_projection;

pub use qwen_delta::{step_qwen_delta, QwenDeltaError};
pub use qwen_projection::{
    gated_rms_norm, normalize_qwen_qk, qwen_gates, silu_into, unpack_qwen_projections,
    QwenHeadLayout, QwenProjectionPacking,
};
