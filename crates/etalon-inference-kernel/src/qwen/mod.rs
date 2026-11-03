//! Caller-buffered Qwen vector and Hyper-Connection arithmetic.
// Preserve the original stream accumulation order during extraction.
#[allow(clippy::needless_range_loop)]
pub mod hyper_connection;
pub mod numerics;
pub mod ple;

pub use hyper_connection::{
    group_rms_norm_in_place, group_rms_norm_into, inject_stream_update, mix_streams_into,
};
pub use numerics::{add_assign, add_into, rms_norm_into, QwenNumericError};
