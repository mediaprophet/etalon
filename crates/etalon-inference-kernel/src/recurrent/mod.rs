//! Caller-buffered convolution and recurrent state updates.
// Preserve the existing caller-buffer API and numerical accumulation order.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub mod gated_deltanet;
