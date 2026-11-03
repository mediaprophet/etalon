//! Existing routed-expert math and NVFP4 arithmetic, with caller-owned buffers.
// Preserve the original loop and shape-check arithmetic during extraction.
#[allow(clippy::manual_is_multiple_of, clippy::needless_range_loop)]
pub mod nvfp4;
#[allow(clippy::needless_range_loop)]
pub mod routing;
