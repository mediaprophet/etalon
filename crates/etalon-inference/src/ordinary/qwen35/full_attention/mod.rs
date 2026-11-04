//! Caller-buffered one-token HF Qwen3.5 gated attention; no full-model admission.
mod execute;
mod layout;
mod math;
pub use execute::execute_normalized;
pub use layout::WorkspaceLayout;

/// Unscaled text RoPE only: all three multimodal axes must have the same
/// position. Frequencies are explicit, not inferred from an architecture default.
#[derive(Clone, Copy, Debug)]
pub struct TextRope<'a> {
    pub position: usize,
    pub inv_freq: &'a [f32],
}
