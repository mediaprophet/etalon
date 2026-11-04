//! Qwen4Exp adapter selecting 512 experts, top-10 and the sigmoid-input shared gate.

use etalon_model::inference::LayerTensors;

use super::{QwenExpertTileReader, TrunkNvmeReader};

pub use super::{QWEN4EXP_EXPERT_COUNT, QWEN4EXP_TOP_EXPERTS};

pub use crate::moe::streamed_executor::StreamedMoeError;

pub fn select_top_experts(
    logits: &[f32],
    indices: &mut [usize; QWEN4EXP_TOP_EXPERTS],
    selected_logits: &mut [f32; QWEN4EXP_TOP_EXPERTS],
) -> Result<(), StreamedMoeError> {
    crate::moe::streamed_routing::select_top_experts(logits, indices, selected_logits)
        .map_err(|_| StreamedMoeError::BufferTooSmall)
}
pub fn normalize_top_experts(
    logits: &[f32; QWEN4EXP_TOP_EXPERTS],
    out: &mut [f32; QWEN4EXP_TOP_EXPERTS],
) {
    crate::moe::streamed_routing::normalize_top_experts(logits, out)
}
#[cfg(test)]
fn swiglu_into(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<(), StreamedMoeError> {
    crate::moe::streamed_routing::swiglu_into(gate, up, out)
        .map_err(|_| StreamedMoeError::BufferTooSmall)
}

/// Source-only compatibility path. New native callers should use
/// [`execute_streamed_moe_with_tiles`] to supply any validated C: tiles.
#[allow(clippy::too_many_arguments)]
pub fn execute_streamed_moe(
    reader: &mut TrunkNvmeReader,
    tensors: &LayerTensors,
    input: &[f32],
    out: &mut [f32],
    router_logits: &mut [f32],
    top_indices: &mut [usize; QWEN4EXP_TOP_EXPERTS],
    top_logits: &mut [f32; QWEN4EXP_TOP_EXPERTS],
    top_weights: &mut [f32; QWEN4EXP_TOP_EXPERTS],
    gate: &mut [f32],
    up: &mut [f32],
    activation: &mut [f32],
    expert_out: &mut [f32],
    raw_row: &mut [u8],
    dequantized_row: &mut [f32],
) -> Result<(), StreamedMoeError> {
    execute_streamed_moe_with_tiles(
        reader,
        tensors,
        input,
        out,
        router_logits,
        top_indices,
        top_logits,
        top_weights,
        &[None; QWEN4EXP_TOP_EXPERTS],
        gate,
        up,
        activation,
        expert_out,
        raw_row,
        dequantized_row,
    )
}

/// Decode one Qwen4Exp MoE block directly from the source trunk.
///
/// `router_logits` is at least 512 elements. `gate`, `up`, and `activation`
/// are at least the expert intermediate width; `expert_out` and `out` are at
/// least the hidden width. `raw_row` and `dequantized_row` are reused for each
/// disk row. No expert tensor, temporary vector, or router sorting allocation
/// is created here.
#[allow(clippy::too_many_arguments)]
pub fn execute_streamed_moe_with_tiles(
    reader: &mut TrunkNvmeReader,
    tensors: &LayerTensors,
    input: &[f32],
    out: &mut [f32],
    router_logits: &mut [f32],
    top_indices: &mut [usize; QWEN4EXP_TOP_EXPERTS],
    top_logits: &mut [f32; QWEN4EXP_TOP_EXPERTS],
    top_weights: &mut [f32; QWEN4EXP_TOP_EXPERTS],
    cached_tiles: &[Option<&QwenExpertTileReader>; QWEN4EXP_TOP_EXPERTS],
    gate: &mut [f32],
    up: &mut [f32],
    activation: &mut [f32],
    expert_out: &mut [f32],
    raw_row: &mut [u8],
    dequantized_row: &mut [f32],
) -> Result<(), StreamedMoeError> {
    let intermediate = tensors
        .moe_gate_exps
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?
        .dims[1];
    let profile = crate::moe::streamed_profile::StreamedSwiGluProfile {
        hidden: input.len(),
        intermediate: usize::try_from(intermediate).map_err(|_| StreamedMoeError::InvalidShape)?,
        expert_count: QWEN4EXP_EXPERT_COUNT,
        shared_expert: crate::moe::streamed_profile::SharedExpertGate::SigmoidInputDotSameWidth,
    };
    crate::moe::streamed_executor::execute_streamed_swiglu_with_tiles(
        profile,
        reader,
        tensors,
        input,
        out,
        router_logits,
        top_indices,
        top_logits,
        top_weights,
        cached_tiles,
        gate,
        up,
        activation,
        expert_out,
        raw_row,
        dequantized_row,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_experts_are_deterministic_and_normalized() {
        let mut indices = [0usize; QWEN4EXP_TOP_EXPERTS];
        let mut selected = [0.0f32; QWEN4EXP_TOP_EXPERTS];
        let mut logits = [0.0f32; QWEN4EXP_EXPERT_COUNT];
        logits[42] = 3.0;
        logits[7] = 3.0;
        logits[256] = 5.0;
        select_top_experts(&logits, &mut indices, &mut selected).unwrap();
        assert_eq!(&indices[..3], &[256, 7, 42]);
        let mut weights = [0.0f32; QWEN4EXP_TOP_EXPERTS];
        normalize_top_experts(&selected, &mut weights);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn swiglu_uses_silu_gate() {
        let mut output = [0.0f32; 2];
        swiglu_into(&[0.0, 1.0], &[4.0, 2.0], &mut output).unwrap();
        assert_eq!(output[0], 0.0);
        assert!(output[1] > 1.4 && output[1] < 1.5);
    }
}
