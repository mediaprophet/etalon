//! Caller-buffered routed SwiGLU with selected-score softmax and one shared expert.
//! Only selected `[input, output, expert]` planes are read. Other router
//! transforms, activations and shared policies require separate profiles.

use super::expert_tile::{ExpertTileError, ExpertTilePlaneKind, ExpertTileReader};
use super::streamed_profile::{SharedExpertGate, StreamedSwiGluProfile};
use super::streamed_routing::{normalize_top_experts, select_top_experts, swiglu_into};
use crate::weights::{TrunkNvmeError, TrunkNvmeReader};
use etalon_model::inference::LayerTensors;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamedMoeError {
    MissingTensor,
    InvalidShape,
    BufferTooSmall,
    Trunk(TrunkNvmeError),
    Tile(ExpertTileError),
    TileIdentityMismatch,
}

impl core::fmt::Display for StreamedMoeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingTensor => write!(f, "streamed SwiGLU MoE tensor is missing"),
            Self::InvalidShape => write!(f, "streamed SwiGLU MoE tensor shape is invalid"),
            Self::BufferTooSmall => write!(f, "streamed SwiGLU MoE caller buffer is too small"),
            Self::Trunk(error) => error.fmt(f),
            Self::Tile(error) => error.fmt(f),
            Self::TileIdentityMismatch => write!(
                f,
                "cached expert tile does not match its routed layer/expert"
            ),
        }
    }
}

impl std::error::Error for StreamedMoeError {}

impl From<TrunkNvmeError> for StreamedMoeError {
    fn from(value: TrunkNvmeError) -> Self {
        Self::Trunk(value)
    }
}

impl From<ExpertTileError> for StreamedMoeError {
    fn from(value: ExpertTileError) -> Self {
        Self::Tile(value)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn execute_streamed_swiglu_with_tiles<const K: usize>(
    profile: StreamedSwiGluProfile,
    reader: &mut TrunkNvmeReader,
    tensors: &LayerTensors,
    input: &[f32],
    out: &mut [f32],
    router_logits: &mut [f32],
    top_indices: &mut [usize; K],
    top_logits: &mut [f32; K],
    top_weights: &mut [f32; K],
    cached_tiles: &[Option<&ExpertTileReader>; K],
    gate: &mut [f32],
    up: &mut [f32],
    activation: &mut [f32],
    expert_out: &mut [f32],
    raw_row: &mut [u8],
    dequantized_row: &mut [f32],
) -> Result<(), StreamedMoeError> {
    let router = tensors
        .moe_router
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;
    let gate_experts = tensors
        .moe_gate_exps
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;
    let up_experts = tensors
        .moe_up_exps
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;
    let down_experts = tensors
        .moe_down_exps
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;
    let shared_gate = tensors
        .moe_shared_gate
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;
    let shared_up = tensors
        .moe_shared_up
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;
    let shared_down = tensors
        .moe_shared_down
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;
    let shared_gate_input = tensors
        .moe_shared_gate_input
        .as_ref()
        .ok_or(StreamedMoeError::MissingTensor)?;

    profile.validate::<K>()?;
    let hidden = profile.hidden;
    let intermediate = profile.intermediate;
    if input.len() != hidden
        || router.n_dims != 2
        || router.dims[0] != hidden as u64
        || router.dims[1] != profile.expert_count as u64
        || gate_experts.n_dims != 3
        || up_experts.n_dims != 3
        || down_experts.n_dims != 3
        || gate_experts.dims[0] != hidden as u64
        || up_experts.dims[0] != hidden as u64
        || gate_experts.dims[1] != intermediate as u64
        || up_experts.dims[1] != intermediate as u64
        || gate_experts.dims[2] != profile.expert_count as u64
        || up_experts.dims[2] != profile.expert_count as u64
        || down_experts.dims[0] != intermediate as u64
        || down_experts.dims[1] != hidden as u64
        || down_experts.dims[2] != profile.expert_count as u64
        || shared_gate.n_dims != 2
        || shared_up.n_dims != 2
        || shared_down.n_dims != 2
        || shared_gate.dims[0] != hidden as u64
        || shared_up.dims[0] != hidden as u64
        || shared_gate.dims[1] != intermediate as u64
        || shared_up.dims[1] != intermediate as u64
        || shared_down.dims[0] != intermediate as u64
        || shared_down.dims[1] != hidden as u64
        || shared_gate_input.dims[0] != hidden as u64
        || !((shared_gate_input.n_dims == 1)
            || (shared_gate_input.n_dims == 2 && shared_gate_input.dims[1] == 1))
    {
        return Err(StreamedMoeError::InvalidShape);
    }
    for tensor in [
        router,
        gate_experts,
        up_experts,
        down_experts,
        shared_gate,
        shared_up,
        shared_down,
        shared_gate_input,
    ] {
        super::streamed_profile::validate_tensor_span(tensor)?;
    }
    if out.len() < hidden
        || router_logits.len() < profile.expert_count
        || gate.len() < intermediate
        || up.len() < intermediate
        || activation.len() < intermediate
        || expert_out.len() < hidden
    {
        return Err(StreamedMoeError::BufferTooSmall);
    }

    reader.gemv_rows_into(
        router,
        0,
        profile.expert_count,
        input,
        router_logits,
        raw_row,
        dequantized_row,
    )?;
    select_top_experts(
        &router_logits[..profile.expert_count],
        top_indices,
        top_logits,
    )
    .map_err(|_| StreamedMoeError::BufferTooSmall)?;
    normalize_top_experts(top_logits, top_weights);
    for value in &mut out[..hidden] {
        *value = 0.0;
    }

    for route in 0..K {
        let expert = top_indices[route];
        if let Some(tile) = cached_tiles[route] {
            if u32::from(tile.descriptor.layer) != tensors.layer_idx
                || usize::from(tile.descriptor.expert) != expert
            {
                return Err(StreamedMoeError::TileIdentityMismatch);
            }
            tile.gemv_into(ExpertTilePlaneKind::Gate, input, gate, dequantized_row)?;
            tile.gemv_into(ExpertTilePlaneKind::Up, input, up, dequantized_row)?;
            swiglu_into(&gate[..intermediate], &up[..intermediate], activation)
                .map_err(|_| StreamedMoeError::BufferTooSmall)?;
            tile.gemv_into(
                ExpertTilePlaneKind::Down,
                &activation[..intermediate],
                expert_out,
                dequantized_row,
            )?;
        } else {
            reader.gemv_expert_plane_into(
                gate_experts,
                expert,
                input,
                gate,
                raw_row,
                dequantized_row,
            )?;
            reader.gemv_expert_plane_into(
                up_experts,
                expert,
                input,
                up,
                raw_row,
                dequantized_row,
            )?;
            swiglu_into(&gate[..intermediate], &up[..intermediate], activation)
                .map_err(|_| StreamedMoeError::BufferTooSmall)?;
            reader.gemv_expert_plane_into(
                down_experts,
                expert,
                &activation[..intermediate],
                expert_out,
                raw_row,
                dequantized_row,
            )?;
        }
        for index in 0..hidden {
            out[index] += top_weights[route] * expert_out[index];
        }
    }

    reader.gemv_rows_into(
        shared_gate,
        0,
        intermediate,
        input,
        gate,
        raw_row,
        dequantized_row,
    )?;
    reader.gemv_rows_into(
        shared_up,
        0,
        intermediate,
        input,
        up,
        raw_row,
        dequantized_row,
    )?;
    swiglu_into(&gate[..intermediate], &up[..intermediate], activation)
        .map_err(|_| StreamedMoeError::BufferTooSmall)?;
    reader.gemv_rows_into(
        shared_down,
        0,
        hidden,
        &activation[..intermediate],
        expert_out,
        raw_row,
        dequantized_row,
    )?;
    let mut shared_gate_logit = [0.0f32; 1];
    reader.gemv_rows_into(
        shared_gate_input,
        0,
        1,
        input,
        &mut shared_gate_logit,
        raw_row,
        dequantized_row,
    )?;
    let shared_scale = match profile.shared_expert {
        SharedExpertGate::SigmoidInputDotSameWidth => 1.0 / (1.0 + (-shared_gate_logit[0]).exp()),
    };
    for index in 0..hidden {
        out[index] += shared_scale * expert_out[index];
    }
    Ok(())
}
