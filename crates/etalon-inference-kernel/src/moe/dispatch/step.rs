use super::super::routing::{route_topk, MAX_MOE_TOPK};
use super::{
    evaluate_swiglu_expert_nvfp4, ExpertWeightView, MoeDispatchError, MAX_INTERMEDIATE_DIM,
    MAX_ROUTED_EXPERTS,
};

/// Dispatch MoE layer step:
/// 1. Compute router logits
/// 2. Select top-k experts
/// 3. Compute SwiGLU for each selected expert and accumulate weighted output
// Preserve the inherited nine-argument compatibility API.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_moe_step<F>(
    input: &[f32],
    emb_dim: usize,
    router_weights: &[f32], // [num_experts, emb_dim]
    num_experts: usize,
    topk: usize,
    mut expert_fetcher: F,
    shared_expert: Option<&ExpertWeightView<'_>>,
    shared_gate_weight: f32,
    scratch_accum: &mut [f32],
) -> Result<(), MoeDispatchError>
where
    F: FnMut(u16) -> Option<ExpertWeightView<'static>>,
{
    if input.len() < emb_dim || scratch_accum.len() < emb_dim {
        return Err(MoeDispatchError::InputOrAccumulatorTooSmall);
    }
    if num_experts > MAX_ROUTED_EXPERTS {
        return Err(MoeDispatchError::TooManyExperts);
    }

    // 1. Router logits: g_e = input . W_router[e]
    let mut gate_logits = [0.0f32; MAX_ROUTED_EXPERTS];
    for e in 0..num_experts {
        let mut dot = 0.0f32;
        let w_row = &router_weights[e * emb_dim..(e + 1) * emb_dim];
        for i in 0..emb_dim {
            dot += input[i] * w_row[i];
        }
        gate_logits[e] = dot;
    }

    // 2. Route top-k
    let mut expert_indices = [0u16; MAX_MOE_TOPK];
    let mut expert_weights = [0.0f32; MAX_MOE_TOPK];
    let selected = route_topk(
        &gate_logits[..num_experts],
        topk,
        true,
        &mut expert_indices,
        &mut expert_weights,
    )
    .map_err(MoeDispatchError::Routing)?;

    // Clear accumulator
    for x in scratch_accum[..emb_dim].iter_mut() {
        *x = 0.0;
    }

    // Scratch buffers for intermediate SwiGLU
    let mut gate_buf = [0.0f32; MAX_INTERMEDIATE_DIM];
    let mut up_buf = [0.0f32; MAX_INTERMEDIATE_DIM];
    let mut swiglu_buf = [0.0f32; MAX_INTERMEDIATE_DIM];
    let mut expert_out = [0.0f32; 8192];

    // 3. Accumulate routed experts
    for i in 0..selected {
        let expert_id = expert_indices[i];
        let weight = expert_weights[i];

        if let Some(view) = expert_fetcher(expert_id) {
            evaluate_swiglu_expert_nvfp4(
                input,
                &view,
                &mut gate_buf,
                &mut up_buf,
                &mut swiglu_buf,
                &mut expert_out[..emb_dim],
            )
            .map_err(MoeDispatchError::RoutedExpert)?;

            for d in 0..emb_dim {
                scratch_accum[d] += weight * expert_out[d];
            }
        }
    }

    // 4. Evaluate shared expert if present
    if let Some(shared) = shared_expert {
        evaluate_swiglu_expert_nvfp4(
            input,
            shared,
            &mut gate_buf,
            &mut up_buf,
            &mut swiglu_buf,
            &mut expert_out[..emb_dim],
        )
        .map_err(MoeDispatchError::SharedExpert)?;

        for d in 0..emb_dim {
            scratch_accum[d] += shared_gate_weight * expert_out[d];
        }
    }

    Ok(())
}
