use super::{experts::*, LayerProvider};
use crate::moe::dispatch::MAX_ROUTED_EXPERTS;
use etalon_model::inference::LayerTensors;
/// Dispatch MoE layer step: pre-norm input -> router -> top-k SwiGLU experts into scratch_a.
pub fn execute_moe<P: LayerProvider>(
    host: &P,
    index: &P::Index,
    emb_dim: usize,
    tensors: &LayerTensors,
    scratch_a: &mut [f32],
    ffn_input: &[f32],
) -> bool {
    let router_info = match tensors.moe_router.as_ref() {
        Some(r) => r,
        None => return false,
    };

    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    let layer_idx = tensors.layer_idx as u16;
    if router_info.n_dims != 2
        || router_info.dims[0] != emb_dim as u64
        || emb_dim == 0
        || emb_dim > 8192
        || scratch_a.len() < emb_dim
        || ffn_input.len() < emb_dim
    {
        return false;
    }
    let Ok(actual_experts) = usize::try_from(router_info.dims[1]) else {
        return false;
    };
    let Ok(routing) = P::moe_routing(index).admit(actual_experts) else {
        return false;
    };
    let num_experts = routing.expert_count();
    let topk = routing.experts_per_token();
    if num_experts > MAX_ROUTED_EXPERTS {
        return false;
    }

    let mut gate_logits = [0.0f32; MAX_ROUTED_EXPERTS];
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut router_ready = false;

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(pkg) = host.expert_source() {
            let router_bytes = pkg
                .layer_tensor(layer_idx, b"mlp.gate.weight")
                .or_else(|| pkg.canonical_layer_tensor(layer_idx, b"ffn_gate_inp.weight"));

            if let Some(raw) = router_bytes {
                compute_router_logits_from_raw(
                    raw,
                    router_info.ggml_type,
                    &ffn_input[..emb_dim],
                    emb_dim,
                    num_experts,
                    &mut gate_logits[..num_experts],
                );
                router_ready = true;
            } else if let Some(mmap) = host.model_bytes() {
                let off = router_info.byte_offset as usize;
                if off < mmap.len() {
                    compute_router_logits_from_raw(
                        &mmap[off..],
                        router_info.ggml_type,
                        &ffn_input[..emb_dim],
                        emb_dim,
                        num_experts,
                        &mut gate_logits[..num_experts],
                    );
                    router_ready = true;
                }
            }
        }
    }

    // Ordinary GGUF expert tensors are 3-D ([input, intermediate, expert]).
    // Their rows must be addressed through the tensor-data base, not the
    // raw `byte_offset` (which is relative to that base).  This also keeps
    // Q4_K/Q6_K router rows on the canonical dequantization path.
    if !router_ready {
        let Some(mmap) = host.model_bytes() else {
            return false;
        };
        if let Ok(raw) =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::data_start(index), router_info)
        {
            if !compute_router_logits_from_gguf(
                raw,
                router_info,
                &ffn_input[..emb_dim],
                num_experts,
                &mut gate_logits[..num_experts],
            ) {
                return false;
            }
        } else {
            return false;
        }
    }

    let mut expert_indices = [0u16; crate::moe::routing::MAX_MOE_TOPK];
    let mut expert_weights = [0.0f32; crate::moe::routing::MAX_MOE_TOPK];
    let selected = match crate::moe::routing::route_topk(
        &gate_logits[..num_experts],
        topk,
        true,
        &mut expert_indices,
        &mut expert_weights,
    ) {
        Ok(s) => s,
        Err(_) => return false,
    };

    for x in scratch_a[..emb_dim].iter_mut() {
        *x = 0.0;
    }

    let mut gate_buf = [0.0f32; crate::moe::dispatch::MAX_INTERMEDIATE_DIM];
    let mut up_buf = [0.0f32; crate::moe::dispatch::MAX_INTERMEDIATE_DIM];
    let mut swiglu_buf = [0.0f32; crate::moe::dispatch::MAX_INTERMEDIATE_DIM];
    let mut expert_out = [0.0f32; 8192];

    #[cfg(not(target_arch = "wasm32"))]
    let ftw_pkg = host.expert_source();
    let mmap = host.model_bytes();

    let direct_gate = tensors.moe_gate_exps.as_ref();
    let direct_up = tensors.moe_up_exps.as_ref();
    let direct_gate_up = tensors.moe_gate_up_exps.as_ref();
    let direct_down = tensors.moe_down_exps.as_ref();

    for i in 0..selected {
        let expert_id = expert_indices[i];
        let weight = expert_weights[i];

        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut evaluated = false;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(pkg) = ftw_pkg {
            if let Some(data) = pkg.get_expert_data(layer_idx, expert_id) {
                // Derive intermediate_dim from actual tensor geometry.
                // gate_up_packed is gate+up concatenated: len = 2 * inter_dim * (emb_dim/2)
                // so inter_dim = len / emb_dim
                let bytes_per_row = emb_dim / 2; // NVFP4: 2 nibbles/byte
                let inter_dim = if bytes_per_row > 0 {
                    data.gate_up_packed.len() / (2 * bytes_per_row)
                } else {
                    0
                };
                let view = data.to_view(emb_dim, inter_dim);
                if crate::moe::dispatch::evaluate_swiglu_expert_nvfp4(
                    &ffn_input[..emb_dim],
                    &view,
                    &mut gate_buf,
                    &mut up_buf,
                    &mut swiglu_buf,
                    &mut expert_out[..emb_dim],
                )
                .is_ok()
                {
                    for d in 0..emb_dim {
                        scratch_a[d] += weight * expert_out[d];
                    }
                    evaluated = true;
                }
            }
        }
        if !evaluated {
            let Some(mmap) = mmap.as_deref() else {
                return false;
            };
            let direct_ok = match (direct_gate, direct_up, direct_gate_up, direct_down) {
                (Some(gate), Some(up), _, Some(down)) => evaluate_gguf_swiglu_expert(
                    mmap,
                    P::data_start(index),
                    gate,
                    up,
                    down,
                    expert_id as usize,
                    &ffn_input[..emb_dim],
                    &mut gate_buf,
                    &mut up_buf,
                    &mut swiglu_buf,
                    &mut expert_out[..emb_dim],
                ),
                (_, _, Some(gate_up), Some(down)) => evaluate_gguf_swiglu_expert_fused_gate_up(
                    mmap,
                    P::data_start(index),
                    gate_up,
                    down,
                    expert_id as usize,
                    &ffn_input[..emb_dim],
                    &mut gate_buf,
                    &mut up_buf,
                    &mut swiglu_buf,
                    &mut expert_out[..emb_dim],
                ),
                _ => false,
            };
            if !direct_ok {
                return false;
            }
            for d in 0..emb_dim {
                scratch_a[d] += weight * expert_out[d];
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(pkg) = ftw_pkg {
        if let (Some(gup), Some(gus), Some(gug), Some(dnp), Some(dns), Some(dng), Some(gate_raw)) = (
            pkg.layer_tensor(layer_idx, b"mlp.shared_expert.gate_up_proj.weight"),
            pkg.layer_tensor(layer_idx, b"mlp.shared_expert.gate_up_proj.weight_scale"),
            pkg.layer_tensor(layer_idx, b"mlp.shared_expert.gate_up_proj.weight_global"),
            pkg.layer_tensor(layer_idx, b"mlp.shared_expert.down_proj.weight"),
            pkg.layer_tensor(layer_idx, b"mlp.shared_expert.down_proj.weight_scale"),
            pkg.layer_tensor(layer_idx, b"mlp.shared_expert.down_proj.weight_global"),
            pkg.layer_tensor(layer_idx, b"mlp.shared_expert_gate.weight"),
        ) {
            let shared_data = super::expert_source::FtwExpertData {
                gate_up_packed: gup,
                gate_up_scale: gus,
                gate_up_global: gug,
                down_packed: dnp,
                down_scale: dns,
                down_global: dng,
            };
            let bytes_per_row = emb_dim / 2;
            let inter_dim = if bytes_per_row > 0 {
                shared_data.gate_up_packed.len() / (2 * bytes_per_row)
            } else {
                0
            };
            let view = shared_data.to_view(emb_dim, inter_dim);
            if crate::moe::dispatch::evaluate_swiglu_expert_nvfp4(
                &ffn_input[..emb_dim],
                &view,
                &mut gate_buf,
                &mut up_buf,
                &mut swiglu_buf,
                &mut expert_out[..emb_dim],
            )
            .is_ok()
            {
                let mut gate_row = [0.0f32; 8192];
                let _ = etalon_model::ggml::dequantize_row_into(
                    gate_raw,
                    etalon_model::ggml::GGML_TYPE_BF16,
                    emb_dim,
                    &mut gate_row[..emb_dim],
                );
                let gate_val =
                    1.0 / (1.0 + (-dot(&ffn_input[..emb_dim], &gate_row[..emb_dim])).exp());
                for d in 0..emb_dim {
                    scratch_a[d] += gate_val * expert_out[d];
                }
            }
        }
    }

    // Qwen-style shared expert: an ordinary SwiGLU MLP whose output is
    // scaled by sigmoid(W_shared_gate x). It is mandatory whenever any
    // of its tensors are present; partial tensor sets fail closed.
    #[cfg(target_arch = "wasm32")]
    let has_ftw = false;
    #[cfg(not(target_arch = "wasm32"))]
    let has_ftw = ftw_pkg.is_some();

    if !has_ftw
        && (tensors.moe_shared_gate.is_some()
            || tensors.moe_shared_up.is_some()
            || tensors.moe_shared_down.is_some()
            || tensors.moe_shared_gate_input.is_some())
    {
        let (Some(gate), Some(up), Some(down), Some(gate_input), Some(mmap)) = (
            tensors.moe_shared_gate.as_ref(),
            tensors.moe_shared_up.as_ref(),
            tensors.moe_shared_down.as_ref(),
            tensors.moe_shared_gate_input.as_ref(),
            mmap.as_deref(),
        ) else {
            return false;
        };
        if !evaluate_gguf_dense_swiglu(
            mmap,
            P::data_start(index),
            gate,
            up,
            down,
            &ffn_input[..emb_dim],
            &mut gate_buf,
            &mut up_buf,
            &mut swiglu_buf,
            &mut expert_out[..emb_dim],
        ) {
            return false;
        }
        let Ok(gate_raw) =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::data_start(index), gate_input)
        else {
            return false;
        };
        let mut gate_row = [0.0f32; crate::moe::dispatch::MAX_INTERMEDIATE_DIM];
        if gate_input.n_dims != 1
            || gate_input.dims[0] as usize != emb_dim
            || emb_dim > gate_row.len()
            || etalon_model::ggml::dequantize_row_into(
                gate_raw,
                gate_input.ggml_type,
                emb_dim,
                &mut gate_row[..emb_dim],
            )
            .is_err()
        {
            return false;
        }
        let gate_weight = 1.0 / (1.0 + (-dot(&ffn_input[..emb_dim], &gate_row[..emb_dim])).exp());
        for d in 0..emb_dim {
            scratch_a[d] += gate_weight * expert_out[d];
        }
    }

    true
}
