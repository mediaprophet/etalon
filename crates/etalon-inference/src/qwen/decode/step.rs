use super::trace::push_trace;
use super::*;
/// Run one complete token step through the trained Qwen4Exp graph and return
/// the streamed vocabulary argmax.  Every prompt and generated token passes
/// the same path: embedding → optional PLE residual → per-layer HC/mixer/HC/
/// MoE → final Hyper-Connection mixer → `output.weight` argmax.
pub fn decode_step<H: super::super::QwenHyperparams>(
    runtime: &mut Qwen4ExpRuntime<'_, H>,
    session: &mut Qwen4ExpSession,
    scratch: &mut Qwen4ExpDecodeScratch,
    token: u32,
    trace: &mut impl Qwen4ExpTraceSink,
    capture: bool,
) -> Result<StreamedArgmax, Qwen4ExpDecodeError> {
    let index = runtime.index;
    let hyperparams: etalon_model::inference::ModelHyperparams = index.hyperparams.into();
    let hidden = session.hidden;
    let embedding = *index
        .token_embd_info()
        .ok_or(Qwen4ExpDecodeError::MissingTensor)?;
    let output_norm = index
        .tensor_info(b"output_hc_norm.weight")
        .ok_or(Qwen4ExpDecodeError::MissingTensor)?;
    let output_down = index
        .tensor_info(b"output_hc_down.weight")
        .ok_or(Qwen4ExpDecodeError::MissingTensor)?;
    let output_up = index
        .tensor_info(b"output_hc_up.weight")
        .ok_or(Qwen4ExpDecodeError::MissingTensor)?;
    let logits = *index
        .logits_projection_info()
        .ok_or(Qwen4ExpDecodeError::MissingTensor)?;
    if token as u64 >= embedding.dims[1] {
        return Err(Qwen4ExpDecodeError::MissingTensor);
    }

    runtime.trunk.read_row_into(
        &embedding,
        token as usize,
        &mut scratch.raw_row,
        &mut scratch.embedding[..hidden],
    )?;
    for stream in 0..QWEN4EXP_HYPER_STREAMS {
        let start = stream * hidden;
        session.residual[start..start + hidden].copy_from_slice(&scratch.embedding[..hidden]);
    }
    let step = session.position;
    push_trace(
        trace,
        step,
        QWEN4EXP_TRACE_META_LAYER,
        "embed",
        &session.residual,
        capture,
    );

    let rope_theta = hyperparams.effective_rope_freq_base();
    let ple_config = index.ple_config;
    for layer_idx in 0..session.n_layers as u32 {
        let layer = index.get_layer_tensors(layer_idx);
        if session.ple_layer == Some(layer_idx) {
            let config = ple_config.ok_or(Qwen4ExpDecodeError::MissingTensor)?;
            let mut ple_buffers = PleBlockBuffers {
                raw_ple_row: &mut scratch.ple_raw,
                raw_trunk_row: &mut scratch.raw_row,
                projection_row: &mut scratch.ple_projection,
                embedding: &mut scratch.ple_embedding,
                key: &mut scratch.ple_key,
                value: &mut scratch.ple_value,
                query_norm: &mut scratch.ple_query_norm,
                gated: &mut scratch.ple_gated,
                conv_norm: &mut scratch.ple_conv_norm,
                conv_weights: &mut scratch.ple_conv_weights,
                norm_key: &mut scratch.ple_norm_key,
                norm_query: &mut scratch.ple_norm_query,
                norm_conv: &mut scratch.ple_norm_conv,
            };
            execute_ple_block(
                &config,
                &layer,
                token,
                &mut session.ple_tokens,
                &mut runtime.ple,
                &mut runtime.trunk,
                &mut session.residual,
                &mut session.ple_conv_history,
                &mut ple_buffers,
            )?;
            push_trace(
                trace,
                step,
                layer_idx,
                "post_ple",
                &session.residual,
                capture,
            );
        }
        if layer.is_hybrid_ssm_layer() {
            let mut state = GatedDeltaState {
                convolution: &mut session.gdn_convolution[layer_idx as usize],
                delta: &mut session.gdn_delta[layer_idx as usize],
            };
            let mut buffers = StreamedGdnMoeLayerBuffers {
                attn_hyper: StreamedHyperBuffers {
                    raw_row: &mut scratch.attn_raw,
                    projection_row: &mut scratch.attn_hyper_projection,
                    norm_weight: &mut scratch.attn_hyper_norm,
                    normalized: &mut scratch.attn_hyper_normalized,
                    low_rank: &mut scratch.attn_hyper_low_rank,
                    gates: &mut scratch.attn_hyper_gates,
                },
                ffn_hyper: StreamedHyperBuffers {
                    raw_row: &mut scratch.ffn_raw,
                    projection_row: &mut scratch.ffn_hyper_projection,
                    norm_weight: &mut scratch.ffn_hyper_norm,
                    normalized: &mut scratch.ffn_hyper_normalized,
                    low_rank: &mut scratch.ffn_hyper_low_rank,
                    gates: &mut scratch.ffn_hyper_gates,
                },
                gdn: GatedDeltaBuffers {
                    raw_row: &mut scratch.gdn_raw,
                    projection_row: &mut scratch.gdn_projection,
                    qkv: &mut scratch.gdn_qkv,
                    gate: &mut scratch.gdn_gate,
                    alpha: &mut scratch.gdn_alpha,
                    beta: &mut scratch.gdn_beta,
                    convolved: &mut scratch.gdn_convolved,
                    head_norm: &mut scratch.gdn_head_norm,
                    head_vector: &mut scratch.gdn_head_vector,
                    output_inner: &mut scratch.gdn_output_inner,
                    head_a: &mut scratch.gdn_head_a,
                    dt_bias: &mut scratch.gdn_dt_bias,
                },
                moe: StreamedMoeBuffers {
                    raw_row: &mut scratch.moe_raw,
                    dequantized_row: &mut scratch.moe_dequantized,
                    router_logits: &mut scratch.moe_router_logits,
                    top_indices: &mut scratch.moe_top_indices,
                    top_logits: &mut scratch.moe_top_logits,
                    top_weights: &mut scratch.moe_top_weights,
                    gate: &mut scratch.moe_gate,
                    up: &mut scratch.moe_up,
                    activation: &mut scratch.moe_activation,
                    expert_out: &mut scratch.moe_expert_out,
                },
                attn_mixed: &mut scratch.attn_mixed,
                token_mixer_branch: &mut scratch.token_mixer_branch,
                ffn_mixed: &mut scratch.ffn_mixed,
                moe_branch: &mut scratch.moe_branch,
                attn_inject: &mut scratch.attn_inject,
                ffn_inject: &mut scratch.ffn_inject,
            };
            execute_streamed_gdn_moe_layer(
                &mut runtime.trunk,
                &layer,
                &mut session.residual,
                &mut state,
                &mut buffers,
            )?;
        } else if layer.has_qwen_sparse_attention() {
            let slot = layer_idx as usize;
            let mut state = QsaState {
                keys: &mut session.qsa_keys[slot],
                values: &mut session.qsa_values[slot],
                indexer_keys: &mut session.qsa_indexer_keys[slot],
                tokens: session.qsa_tokens[slot],
            };
            let mut buffers = StreamedQsaMoeLayerBuffers {
                attn_hyper: StreamedHyperBuffers {
                    raw_row: &mut scratch.attn_raw,
                    projection_row: &mut scratch.attn_hyper_projection,
                    norm_weight: &mut scratch.attn_hyper_norm,
                    normalized: &mut scratch.attn_hyper_normalized,
                    low_rank: &mut scratch.attn_hyper_low_rank,
                    gates: &mut scratch.attn_hyper_gates,
                },
                ffn_hyper: StreamedHyperBuffers {
                    raw_row: &mut scratch.ffn_raw,
                    projection_row: &mut scratch.ffn_hyper_projection,
                    norm_weight: &mut scratch.ffn_hyper_norm,
                    normalized: &mut scratch.ffn_hyper_normalized,
                    low_rank: &mut scratch.ffn_hyper_low_rank,
                    gates: &mut scratch.ffn_hyper_gates,
                },
                qsa: StreamedQsaBuffers {
                    raw_row: &mut scratch.qsa_raw,
                    projection_row: &mut scratch.qsa_projection,
                    q_full: &mut scratch.qsa_q_full,
                    key: &mut scratch.qsa_key,
                    value: &mut scratch.qsa_value,
                    index_query: &mut scratch.qsa_index_query,
                    index_key: &mut scratch.qsa_index_key,
                    q_norm: &mut scratch.qsa_q_norm,
                    k_norm: &mut scratch.qsa_k_norm,
                    index_q_norm: &mut scratch.qsa_index_q_norm,
                    index_k_norm: &mut scratch.qsa_index_k_norm,
                    selected: &mut scratch.qsa_selected,
                    scores: &mut scratch.qsa_scores,
                    block_selected: &mut scratch.qsa_block_selected,
                    block_scores: &mut scratch.qsa_block_scores,
                    attention: &mut scratch.qsa_attention,
                },
                moe: StreamedMoeBuffers {
                    raw_row: &mut scratch.moe_raw,
                    dequantized_row: &mut scratch.moe_dequantized,
                    router_logits: &mut scratch.moe_router_logits,
                    top_indices: &mut scratch.moe_top_indices,
                    top_logits: &mut scratch.moe_top_logits,
                    top_weights: &mut scratch.moe_top_weights,
                    gate: &mut scratch.moe_gate,
                    up: &mut scratch.moe_up,
                    activation: &mut scratch.moe_activation,
                    expert_out: &mut scratch.moe_expert_out,
                },
                attn_mixed: &mut scratch.attn_mixed,
                token_mixer_branch: &mut scratch.token_mixer_branch,
                ffn_mixed: &mut scratch.ffn_mixed,
                moe_branch: &mut scratch.moe_branch,
                attn_inject: &mut scratch.attn_inject,
                ffn_inject: &mut scratch.ffn_inject,
            };
            execute_streamed_qsa_moe_layer(
                &mut runtime.trunk,
                &layer,
                &mut session.residual,
                &mut state,
                rope_theta,
                &mut buffers,
            )?;
            session.qsa_tokens[slot] = state.tokens;
        } else {
            return Err(Qwen4ExpDecodeError::MissingTokenMixer { layer: layer_idx });
        }
        // The fused layer call leaves every intermediate in scratch, so the
        // trace sees each stage without the operators knowing about it.
        push_trace(
            trace,
            step,
            layer_idx,
            "attn_in",
            &scratch.attn_mixed,
            capture,
        );
        push_trace(
            trace,
            step,
            layer_idx,
            "mixer_out",
            &scratch.token_mixer_branch,
            capture,
        );
        push_trace(
            trace,
            step,
            layer_idx,
            "attn_inject",
            &scratch.attn_inject,
            capture,
        );
        push_trace(
            trace,
            step,
            layer_idx,
            "ffn_in",
            &scratch.ffn_mixed,
            capture,
        );
        push_trace(
            trace,
            step,
            layer_idx,
            "moe_out",
            &scratch.moe_branch,
            capture,
        );
        push_trace(
            trace,
            step,
            layer_idx,
            "ffn_inject",
            &scratch.ffn_inject,
            capture,
        );
        push_trace(
            trace,
            step,
            layer_idx,
            "post_layer",
            &session.residual,
            capture,
        );
    }

    execute_streamed_final_hyper_connection(
        &mut runtime.trunk,
        output_norm,
        output_down,
        output_up,
        &session.residual,
        &mut scratch.mixed[..hidden],
        &mut StreamedHyperBuffers {
            raw_row: &mut scratch.attn_raw,
            projection_row: &mut scratch.attn_hyper_projection,
            norm_weight: &mut scratch.attn_hyper_norm,
            normalized: &mut scratch.attn_hyper_normalized,
            low_rank: &mut scratch.attn_hyper_low_rank,
            gates: &mut scratch.attn_hyper_gates,
        },
    )?;
    push_trace(
        trace,
        step,
        QWEN4EXP_TRACE_META_LAYER,
        "final_mixed",
        &scratch.mixed[..hidden],
        capture,
    );
    let winner = runtime.trunk.argmax_rows(
        &logits,
        &scratch.mixed[..hidden],
        &mut scratch.raw_row,
        &mut scratch.argmax_row,
    )?;
    session.position += 1;
    Ok(winner)
}
