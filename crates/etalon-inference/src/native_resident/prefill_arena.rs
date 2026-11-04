use super::prefill_arena_plan::*;
use super::prefill_arena_slots::*;
use super::{ResidentEngine, ResidentProvider};
use crate::ordinary::{gpu_params::*, RMS_NORM_EPS};

impl<P: ResidentProvider> ResidentEngine<'_, P> {
    pub fn run_prefill_chunk(
        &self,
        plan: &mut PrefillArenaPlan,
        batch_hidden: &[f32],
        emb_dim: usize,
        n_tokens: u32,
        batch_start_token_idx: u32,
        max_layers: u32,
    ) -> Option<()> {
        let r = self.host.resident_resources();
        let b = n_tokens as usize;
        let n_embd = plan.n_embd;
        if emb_dim < n_embd
            || b == 0
            || b.checked_mul(n_embd)? > batch_hidden.len()
            || (b.checked_mul(n_embd)?.checked_mul(4)? as u64) > plan.hidden_a.size()
            || (batch_start_token_idx as usize).checked_add(b)? > plan.max_context as usize
        {
            return None;
        }
        let n_layer_total = plan.layers.len() as u32;
        let limit = if max_layers == 0 {
            n_layer_total
        } else {
            max_layers.min(n_layer_total)
        };
        let queue = self.host.gpu_queue();

        // 1) Upload the B embedded tokens (packed B × n_embd, contiguous rows).
        queue.write_buffer(
            &plan.hidden_a,
            0,
            bytemuck::cast_slice(&batch_hidden[..b * n_embd]),
        );

        // 2) Rewrite the param arena with this chunk's batch fields. `protos` (read) and
        //    `scratch` (write) are disjoint fields, so the split borrow is safe.
        let gp = std::mem::size_of::<GemmGpuParams>();
        let ap = std::mem::size_of::<AttentionGpuParams>();
        let ep = std::mem::size_of::<ElemGpuParams>();
        let bu = n_tokens; // B as u32
        {
            let scratch = &mut plan.scratch;
            for (l, proto) in plan.protos.iter().enumerate() {
                let base = l as u64 * SLOTS_PER_LAYER * SLOT;
                for (i, slot) in [
                    S_GEMM_K,
                    S_GEMM_V,
                    S_GEMM_Q,
                    S_GEMM_O,
                    S_GEMM_GATE,
                    S_GEMM_UP,
                    S_GEMM_DOWN,
                ]
                .into_iter()
                .enumerate()
                {
                    let mut g = proto.gemm[i];
                    g.n_batch = bu;
                    let off = (base + slot * SLOT) as usize;
                    scratch[off..off + gp].copy_from_slice(bytemuck::bytes_of(&g));
                }
                for (i, slot) in [S_ATTN_KW, S_ATTN_VW, S_ATTN_Q].into_iter().enumerate() {
                    let mut a = proto.attn[i];
                    a.num_tokens_in_batch = bu;
                    a.batch_start_token_idx = batch_start_token_idx;
                    a.token_idx = batch_start_token_idx;
                    let off = (base + slot * SLOT) as usize;
                    scratch[off..off + ap].copy_from_slice(bytemuck::bytes_of(&a));
                }
                for (i, slot) in [
                    S_ELEM_RMS_ATTN,
                    S_ELEM_ADD1,
                    S_ELEM_RMS_FFN,
                    S_ELEM_SILU,
                    S_ELEM_ADD2,
                ]
                .into_iter()
                .enumerate()
                {
                    let mut e = proto.elem[i];
                    e.batch = bu;
                    let off = (base + slot * SLOT) as usize;
                    scratch[off..off + ep].copy_from_slice(bytemuck::bytes_of(&e));
                }
            }
        }
        queue.write_buffer(&plan.param_arena, 0, &plan.scratch);

        // 3) Encode every layer's batched forward into ONE encoder.
        let mut encoder =
            self.host
                .gpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ResidentPrefillEncoder"),
                });
        let gemv = if plan.use_coop {
            r.coop_gemv_pipeline
        } else {
            r.pipeline
        };
        let gemv_resid = if plan.use_coop {
            r.coop_gemv_residual_pipeline
        } else {
            r.pipeline
        };
        let rms = r.rms_pipeline?;
        let silu = r.elem_silu_mul_pipeline;
        let attn = r.attention_pipeline;
        let fused_pipe = if plan.use_coop {
            r.ffn_fused_coop_pipeline
        } else {
            r.ffn_fused_pipeline
        };

        let gemv_wg = |n_out: u32| {
            if plan.use_coop {
                n_out // prefill keeps single-row coop (batch axis is y)
            } else {
                n_out.div_ceil(64)
            }
        };
        let elem_wg = |n: u32| n.div_ceil(64);
        let (n_embd_u, n_ffn_u, kv_dim_u) = (n_embd as u32, plan.n_ffn as u32, plan.kv_dim as u32);
        let q_dim_u = plan.q_dim as u32;
        let kv_pairs = plan.n_kv_head.max(1) * bu;

        // ONE compute pass for the whole chunk (all layers). Previously each of the
        // 15 ops/layer opened its own begin_compute_pass — ~420 driver passes/chunk
        // and catastrophic TTFT. Mirror resident_decode: one pass, many dispatches.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ResidentPrefillChunk"),
                timestamp_writes: None,
            });
            for lb in plan.layers.iter().take(limit as usize) {
                // ~11–13 dispatches/layer with residual-fused O/down (+ optional fused FFN).
                let dispatch = |cpass: &mut wgpu::ComputePass,
                                pipe: &wgpu::ComputePipeline,
                                bg: &wgpu::BindGroup,
                                wg_x: u32,
                                wg_y: u32| {
                    cpass.set_pipeline(pipe);
                    cpass.set_bind_group(0, bg, &[]);
                    cpass.dispatch_workgroups(wg_x, wg_y, 1);
                };
                dispatch(&mut cpass, rms, &lb.rms_attn, 1, bu);
                if let Some(ref dual) = lb.dual_kv {
                    // Match decode: dual multi-row is opt-in (A/B lost on A2000).
                    let dual_mr = self.host.dual_multirow();
                    const DUAL_ROWS: u32 = 4;
                    if dual_mr {
                        dispatch(
                            &mut cpass,
                            r.dual_gemv_mr_pipeline,
                            dual,
                            gemv_wg(kv_dim_u).div_ceil(DUAL_ROWS).max(1),
                            bu,
                        );
                    } else {
                        dispatch(
                            &mut cpass,
                            r.dual_gemv_pipeline,
                            dual,
                            gemv_wg(kv_dim_u),
                            bu,
                        );
                    }
                } else {
                    dispatch(&mut cpass, gemv, &lb.k_gemm, gemv_wg(kv_dim_u), bu);
                    dispatch(&mut cpass, gemv, &lb.v_gemm, gemv_wg(kv_dim_u), bu);
                }
                dispatch(&mut cpass, attn, &lb.k_write, kv_pairs, 1);
                dispatch(&mut cpass, attn, &lb.v_write, kv_pairs, 1);
                dispatch(&mut cpass, gemv, &lb.q_gemm, gemv_wg(q_dim_u), bu);
                dispatch(&mut cpass, attn, &lb.q, plan.n_head.max(1), bu);
                // O·attn + residual(hidden_a) → hidden_b
                dispatch(&mut cpass, gemv_resid, &lb.o_resid, gemv_wg(n_embd_u), bu);
                dispatch(&mut cpass, rms, &lb.rms_ffn, 1, bu);
                if let Some(ref fbg) = lb.fused_ffn {
                    dispatch(&mut cpass, fused_pipe, fbg, gemv_wg(n_ffn_u), bu);
                } else {
                    dispatch(&mut cpass, gemv, &lb.gate, gemv_wg(n_ffn_u), bu);
                    dispatch(&mut cpass, gemv, &lb.up, gemv_wg(n_ffn_u), bu);
                    dispatch(&mut cpass, silu, &lb.silu, elem_wg(n_ffn_u), bu);
                }
                // Down·mid + residual(hidden_b) → hidden_a
                dispatch(
                    &mut cpass,
                    gemv_resid,
                    &lb.down_resid,
                    gemv_wg(n_embd_u),
                    bu,
                );
            }
        }

        // ONE submit, ONE fence — the KV cache is populated as a side effect.
        if !self.host.submit(encoder.finish()) {
            return None;
        }
        if !self.host.poll_wait() {
            return None;
        }
        Some(())
    }
}
