use super::resident_decode_plan::*;
use super::resident_decode_slots::*;
use super::{ResidentEngine, ResidentProvider};
use crate::ordinary::{gpu_params::*, RMS_NORM_EPS};

impl<P: ResidentProvider> ResidentEngine<'_, P> {
    pub fn run_resident_token(
        &self,
        plan: &mut ResidentDecodePlan,
        emb: &[f32],
        token_idx: u32,
        out_hidden: Option<&mut [f32]>,
    ) -> Option<ResidentTokenOutcome> {
        let r = self.host.resident_resources();
        let n_embd = plan.n_embd;
        if emb.len() < n_embd
            || token_idx >= plan.max_context
            || out_hidden.as_ref().is_some_and(|out| out.len() < n_embd)
        {
            return None;
        }
        let queue = self.host.gpu_queue();

        // 1) Per-token uploads: embedding + dynamic attention params.
        // Skip full KV-mask upload when route mask is inactive (common path).
        queue.write_buffer(&plan.hidden_a, 0, bytemuck::cast_slice(&emb[..n_embd]));

        let (mask_words, mask_active) = self.host.attention_kv_mask(token_idx, plan.max_context);
        if mask_active != 0 {
            queue.write_buffer(
                r.attention_mask_buf?,
                0,
                bytemuck::cast_slice(mask_words.as_ref()),
            );
        }

        let ap_size = std::mem::size_of::<AttentionGpuParams>();
        for (l, protos) in plan.layer_protos.iter().enumerate() {
            let base = l as u64 * ATTN_SLOTS_PER_LAYER * SLOT;
            let mut k = protos.k_write;
            k.token_idx = token_idx;
            k.batch_start_token_idx = token_idx;
            let mut v = protos.v_write;
            v.token_idx = token_idx;
            v.batch_start_token_idx = token_idx;
            let mut q = protos.q;
            q.token_idx = token_idx;
            q.batch_start_token_idx = token_idx;
            q.mask_active = mask_active;
            q.mask_word_count = if mask_active != 0 {
                mask_words.as_ref().len() as u32
            } else {
                0
            };
            for (slot, p) in [(0u64, &k), (1, &v), (2, &q)] {
                let off = (base + slot * SLOT) as usize;
                plan.dyn_scratch[off..off + ap_size].copy_from_slice(bytemuck::bytes_of(p));
            }
        }
        queue.write_buffer(&plan.attn_dyn_arena, 0, &plan.dyn_scratch);

        // 2) Encode the WHOLE token in ONE compute pass: all layers + output norm + logits.
        // Previously: 15 passes/layer × 28–32 layers + norm + per-chunk logits ≈ 450+ passes.
        // Then: 1 pass/layer. Now: **one pass for the entire forward**. Driver overhead collapses.
        let mut encoder =
            self.host
                .gpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ResidentTokenEncoder"),
                });
        let rms = r.rms_pipeline?;
        let silu = r.elem_silu_mul_pipeline;
        let attn = r.attention_pipeline;
        let sample_path = out_hidden.is_some();
        let topk = if sample_path {
            None
        } else {
            Some(r.output_topk_pipeline?)
        };

        // Projection GEMV (Q, K/V fallback): always 1-row/WG.
        // Residual O/down + logits: multi-row when plan.use_multirow.
        let gemv_proj_wg = |n_out: u32| {
            if plan.use_coop {
                n_out
            } else {
                n_out.div_ceil(64)
            }
        };
        let gemv_large_wg = |n_out: u32| {
            if plan.use_coop && plan.use_multirow {
                self.host.coop_gemv_workgroups(n_out)
            } else if plan.use_coop {
                n_out
            } else {
                n_out.div_ceil(64)
            }
        };
        let elem_wg = |n: u32| n.div_ceil(64);
        let (n_embd_u, n_ffn_u) = (n_embd as u32, plan.n_ffn as u32);
        let kv_dim_u = plan.kv_dim as u32;
        let q_dim_u = plan.q_dim as u32;

        {
            // Lab L1.1: optional TIMESTAMP_QUERY around the mega-pass (FusedBlock phase).
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ResidentFullToken"),
                timestamp_writes: self.host.pass_writes(),
            });
            // Fused FFN: multi-row (4 rows/WG) for Q4 SoA; else coop 256; else naive.
            const FFN_MR_ROWS: u32 = 4;
            let (fused_pipe, fused_wg) = if plan.use_coop && plan.use_ffn_mr {
                (
                    r.ffn_fused_mr_pipeline,
                    n_ffn_u.div_ceil(FFN_MR_ROWS).max(1),
                )
            } else if plan.use_coop && plan.use_ffn_warp {
                (r.ffn_fused_warp_pipeline, n_ffn_u)
            } else if plan.use_coop {
                (r.ffn_fused_coop_pipeline, n_ffn_u)
            } else {
                (r.ffn_fused_pipeline, n_ffn_u.div_ceil(64))
            };
            let gemv_proj = if plan.use_coop {
                r.coop_gemv_pipeline
            } else {
                r.pipeline
            };
            let gemv_large = if plan.use_coop && plan.use_multirow {
                r.coop_gemv_mr_pipeline
            } else if plan.use_coop && plan.use_warp {
                r.coop_gemv_warp_pipeline
            } else if plan.use_coop {
                r.coop_gemv_pipeline
            } else {
                r.pipeline
            };
            let gemv_resid = if plan.use_multirow {
                r.coop_gemv_residual_mr_pipeline
            } else if plan.use_warp {
                r.coop_gemv_residual_warp_pipeline
            } else {
                r.coop_gemv_residual_pipeline
            };
            for lb in &plan.layers {
                // RMS attn
                cpass.set_pipeline(rms);
                cpass.set_bind_group(0, &lb.rms_attn, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
                // Triple Q+K+V (shared act) or dual K+V + Q or split.
                if let Some(ref tri) = lb.triple_qkv {
                    cpass.set_pipeline(r.triple_gemv_pipeline);
                    cpass.set_bind_group(0, tri, &[]);
                    cpass.dispatch_workgroups(q_dim_u.max(1), 1, 1);
                    cpass.set_pipeline(attn);
                    cpass.set_bind_group(0, &lb.k_write, &[]);
                    cpass.dispatch_workgroups(plan.n_kv_head.max(1), 1, 1);
                    cpass.set_pipeline(attn);
                    cpass.set_bind_group(0, &lb.v_write, &[]);
                    cpass.dispatch_workgroups(plan.n_kv_head.max(1), 1, 1);
                } else if let Some(ref dual) = lb.dual_kv {
                    // Dual multi-row (4 rows/WG): opt-in. A/B on A2000 3B lost vs 1-row dual
                    // (~8.75 vs ~9.0). QUALIA_LLM_DUAL_MR=1 to force.
                    let dual_mr = self.host.dual_multirow();
                    const DUAL_ROWS: u32 = 4;
                    if dual_mr {
                        cpass.set_pipeline(r.dual_gemv_mr_pipeline);
                        cpass.set_bind_group(0, dual, &[]);
                        cpass.dispatch_workgroups(kv_dim_u.div_ceil(DUAL_ROWS).max(1), 1, 1);
                    } else {
                        cpass.set_pipeline(r.dual_gemv_pipeline);
                        cpass.set_bind_group(0, dual, &[]);
                        cpass.dispatch_workgroups(kv_dim_u.max(1), 1, 1);
                    }
                    cpass.set_pipeline(attn);
                    cpass.set_bind_group(0, &lb.k_write, &[]);
                    cpass.dispatch_workgroups(plan.n_kv_head.max(1), 1, 1);
                    cpass.set_pipeline(attn);
                    cpass.set_bind_group(0, &lb.v_write, &[]);
                    cpass.dispatch_workgroups(plan.n_kv_head.max(1), 1, 1);
                    cpass.set_pipeline(gemv_proj);
                    cpass.set_bind_group(0, &lb.q_gemm, &[]);
                    cpass.dispatch_workgroups(gemv_proj_wg(q_dim_u), 1, 1);
                } else {
                    cpass.set_pipeline(gemv_proj);
                    cpass.set_bind_group(0, &lb.k_gemm, &[]);
                    cpass.dispatch_workgroups(gemv_proj_wg(kv_dim_u), 1, 1);
                    cpass.set_pipeline(attn);
                    cpass.set_bind_group(0, &lb.k_write, &[]);
                    cpass.dispatch_workgroups(plan.n_kv_head.max(1), 1, 1);
                    cpass.set_pipeline(gemv_proj);
                    cpass.set_bind_group(0, &lb.v_gemm, &[]);
                    cpass.dispatch_workgroups(gemv_proj_wg(kv_dim_u), 1, 1);
                    cpass.set_pipeline(attn);
                    cpass.set_bind_group(0, &lb.v_write, &[]);
                    cpass.dispatch_workgroups(plan.n_kv_head.max(1), 1, 1);
                    cpass.set_pipeline(gemv_proj);
                    cpass.set_bind_group(0, &lb.q_gemm, &[]);
                    cpass.dispatch_workgroups(gemv_proj_wg(q_dim_u), 1, 1);
                }
                // SDPA + O resid + RMS ffn
                cpass.set_pipeline(attn);
                cpass.set_bind_group(0, &lb.q, &[]);
                cpass.dispatch_workgroups(plan.n_head.max(1), 1, 1);
                cpass.set_pipeline(gemv_resid);
                cpass.set_bind_group(0, &lb.o_resid, &[]);
                cpass.dispatch_workgroups(gemv_large_wg(n_embd_u), 1, 1);
                cpass.set_pipeline(rms);
                cpass.set_bind_group(0, &lb.rms_ffn, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
                // T-A1 FFN expansion: fused_ffn OR gate+up+silu.
                if let Some(ref fbg) = lb.fused_ffn {
                    cpass.set_pipeline(fused_pipe);
                    cpass.set_bind_group(0, fbg, &[]);
                    cpass.dispatch_workgroups(fused_wg, 1, 1);
                } else if let (Some(g), Some(u), Some(s)) =
                    (lb.gate.as_ref(), lb.up.as_ref(), lb.silu.as_ref())
                {
                    for (pipe, bg, wg_x) in [
                        (gemv_large, g, gemv_large_wg(n_ffn_u)),
                        (gemv_large, u, gemv_large_wg(n_ffn_u)),
                        (silu, s, elem_wg(n_ffn_u)),
                    ] {
                        cpass.set_pipeline(pipe);
                        cpass.set_bind_group(0, bg, &[]);
                        cpass.dispatch_workgroups(wg_x, 1, 1);
                    }
                }
                // Down + residual fused. Multi-row opt-in only (A/B lost on A2000 3B).
                cpass.set_pipeline(gemv_resid);
                cpass.set_bind_group(0, &lb.down_resid, &[]);
                cpass.dispatch_workgroups(gemv_large_wg(n_embd_u), 1, 1);
            }
            // Output RMSNorm → `plan.normed`.
            cpass.set_pipeline(rms);
            cpass.set_bind_group(0, &plan.rms_out, &[]);
            cpass.dispatch_workgroups(1, 1, 1);

            // Logits + topk (E3: one full-vocab chunk when cap ≥ vocab).
            // Device maxComputeWorkgroupsPerDimension is typically 65535 — for
            // vocab > 60k always use multi-row GEMV (8 rows/WG) even if residual
            // multirow is off, so a single 128k dispatch stays legal.
            if !sample_path {
                let topk = topk?;
                let logits_mr = plan.use_coop && plan.out_chunks.iter().any(|c| c.rows > 60_000);
                let logits_pipe = if logits_mr {
                    r.coop_gemv_mr_pipeline
                } else {
                    gemv_large
                };
                let logits_wg = |rows: u32| {
                    if logits_mr {
                        self.host.coop_gemv_workgroups(rows)
                    } else {
                        gemv_large_wg(rows)
                    }
                };
                for chunk in &plan.out_chunks {
                    cpass.set_pipeline(logits_pipe);
                    cpass.set_bind_group(0, &chunk.gemm, &[]);
                    cpass.dispatch_workgroups(logits_wg(chunk.rows), 1, 1);
                    cpass.set_pipeline(topk);
                    cpass.set_bind_group(0, &chunk.topk, &[]);
                    cpass.dispatch_workgroups(chunk.cand_count as u32, 1, 1);
                }
            }
            drop(cpass);
            self.host.resolve(&mut encoder);

            if sample_path {
                let hb = (n_embd * 4) as wgpu::BufferAddress;
                encoder.copy_buffer_to_buffer(&plan.normed, 0, &plan.hidden_staging, 0, hb);
            } else {
                let cand_val = r.topk_cand_val_buf?;
                let cand_idx = r.topk_cand_idx_buf?;
                // One copy of the full candidate pack (cand_base laid them out contiguously).
                let cand_bytes = (plan.total_cands * 4) as wgpu::BufferAddress;
                encoder.copy_buffer_to_buffer(cand_val, 0, &plan.staging, 0, cand_bytes);
                encoder.copy_buffer_to_buffer(cand_idx, 0, &plan.staging, cand_bytes, cand_bytes);
            }
        }

        // 3) ONE submit, ONE fence, tiny readback.
        if !self.host.submit(encoder.finish()) {
            return None;
        }
        self.host.accumulate_fused();

        if let Some(out) = out_hidden {
            if out.len() < n_embd {
                return None;
            }
            let map_bytes = (n_embd * 4) as wgpu::BufferAddress;
            let _mapping =
                super::map_read(self.host, &plan.hidden_staging, map_bytes, &plan.map_status)?;
            let slice = plan.hidden_staging.slice(..map_bytes);
            {
                let data = slice
                    .get_mapped_range()
                    .expect("wgpu buffer map_range failed");
                let floats: &[f32] = bytemuck::cast_slice(&data[..n_embd * 4]);
                out[..n_embd].copy_from_slice(floats);
            }

            return Some(ResidentTokenOutcome::HiddenReady);
        }

        let map_bytes = (plan.total_cands * 8) as wgpu::BufferAddress;
        let _mapping = super::map_read(self.host, &plan.staging, map_bytes, &plan.map_status)?;
        let slice = plan.staging.slice(..map_bytes);

        let mut best_token_id = 0u32;
        let mut max_logit = f32::NEG_INFINITY;
        {
            let data = slice
                .get_mapped_range()
                .expect("wgpu buffer map_range failed");
            let val_bytes = plan.total_cands * 4;
            let vals: &[f32] = bytemuck::cast_slice(&data[..val_bytes]);
            let idxs: &[u32] = bytemuck::cast_slice(&data[val_bytes..val_bytes * 2]);
            let mut offset = 0usize;
            let mut row_start = 0u32;
            for chunk in &plan.out_chunks {
                for i in 0..chunk.cand_count {
                    let pos = offset + i;
                    let v = vals[pos];
                    let token_id = row_start + idxs[pos];
                    if v > f32::NEG_INFINITY
                        && (v > max_logit || (v == max_logit && token_id < best_token_id))
                    {
                        max_logit = v;
                        best_token_id = token_id;
                    }
                }
                offset += chunk.cand_count;
                row_start += chunk.rows;
            }
        }

        if max_logit == f32::NEG_INFINITY {
            None
        } else {
            Some(ResidentTokenOutcome::Argmax(ResidentArgmax {
                best_token_id,
                max_logit,
            }))
        }
    }
}
