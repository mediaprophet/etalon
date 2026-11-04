use super::verify_arena_plan::*;
use super::verify_arena_slots::*;
use super::{ResidentEngine, ResidentProvider};
use crate::ordinary::{gpu_params::*, RMS_NORM_EPS};

impl<P: ResidentProvider> ResidentEngine<'_, P> {
    pub fn run_verify_batch(
        &self,
        plan: &mut VerifyArenaPlan,
        b: usize,
        start_pos: u32,
        out_argmax: &mut [u32],
        out_logit: &mut [f32],
    ) -> Option<()> {
        let r = self.host.resident_resources();
        let n_embd = plan.n_embd;
        let emb = &plan.embedding_scratch;
        let vocab = plan.vocab;
        if (start_pos as usize).checked_add(b)? > plan.max_context as usize {
            return None;
        }
        if b == 0
            || out_argmax.len() < b
            || out_logit.len() < b
            || emb.len() < b.checked_mul(n_embd)?
        {
            return None;
        }
        let queue = self.host.gpu_queue();
        queue.write_buffer(&plan.hidden_a, 0, bytemuck::cast_slice(&emb[..b * n_embd]));

        // 2) Rewrite the param arena with this batch's fields.
        let gp = std::mem::size_of::<GemmGpuParams>();
        let ap = std::mem::size_of::<AttentionGpuParams>();
        let ep = std::mem::size_of::<ElemGpuParams>();
        let bu = b as u32;
        {
            let scratch = &mut plan.scratch;
            for (l, proto) in plan.protos.iter().enumerate() {
                let base = l as u64 * SLOTS_PER_LAYER * SLOT;
                for (i, slot) in [
                    S_GEMM_K,
                    S_GEMM_V,
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
                    a.batch_start_token_idx = start_pos;
                    a.token_idx = start_pos;
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
            // Tail: output RMSNorm elem (batch = B).
            {
                let mut e = ElemGpuParams {
                    n: n_embd as u32,
                    batch: bu,
                    op: ELEM_OP_RMS_NORM,
                    eps: RMS_NORM_EPS,
                    a_row_stride: 0,
                    b_row_stride: 0,
                    out_row_stride: 0,
                    a_slot: 0,
                    b_slot: 0,
                    out_slot: 0,
                    _pad: 0,
                };
                e.batch = bu;
                let off = (plan.tail_rms_slot * SLOT) as usize;
                scratch[off..off + ep].copy_from_slice(bytemuck::bytes_of(&e));
            }
            // Tail: per-chunk logits GEMV params (full struct; only `n_batch` varies). Must write the
            // WHOLE struct into scratch — the full-arena upload below would otherwise zero these slots.
            for chunk in &plan.chunks {
                let mut g = chunk.params;
                g.n_batch = bu;
                let off = (chunk.slot * SLOT) as usize;
                scratch[off..off + gp].copy_from_slice(bytemuck::bytes_of(&g));
            }
        }
        queue.write_buffer(&plan.param_arena, 0, &plan.scratch);

        // 3) Encode: batched forward (all layers) → output RMSNorm → per-chunk logits GEMV.
        let mut encoder =
            self.host
                .gpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("VerifyBatchEncoder"),
                });
        let gemv = if plan.use_coop {
            r.coop_gemv_pipeline
        } else {
            r.pipeline
        };
        let rms = r.rms_pipeline?;
        let add = r.add_pipeline?;
        let silu = r.elem_silu_mul_pipeline;
        let attn = r.attention_pipeline;

        let gemv_wg = |n_out: u32| {
            if plan.use_coop {
                n_out
            } else {
                n_out.div_ceil(64)
            }
        };
        let elem_wg = |n: u32| n.div_ceil(64);
        let (n_embd_u, n_ffn_u, kv_dim_u) = (n_embd as u32, plan.n_ffn as u32, plan.kv_dim as u32);
        let kv_pairs = plan.n_kv_head.max(1) * bu;

        for lb in &plan.layers {
            let seq: [(&wgpu::ComputePipeline, &wgpu::BindGroup, u32, u32); 14] = [
                (rms, &lb.rms_attn, 1, bu),
                (gemv, &lb.k_gemm, gemv_wg(kv_dim_u), bu),
                (attn, &lb.k_write, kv_pairs, 1),
                (gemv, &lb.v_gemm, gemv_wg(kv_dim_u), bu),
                (attn, &lb.v_write, kv_pairs, 1),
                (attn, &lb.q, plan.n_head.max(1), bu),
                (gemv, &lb.o, gemv_wg(n_embd_u), bu),
                (add, &lb.add1, elem_wg(n_embd_u), bu),
                (rms, &lb.rms_ffn, 1, bu),
                (gemv, &lb.gate, gemv_wg(n_ffn_u), bu),
                (gemv, &lb.up, gemv_wg(n_ffn_u), bu),
                (silu, &lb.silu, elem_wg(n_ffn_u), bu),
                (gemv, &lb.down, gemv_wg(n_embd_u), bu),
                (add, &lb.add2, elem_wg(n_embd_u), bu),
            ];
            for (pipe, bg, wg_x, wg_y) in seq {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipe);
                cpass.set_bind_group(0, bg, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        // Output RMSNorm (hidden_a → normed).
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VerifyOutputNorm"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(rms);
            cpass.set_bind_group(0, &plan.rms_out, &[]);
            cpass.dispatch_workgroups(1, bu, 1);
        }
        // Per-chunk logits GEMV (all B rows) into logits_out.
        for chunk in &plan.chunks {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VerifyLogitsGemv"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(gemv);
            cpass.set_bind_group(0, &chunk.gemm, &[]);
            cpass.dispatch_workgroups(gemv_wg(chunk.rows), bu, 1);
        }
        // Copy the B × vocab logit block to the readback staging.
        let logit_bytes = (b * vocab * 4) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(&plan.logits_out, 0, &plan.logits_staging, 0, logit_bytes);

        // 4) ONE submit, ONE fence, then CPU argmax per position.
        if !self.host.submit(encoder.finish()) {
            return None;
        }
        let _mapping = super::map_read(
            self.host,
            &plan.logits_staging,
            logit_bytes,
            &plan.map_status,
        )?;
        let slice = plan.logits_staging.slice(..logit_bytes);
        {
            let data = slice
                .get_mapped_range()
                .expect("wgpu buffer map_range failed");
            let all: &[f32] = bytemuck::cast_slice(&data[..(b * vocab * 4)]);
            for m in 0..b {
                let row = &all[m * vocab..(m + 1) * vocab];
                let mut best_id = 0u32;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    // Ties → lower index (matches CPU argmax / topk_reduction semantics).
                    if v > best_v {
                        best_v = v;
                        best_id = i as u32;
                    }
                }
                out_argmax[m] = best_id;
                out_logit[m] = best_v;
            }
        }

        Some(())
    }
}
