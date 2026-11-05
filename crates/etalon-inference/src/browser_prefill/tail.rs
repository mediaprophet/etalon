use super::*;
use crate::ordinary::gpu_params::*;
use etalon_model::inference::LayerTensors;
impl<P: PrefillProvider> PrefillEngine<'_, P> {
    pub fn encode_prefill_q_ffn_tail_fused(
        &self,
        pipeline: &mut P::Pipeline,
        index: &P::Index,
        layer: u32,
        tensors: &LayerTensors,
        batch_buf: &wgpu::Buffer,
        attn_src: &wgpu::Buffer,
        work_a: &wgpu::Buffer,
        work_b: &wgpu::Buffer,
        n_tokens: u32,
        batch_start_token_idx: u32,
        emb_dim: usize,
        used_attn_norm: bool,
        uniforms: &Mc8PrefillLayerUniforms,
        geom: &Mc8PrefillLayerGeom,
    ) -> bool {
        let resources = self.host.prefill_resources();
        let mmap = match self.host.model_bytes() {
            Some(m) => m,
            None => return false,
        };
        let h = P::hyperparams(index);
        let n_embd = h.n_embd as usize;
        let layout = match self.host.kv_layout() {
            Some(l) => l,
            None => return false,
        };
        debug_assert!(
            batch_start_token_idx < P::max_context(&layout) || P::max_context(&layout) == 0
        );
        let q_info = match tensors.attn_q.as_ref() {
            Some(i) => i,
            None => return false,
        };
        let q_raw =
            match etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), q_info)
            {
                Ok(s) => s,
                Err(_) => return false,
            };
        let row_stride = geom.row_stride;
        let row_stride_u32 = geom.row_stride_u32;
        let emb_bytes = geom.emb_bytes;
        let n_embd_bytes = geom.n_embd_bytes;
        let slot_o = geom.slot_o;
        let slot_gate = geom.slot_gate;
        let slot_up = geom.slot_up;
        let slot_save = geom.slot_save;
        let slot_scratch_half = geom.slot_scratch_half;
        let work_span_bytes = geom.work_span_bytes;
        let batch_in_bytes = geom.batch_in_bytes;
        let slot_o_f = geom.slot_o_f;
        let slot_save_f = geom.slot_save_f;
        let slot_gate_f = geom.slot_gate_f;
        let slot_up_f = geom.slot_up_f;
        let slot_scratch_half_f = geom.slot_scratch_half_f;
        let off_attn_res = uniforms.off_attn_res;
        let off_ffn_norm = uniforms.off_ffn_norm;
        let off_gate = uniforms.off_gate;
        let off_up = uniforms.off_up;
        let off_silu = uniforms.off_silu;
        let off_down = uniforms.off_down;
        let off_ffn_res = uniforms.off_ffn_res;
        let off_o = uniforms.off_o;

        let gate_info = match tensors.ffn_gate.as_ref() {
            Some(i) => i,
            None => return false,
        };
        let up_info = match tensors.ffn_up.as_ref() {
            Some(i) => i,
            None => return false,
        };
        let down_info = match tensors.ffn_down.as_ref() {
            Some(i) => i,
            None => return false,
        };
        let (_, n_ffn) = crate::ordinary::matmul_dims(gate_info);
        let q_dim = (h.n_head * h.head_dim()) as usize;
        let (gate_in, _) = crate::ordinary::matmul_dims(gate_info);
        let (up_in, up_out) = crate::ordinary::matmul_dims(up_info);
        let (dn_in, dn_out) = crate::ordinary::matmul_dims(down_info);
        if gate_in > n_embd
            || up_in != gate_in
            || up_out != n_ffn
            || dn_in != n_ffn
            || dn_out < n_embd
        {
            return false;
        }
        let out_info = tensors.attn_output.as_ref();
        let (o_in, o_out) = out_info
            .map(crate::ordinary::matmul_dims)
            .unwrap_or((q_dim, n_embd));
        if out_info.is_some() && o_in > q_dim {
            return false;
        }
        let o_raw = out_info.and_then(|out_info| {
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), out_info).ok()
        });
        let gate_raw = match etalon_model::ggml::fetch_tensor_bytes(
            mmap,
            P::tensor_data_start(index),
            gate_info,
        ) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let up_raw = match etalon_model::ggml::fetch_tensor_bytes(
            mmap,
            P::tensor_data_start(index),
            up_info,
        ) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let down_raw = match etalon_model::ggml::fetch_tensor_bytes(
            mmap,
            P::tensor_data_start(index),
            down_info,
        ) {
            Ok(s) => s,
            Err(_) => return false,
        };

        // Phase 5.5: Q projection on the parallel GEMM → q_proj; the (now SDPA-only) attention
        // shader reads it. Moves the heavy 64×960 matmul off the @workgroup_size(1) attention kernel.
        let q_in = if used_attn_norm { attn_src } else { batch_buf };
        let q_proj = resources.mc8_q_proj_buf.unwrap();
        let q_proj_bytes = (q_dim * n_tokens as usize * 4) as wgpu::BufferAddress;
        if !self.host.encode_gemm_bufs_offset(
            pipeline,
            q_info,
            q_raw,
            n_embd,
            q_dim,
            q_in,
            0,
            batch_in_bytes,
            q_proj,
            0,
            q_proj_bytes,
            n_tokens,
            n_embd as u32,
            q_dim as u32,
            uniforms.off_q_gemm,
            layer,
            Mc8WeightRole::AttnQ,
        ) {
            return false;
        }
        // Part 3u: Q SDPA — attn params pre-staged at uniforms.q_off (now reads pre-computed Q).
        if !self.host.encode_attention_batched_q_prefill(
            pipeline,
            q_proj,
            0,
            q_proj_bytes,
            work_a,
            0,
            work_span_bytes,
            n_embd,
            n_tokens,
            &layout,
            layer,
            &h,
            q_raw,
            uniforms.q_off,
        ) {
            return false;
        }

        // Snapshot pristine hidden (MC8 pt3i `work_aliases_hidden`) before o_proj overwrites paths.
        for t in 0..n_tokens {
            let emb_off = params::mc8_emb_off(t, n_embd);
            let row_off = params::mc8_prefill_row_off(t, row_stride);
            P::encoder(pipeline).copy_buffer_to_buffer(
                batch_buf,
                emb_off,
                work_a,
                row_off + slot_save,
                emb_bytes,
            );
        }

        // Part 3s — attention block: o_proj + attn residual (dynamic uniform offsets).
        if let (Some(out_info), Some(o_raw)) = (out_info, o_raw) {
            if !self.host.encode_gemm_bufs_offset(
                pipeline,
                out_info,
                o_raw,
                o_in,
                o_out,
                work_a,
                0,
                work_span_bytes,
                work_b,
                slot_o,
                work_span_bytes,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
                off_o.unwrap(),
                layer,
                Mc8WeightRole::OProj,
            ) {
                return false;
            }
            self.host.encode_elem_offset(
                pipeline,
                ELEM_OP_ADD_RESIDUAL,
                emb_dim as u32,
                n_tokens,
                work_a,
                0,
                work_span_bytes,
                work_b,
                0,
                work_span_bytes,
                batch_buf,
                0,
                batch_in_bytes,
                row_stride_u32,
                row_stride_u32,
                0,
                slot_save_f,
                slot_o_f,
                0,
                off_attn_res,
            );
        } else {
            self.host.encode_elem_offset(
                pipeline,
                ELEM_OP_ADD_RESIDUAL,
                emb_dim as u32,
                n_tokens,
                work_a,
                0,
                work_span_bytes,
                work_a,
                0,
                work_span_bytes,
                batch_buf,
                0,
                batch_in_bytes,
                row_stride_u32,
                row_stride_u32,
                0,
                slot_save_f,
                0,
                0,
                off_attn_res,
            );
        }

        // Refresh post-attn hidden for FFN residual (overwrites pristine snapshot in slot_save).
        for t in 0..n_tokens {
            let emb_off = params::mc8_emb_off(t, n_embd);
            let row_off = params::mc8_prefill_row_off(t, row_stride);
            P::encoder(pipeline).copy_buffer_to_buffer(
                batch_buf,
                emb_off,
                work_a,
                row_off + slot_save,
                emb_bytes,
            );
        }

        // Part 3s — FFN block: norm + gate/up (one submit via dynamic offsets + weight ping-pong).
        if let Some(norm) = tensors.ffn_norm.as_ref() {
            let (norm_b, norm_b_off) = match self.host.mc8_norm_source(
                mmap,
                P::tensor_data_start(index),
                norm,
                n_embd,
                layer,
                true,
            ) {
                Some(v) => v,
                None => return false,
            };
            self.host.encode_elem_offset(
                pipeline,
                ELEM_OP_RMS_NORM,
                n_embd as u32,
                n_tokens,
                batch_buf,
                0,
                batch_in_bytes,
                norm_b,
                norm_b_off,
                n_embd_bytes,
                work_b,
                0,
                work_span_bytes,
                0,
                0,
                row_stride_u32,
                0,
                0,
                0,
                off_ffn_norm.unwrap(),
            );
        } else {
            for t in 0..n_tokens {
                let emb_off = params::mc8_emb_off(t, n_embd);
                let row_off = params::mc8_prefill_row_off(t, row_stride);
                P::encoder(pipeline)
                    .copy_buffer_to_buffer(batch_buf, emb_off, work_b, row_off, emb_bytes);
            }
        }
        // Phase 5 — FFN expansion fusion: collapse gate GEMM + up GEMM + SiLU×mul into a
        // single dispatch. The result `silu(gate·x)·(up·x)` lands in work_b@slot_scratch_half
        // — exactly where the separate SiLU path wrote — so the `down` projection below is
        // unchanged. Requires resident weights (so both roles are GPU-resident) and gate/up
        // sharing ggml_type+dims (one staged GemmParams describes both). Otherwise fall back
        // to the proven 3-dispatch path.
        let fuse_ffn_expansion =
            resources.mc8_weights_resident && gate_info.ggml_type == up_info.ggml_type;
        if fuse_ffn_expansion {
            // WebGPU forbids binding one buffer as both read-only and writable storage within a
            // single pass (validated at buffer granularity, not by offset). The norm wrote the
            // hidden to work_b@0, and the fused intermediate must land in work_b@slot_scratch_half
            // for `down` below — same buffer, so reading + writing work_b in one pass is illegal.
            // Stage the normalized hidden into work_a@0 (small n_embd copy; `down` overwrites
            // work_a@0 immediately after), read it there, and write the silu·up result to
            // work_b@slot_scratch_half exactly as the separate-SiLU path did — `down` unchanged.
            for t in 0..n_tokens {
                let row_off = params::mc8_prefill_row_off(t, row_stride);
                P::encoder(pipeline)
                    .copy_buffer_to_buffer(work_b, row_off, work_a, row_off, emb_bytes);
            }
            if !self.host.encode_fused_ffn_expansion(
                pipeline,
                layer,
                work_a,
                0,
                work_span_bytes,
                work_b,
                slot_scratch_half,
                work_span_bytes,
                n_ffn as u32,
                n_tokens,
                off_gate,
            ) {
                return false;
            }
        } else {
            if !self.host.encode_gemm_bufs_offset(
                pipeline,
                gate_info,
                gate_raw,
                gate_in,
                n_ffn,
                work_b,
                0,
                work_span_bytes,
                work_a,
                slot_gate,
                work_span_bytes,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
                off_gate,
                layer,
                Mc8WeightRole::Gate,
            ) {
                return false;
            }
            if !self.host.encode_gemm_bufs_offset(
                pipeline,
                up_info,
                up_raw,
                up_in,
                n_ffn,
                work_b,
                0,
                work_span_bytes,
                work_a,
                slot_up,
                work_span_bytes,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
                off_up,
                layer,
                Mc8WeightRole::Up,
            ) {
                return false;
            }

            self.host.encode_elem_offset(
                pipeline,
                ELEM_OP_SILU_MUL,
                n_ffn as u32,
                n_tokens,
                work_a,
                0,
                work_span_bytes,
                work_a,
                0,
                work_span_bytes,
                work_b,
                0,
                work_span_bytes,
                row_stride_u32,
                row_stride_u32,
                row_stride_u32,
                slot_gate_f,
                slot_up_f,
                slot_scratch_half_f,
                off_silu,
            );
        }
        if !self.host.encode_gemm_bufs_offset(
            pipeline,
            down_info,
            down_raw,
            dn_in,
            dn_out,
            work_b,
            slot_scratch_half,
            work_span_bytes,
            work_a,
            0,
            work_span_bytes,
            n_tokens,
            row_stride_u32,
            row_stride_u32,
            off_down,
            layer,
            Mc8WeightRole::Down,
        ) {
            return false;
        }
        self.host.encode_elem_offset(
            pipeline,
            ELEM_OP_ADD_RESIDUAL,
            emb_dim as u32,
            n_tokens,
            work_a,
            0,
            work_span_bytes,
            work_a,
            0,
            work_span_bytes,
            batch_buf,
            0,
            batch_in_bytes,
            row_stride_u32,
            row_stride_u32,
            0,
            slot_save_f,
            0,
            0,
            off_ffn_res,
        );
        true
    }
}
