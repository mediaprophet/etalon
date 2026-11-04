use super::*;
use crate::ordinary::gpu_params::*;
use etalon_model::inference::{LayerTensors, TensorInfo as GgufTensorInfo};
impl<P: PrefillProvider> PrefillEngine<'_, P> {
    pub fn mc8_stage_prefill_layer_super_arena(
        &self,
        index: &P::Index,
        layer: u32,
        tensors: &LayerTensors,
        batch_start_token_idx: u32,
        n_tokens: u32,
        emb_dim: usize,
        used_attn_norm: bool,
        k_info: &GgufTensorInfo,
        k_raw: &[u8],
        v_info: &GgufTensorInfo,
        v_raw: &[u8],
        cursors: &mut Mc8ChunkUniformCursors,
    ) -> Option<(Mc8PrefillLayerUniforms, Mc8PrefillLayerGeom)> {
        let resources = self.host.prefill_resources();
        let mmap = self.host.model_bytes()?;
        let layout = self.host.kv_layout()?;
        let h = P::hyperparams(index);
        let n_embd = h.n_embd as usize;
        let q_info = tensors.attn_q.as_ref()?;
        let gate_info = tensors.ffn_gate.as_ref()?;
        let up_info = tensors.ffn_up.as_ref()?;
        let down_info = tensors.ffn_down.as_ref()?;
        let q_raw =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), q_info)
                .ok()?;
        let n_ffn_est = crate::ordinary::matmul_dims(gate_info).1;
        let row_stride =
            params::mc8_prefill_row_stride(n_embd, n_ffn_est, resources.gemm_max_out_dim);
        let emb_bytes = (emb_dim * 4) as wgpu::BufferAddress;
        let n_embd_bytes = (n_embd * 4) as wgpu::BufferAddress;
        let q_dim = (h.n_head * h.head_dim()) as usize;
        let q_bytes = (q_dim * 4) as wgpu::BufferAddress;
        let slot_o = q_bytes;
        let slot_scratch_half = ((row_stride / 2) * 4) as wgpu::BufferAddress;
        let (_, n_ffn) = crate::ordinary::matmul_dims(gate_info);
        let slot_gate = n_embd_bytes;
        let slot_up = slot_gate + (n_ffn * 4) as wgpu::BufferAddress;
        let slot_save = slot_up + (n_ffn * 4) as wgpu::BufferAddress;
        let row_stride_u32 = row_stride as u32;
        let batch_in_bytes = (n_embd * n_tokens as usize * 4) as wgpu::BufferAddress;
        let work_span_bytes =
            (n_tokens as usize * row_stride * 4).max(q_bytes as usize) as wgpu::BufferAddress;
        let geom = Mc8PrefillLayerGeom {
            row_stride,
            row_stride_u32,
            batch_in_bytes,
            work_span_bytes,
            emb_bytes,
            n_embd_bytes,
            slot_o,
            slot_gate,
            slot_up,
            slot_save,
            slot_scratch_half,
            slot_o_f: (slot_o / 4) as u32,
            slot_save_f: (slot_save / 4) as u32,
            slot_gate_f: (slot_gate / 4) as u32,
            slot_up_f: (slot_up / 4) as u32,
            slot_scratch_half_f: (slot_scratch_half / 4) as u32,
        };
        let (gate_in, _) = crate::ordinary::matmul_dims(gate_info);
        let (up_in, up_out) = crate::ordinary::matmul_dims(up_info);
        let (dn_in, dn_out) = crate::ordinary::matmul_dims(down_info);
        if gate_in > n_embd
            || up_in != gate_in
            || up_out != n_ffn
            || dn_in != n_ffn
            || dn_out < n_embd
        {
            return None;
        }
        let out_info = tensors.attn_output.as_ref();
        let (o_in, o_out) = out_info
            .map(crate::ordinary::matmul_dims)
            .unwrap_or((q_dim, n_embd));
        if out_info.is_some() && o_in > q_dim {
            return None;
        }
        let o_raw = out_info.and_then(|i| {
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), i).ok()
        });
        let gate_raw =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), gate_info)
                .ok()?;
        let up_raw =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), up_info)
                .ok()?;
        let down_raw =
            etalon_model::ggml::fetch_tensor_bytes(mmap, P::tensor_data_start(index), down_info)
                .ok()?;

        let attn_base_byte = cursors.attn_base_byte();
        let elem_base_byte = cursors.elem_base_byte();
        let gemm_base_byte = cursors.gemm_base_byte();
        let mut attn_arena = Mc8AttnUniformArena {
            bytes: [0u8; MC8_MAX_ATTN_UNIFORM_SLOTS * MC8_UNIFORM_ALIGN],
            slots: 0,
        };
        // Phase 5.5: projection decoupling — the attention shader reads PRE-COMPUTED Q/K/V
        // (parallel GEMM) at `proj_row_stride` floats/token instead of doing the matmul itself.
        let kv_dim = (h.effective_n_kv_head() * h.head_dim()) as u32;
        let q_dim_u32 = (h.n_head * h.head_dim()) as u32;
        let mut k_params = P::attention_gpu_params(
            &h,
            &layout,
            layer,
            batch_start_token_idx,
            k_info,
            k_raw.len(),
            1,
            n_tokens,
            batch_start_token_idx,
            0,
            0,
            0,
        );
        k_params.proj_row_stride = kv_dim;
        let k_off = attn_base_byte + attn_arena.push(&k_params);
        let mut v_params = P::attention_gpu_params(
            &h,
            &layout,
            layer,
            batch_start_token_idx,
            v_info,
            v_raw.len(),
            2,
            n_tokens,
            batch_start_token_idx,
            0,
            0,
            0,
        );
        v_params.proj_row_stride = kv_dim;
        let v_off = attn_base_byte + attn_arena.push(&v_params);
        let mut q_params = P::attention_gpu_params(
            &h,
            &layout,
            layer,
            batch_start_token_idx,
            q_info,
            q_raw.len(),
            0,
            n_tokens,
            batch_start_token_idx,
            0,
            P::mask_word_count(),
            row_stride_u32,
        );
        q_params.proj_row_stride = q_dim_u32;
        let q_off = attn_base_byte + attn_arena.push(&q_params);

        let mut elem_arena = Mc8ElemUniformArena {
            bytes: [0u8; MC8_MAX_ELEM_UNIFORM_LAYER_SLOTS * MC8_UNIFORM_ALIGN],
            slots: 0,
        };
        let attn_norm_elem_off = if used_attn_norm {
            Some(
                elem_base_byte
                    + elem_arena.push(&params::mc8_elem_params(
                        ELEM_OP_RMS_NORM,
                        n_embd as u32,
                        n_tokens,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )),
            )
        } else {
            None
        };
        let off_attn_res = elem_base_byte
            + elem_arena.push(&params::mc8_elem_params(
                ELEM_OP_ADD_RESIDUAL,
                emb_dim as u32,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
                0,
                geom.slot_save_f,
                if out_info.is_some() { geom.slot_o_f } else { 0 },
                0,
            ));
        let off_ffn_norm = tensors.ffn_norm.as_ref().map(|_| {
            elem_base_byte
                + elem_arena.push(&params::mc8_elem_params(
                    ELEM_OP_RMS_NORM,
                    n_embd as u32,
                    n_tokens,
                    0,
                    0,
                    row_stride_u32,
                    0,
                    0,
                    0,
                ))
        });
        let off_silu = elem_base_byte
            + elem_arena.push(&params::mc8_elem_params(
                ELEM_OP_SILU_MUL,
                n_ffn as u32,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
                row_stride_u32,
                geom.slot_gate_f,
                geom.slot_up_f,
                geom.slot_scratch_half_f,
            ));
        let off_ffn_res = elem_base_byte
            + elem_arena.push(&params::mc8_elem_params(
                ELEM_OP_ADD_RESIDUAL,
                emb_dim as u32,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
                0,
                geom.slot_save_f,
                0,
                0,
            ));

        let mut gemm_arena = Mc8UniformArena {
            bytes: [0u8; MC8_MAX_GEMM_UNIFORM_SLOTS * MC8_UNIFORM_ALIGN],
            slots: 0,
        };
        let off_o = out_info.map(|i| {
            gemm_base_byte
                + gemm_arena.push(&params::mc8_gemm_params(
                    i,
                    o_raw.as_ref().map(|r| r.len()).unwrap_or(0),
                    o_in,
                    o_out,
                    n_tokens,
                    row_stride_u32,
                    row_stride_u32,
                ))
        });
        let off_gate = gemm_base_byte
            + gemm_arena.push(&params::mc8_gemm_params(
                gate_info,
                gate_raw.len(),
                gate_in,
                n_ffn,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
            ));
        let off_up = gemm_base_byte
            + gemm_arena.push(&params::mc8_gemm_params(
                up_info,
                up_raw.len(),
                up_in,
                n_ffn,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
            ));
        let off_down = gemm_base_byte
            + gemm_arena.push(&params::mc8_gemm_params(
                down_info,
                down_raw.len(),
                dn_in,
                dn_out,
                n_tokens,
                row_stride_u32,
                row_stride_u32,
            ));
        // Phase 5.5: Q/K/V projection GEMMs run on the parallel kernel (input contiguous n_embd/token,
        // output contiguous q_dim|kv_dim/token) — output feeds the now-lightweight attention shader.
        let off_q_gemm = gemm_base_byte
            + gemm_arena.push(&params::mc8_gemm_params(
                q_info,
                q_raw.len(),
                n_embd,
                q_dim,
                n_tokens,
                n_embd as u32,
                q_dim_u32,
            ));
        let off_k_gemm = gemm_base_byte
            + gemm_arena.push(&params::mc8_gemm_params(
                k_info,
                k_raw.len(),
                n_embd,
                kv_dim as usize,
                n_tokens,
                n_embd as u32,
                kv_dim,
            ));
        let off_v_gemm = gemm_base_byte
            + gemm_arena.push(&params::mc8_gemm_params(
                v_info,
                v_raw.len(),
                n_embd,
                kv_dim as usize,
                n_tokens,
                n_embd as u32,
                kv_dim,
            ));

        let attn_buf = resources.attention_params_buf?;
        let elem_buf = resources.elem_params_buf?;
        let gemm_buf = resources.gemm_params_buf?;
        let queue = self.host.gpu_queue();
        attn_arena.upload_at(queue, attn_buf, cursors.attn);
        cursors.attn += attn_arena.slots;
        elem_arena.upload_at(queue, elem_buf, cursors.elem);
        cursors.elem += elem_arena.slots;
        gemm_arena.upload_at(queue, gemm_buf, cursors.gemm);
        cursors.gemm += gemm_arena.slots;

        Some((
            Mc8PrefillLayerUniforms {
                k_off,
                v_off,
                q_off,
                attn_norm_elem_off,
                off_o,
                off_attn_res,
                off_ffn_norm,
                off_gate,
                off_up,
                off_silu,
                off_down,
                off_ffn_res,
                off_q_gemm,
                off_k_gemm,
                off_v_gemm,
            },
            geom,
        ))
    }
}
