use super::*;
use crate::ordinary::gpu_params::*;
use etalon_model::inference::{LayerTensors, TensorInfo as GgufTensorInfo};
impl<P: PrefillProvider> PrefillEngine<'_, P> {
    pub async fn dispatch_prefill_chunk_async_mc8_gpu(
        &self,
        index: &P::Index,
        batch_hidden: &mut [f32],
        emb_dim: usize,
        n_tokens: u32,
        batch_start_token_idx: u32,
        _scratch_a: &mut [f32],
        _scratch_b: &mut [f32],
        max_layers: u32,
        mut l1_hidden_out: Option<&mut [f32]>,
    ) -> bool {
        let resources = self.host.prefill_resources();
        let n_layer = P::hyperparams(index).n_layer;
        if n_layer == 0 || n_tokens == 0 || !self.host.mc8_buffers_ready() {
            return false;
        }
        let prefill_scratch = match resources.prefill_scratch_buf {
            Some(b) => b,
            None => return false,
        };
        let limit = if max_layers == 0 {
            n_layer
        } else {
            max_layers.min(n_layer)
        };
        let n_embd = P::hyperparams(index).n_embd as usize;
        let batch_elems = n_embd * n_tokens as usize;
        if batch_elems > batch_hidden.len() || emb_dim < n_embd {
            return false;
        }
        let batch_buf = resources.gemm_input_buf.unwrap();
        let token_buf = resources.gemm_output_buf.unwrap();
        if resources.norm_weight_buf.is_none() {
            return false;
        }
        if batch_elems > resources.gemm_max_input_floats {
            return false;
        }
        self.host.gpu_queue().write_buffer(
            batch_buf,
            0,
            bytemuck::cast_slice(&batch_hidden[..batch_elems]),
        );
        let mmap = match self.host.model_bytes() {
            Some(m) => m,
            None => return false,
        };
        let layout = match self.host.kv_layout() {
            Some(l) => l,
            None => return false,
        };
        for layer in 0..limit {
            let tensors = P::tensors(index, layer);
            let k_info = match tensors.attn_k.as_ref() {
                Some(i) => i,
                None => return false,
            };
            let v_info = match tensors.attn_v.as_ref() {
                Some(i) => i,
                None => return false,
            };
            let k_raw = match etalon_model::ggml::fetch_tensor_bytes(
                mmap,
                P::tensor_data_start(index),
                k_info,
            ) {
                Ok(s) => s,
                Err(_) => return false,
            };
            let v_raw = match etalon_model::ggml::fetch_tensor_bytes(
                mmap,
                P::tensor_data_start(index),
                v_info,
            ) {
                Ok(s) => s,
                Err(_) => return false,
            };
            let h = P::hyperparams(index);
            let n_kv = h.effective_n_kv_head();
            let used_attn_norm = tensors.attn_norm.is_some();
            let mut layer_uniform_cursors = Mc8ChunkUniformCursors {
                attn: 0,
                elem: 0,
                gemm: 0,
            };
            let (uniforms, geom) = match self.mc8_stage_prefill_layer_super_arena(
                index,
                layer,
                &tensors,
                batch_start_token_idx,
                n_tokens,
                emb_dim,
                used_attn_norm,
                k_info,
                &k_raw,
                v_info,
                &v_raw,
                &mut layer_uniform_cursors,
            ) {
                Some(v) => v,
                None => return false,
            };
            let mut enc = self.host.begin();
            let attn_src = if used_attn_norm {
                if let (Some(norm), Some(off)) =
                    (tensors.attn_norm.as_ref(), uniforms.attn_norm_elem_off)
                {
                    let (norm_b, norm_b_off) = match self.host.mc8_norm_source(
                        mmap,
                        P::tensor_data_start(index),
                        norm,
                        n_embd,
                        layer,
                        false,
                    ) {
                        Some(v) => v,
                        None => return false,
                    };
                    self.host.encode_elem_offset(
                        &mut enc,
                        ELEM_OP_RMS_NORM,
                        n_embd as u32,
                        n_tokens,
                        batch_buf,
                        0,
                        geom.batch_in_bytes,
                        norm_b,
                        norm_b_off,
                        geom.n_embd_bytes,
                        prefill_scratch,
                        0,
                        geom.batch_in_bytes,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        off,
                    );
                }
                prefill_scratch
            } else {
                batch_buf
            };
            let n_kv_wg = n_tokens.saturating_mul(n_kv);
            // Phase 5.5: K/V projection on the parallel GEMM → proj buffers (attention reads them).
            let kv_dim = (n_kv * h.head_dim()) as usize;
            let kv_proj_bytes = (kv_dim * n_tokens as usize * 4) as wgpu::BufferAddress;
            let k_proj = resources.mc8_k_proj_buf.unwrap();
            let v_proj = resources.mc8_v_proj_buf.unwrap();
            if !self.host.encode_gemm_bufs_offset(
                &mut enc,
                k_info,
                k_raw,
                n_embd,
                kv_dim,
                attn_src,
                0,
                geom.batch_in_bytes,
                k_proj,
                0,
                kv_proj_bytes,
                n_tokens,
                n_embd as u32,
                kv_dim as u32,
                uniforms.off_k_gemm,
                layer,
                Mc8WeightRole::AttnK,
            ) {
                return false;
            }
            if !self.host.encode_gemm_bufs_offset(
                &mut enc,
                v_info,
                v_raw,
                n_embd,
                kv_dim,
                attn_src,
                0,
                geom.batch_in_bytes,
                v_proj,
                0,
                kv_proj_bytes,
                n_tokens,
                n_embd as u32,
                kv_dim as u32,
                uniforms.off_v_gemm,
                layer,
                Mc8WeightRole::AttnV,
            ) {
                return false;
            }
            if !self.host.encode_attention_pass_gpu(
                &mut enc,
                k_proj,
                token_buf,
                n_embd,
                n_tokens,
                batch_start_token_idx,
                &layout,
                layer,
                batch_start_token_idx,
                &h,
                k_info,
                k_raw,
                1,
                n_kv_wg,
                uniforms.k_off,
                Mc8WeightRole::AttnK,
            ) {
                return false;
            }
            if !self.host.encode_attention_pass_gpu(
                &mut enc,
                v_proj,
                token_buf,
                n_embd,
                n_tokens,
                batch_start_token_idx,
                &layout,
                layer,
                batch_start_token_idx,
                &h,
                v_info,
                v_raw,
                2,
                n_kv_wg,
                uniforms.v_off,
                Mc8WeightRole::AttnV,
            ) {
                return false;
            }
            // Part 3u: KV cache writes must be queue-visible before Q-SDPA reads (backend empirical).
            self.host.mc8_flush(&mut enc);
            const MC8_FUSED_PREFILL_TAIL: bool = true;
            let use_fused_tail = MC8_FUSED_PREFILL_TAIL
                && resources.prefill_work_buf_a.is_some()
                && resources.prefill_work_buf_b.is_some();
            if use_fused_tail {
                let work_a = resources.prefill_work_buf_a.unwrap();
                let work_b = resources.prefill_work_buf_b.unwrap();
                if !self.encode_prefill_q_ffn_tail_fused(
                    &mut enc,
                    index,
                    layer,
                    &tensors,
                    batch_buf,
                    attn_src,
                    work_a,
                    work_b,
                    n_tokens,
                    batch_start_token_idx,
                    emb_dim,
                    used_attn_norm,
                    &uniforms,
                    &geom,
                ) {
                    return false;
                }
            } else {
                let token_buf = resources.gemm_output_buf.unwrap();
                let aux_buf = resources.gemm_aux_buf.unwrap();
                let token_bytes = (n_embd * 4) as wgpu::BufferAddress;
                for t in 0..n_tokens {
                    let abs = batch_start_token_idx + t;
                    let off = params::mc8_emb_off(t, n_embd);
                    P::encoder(&mut enc).copy_buffer_to_buffer(
                        batch_buf,
                        off,
                        token_buf,
                        0,
                        token_bytes,
                    );
                    let attn_in = if used_attn_norm {
                        P::encoder(&mut enc).copy_buffer_to_buffer(
                            prefill_scratch,
                            off,
                            aux_buf,
                            0,
                            token_bytes,
                        );
                        self.host.mc8_flush(&mut enc);
                        Some(aux_buf)
                    } else {
                        None
                    };
                    if !self.host.encode_attn_ffn_tail_gpu(
                        &mut enc, index, layer, abs, emb_dim, &tensors, token_buf, attn_in, true,
                    ) {
                        return false;
                    }
                    self.host.mc8_flush(&mut enc);
                    P::encoder(&mut enc).copy_buffer_to_buffer(
                        token_buf,
                        0,
                        batch_buf,
                        off,
                        token_bytes,
                    );
                    self.host.mc8_flush(&mut enc);
                }
            }
            // Part 3u: one submit per layer (KV flush + layer-end flush = 2 submits/layer).
            self.host.mc8_flush(&mut enc);
            let _ = l1_hidden_out.take();
        }
        self.host.log(&format!(
            "[MC8] GPU prefill OK layers={limit} n_tokens={n_tokens} start={batch_start_token_idx}"
        ));
        true
    }
}
