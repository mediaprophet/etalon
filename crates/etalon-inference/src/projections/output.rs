use super::*;
impl<P: ProjectionProvider> ProjectionEngine<'_, P> {
    pub fn dispatch_output_argmax_chunked(
        &self,
        index: &P::Index,
        hidden: &[f32],
        emb_dim: usize,
        chunk_logits: &mut [f32],
        max_chunks: u32,
        sieve_mask: Option<&P::Mask>,
    ) -> Option<StreamingArgmaxResult> {
        if let Some((packed, scales, globals)) = self.host.nvfp4_output() {
            return self.dispatch_output_argmax_ftw_nvfp4(
                packed,
                scales,
                globals,
                hidden,
                emb_dim,
                chunk_logits,
                max_chunks,
                sieve_mask,
            );
        }

        let descriptor = P::logits(index)?;
        let info = &descriptor;
        let (n_in, vocab_size) = crate::ordinary::matmul_dims(info);
        if n_in == 0 || vocab_size == 0 || n_in > emb_dim || n_in > hidden.len() {
            return None;
        }
        if chunk_logits.len() < VOCAB_CHUNK_ROWS {
            return None;
        }
        let mmap = self.host.model_bytes()?;
        let full_chunks = vocab_size.div_ceil(VOCAB_CHUNK_ROWS);
        let n_chunks = if max_chunks == 0 {
            full_chunks
        } else {
            (max_chunks as usize).min(full_chunks)
        };
        let mut best_token_id = 0u32;
        let mut max_logit = f32::NEG_INFINITY;

        for chunk_idx in 0..n_chunks {
            let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
            let chunk_rows = VOCAB_CHUNK_ROWS.min(vocab_size - row_start);
            let raw = etalon_model::ggml::fetch_tensor_row_range_bytes(
                mmap,
                P::data_start(index),
                info,
                row_start,
                chunk_rows,
            )
            .ok()?;
            if !self.host.gemm(
                info,
                raw,
                &hidden[..n_in],
                &mut chunk_logits[..chunk_rows],
                n_in,
                chunk_rows,
            ) {
                return None;
            }
            if let Some(mask) = sieve_mask {
                self.host.update_argmax(
                    &chunk_logits[..chunk_rows],
                    chunk_rows,
                    chunk_idx,
                    Some(mask),
                    &mut best_token_id,
                    &mut max_logit,
                );
            } else {
                self.host.update_argmax_unmasked(
                    &chunk_logits[..chunk_rows],
                    chunk_rows,
                    chunk_idx,
                    &mut best_token_id,
                    &mut max_logit,
                );
            }
            self.host.scrub(&mut chunk_logits[..chunk_rows], chunk_rows);
        }

        if max_logit == f32::NEG_INFINITY {
            return None;
        }
        Some(StreamingArgmaxResult {
            best_token_id,
            max_logit,
        })
    }
    pub fn dispatch_output_argmax_ftw_nvfp4(
        &self,
        packed: &[u8],
        scales: &[u8],
        globals: &[u8],
        hidden: &[f32],
        emb_dim: usize,
        chunk_logits: &mut [f32],
        max_chunks: u32,
        sieve_mask: Option<&P::Mask>,
    ) -> Option<StreamingArgmaxResult> {
        let vocab_size = globals.len() / 2;
        let row_bytes = emb_dim / 2;
        let scale_bytes = emb_dim / 16;
        if vocab_size == 0
            || emb_dim == 0
            || hidden.len() < emb_dim
            || chunk_logits.len() < VOCAB_CHUNK_ROWS
        {
            return None;
        }
        if packed.len() < vocab_size * row_bytes || scales.len() < vocab_size * scale_bytes {
            return None;
        }

        let full_chunks = vocab_size.div_ceil(VOCAB_CHUNK_ROWS);
        let n_chunks = if max_chunks == 0 {
            full_chunks
        } else {
            (max_chunks as usize).min(full_chunks)
        };
        let mut best_token_id = 0u32;
        let mut max_logit = f32::NEG_INFINITY;

        for chunk_idx in 0..n_chunks {
            let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
            let chunk_rows = VOCAB_CHUNK_ROWS.min(vocab_size - row_start);

            for r_idx in 0..chunk_rows {
                let token_id = row_start + r_idx;
                let g_val =
                    half::f16::from_le_bytes([globals[token_id * 2], globals[token_id * 2 + 1]])
                        .to_f32();
                let p_row = &packed[token_id * row_bytes..(token_id + 1) * row_bytes];
                let s_row = &scales[token_id * scale_bytes..(token_id + 1) * scale_bytes];

                chunk_logits[r_idx] =
                    crate::ordinary::output::nvfp4_output_row(p_row, s_row, g_val, hidden, emb_dim);
            }

            if let Some(mask) = sieve_mask {
                self.host.update_argmax(
                    &chunk_logits[..chunk_rows],
                    chunk_rows,
                    chunk_idx,
                    Some(mask),
                    &mut best_token_id,
                    &mut max_logit,
                );
            } else {
                self.host.update_argmax_unmasked(
                    &chunk_logits[..chunk_rows],
                    chunk_rows,
                    chunk_idx,
                    &mut best_token_id,
                    &mut max_logit,
                );
            }
            self.host.scrub(&mut chunk_logits[..chunk_rows], chunk_rows);
        }

        if max_logit == f32::NEG_INFINITY {
            return None;
        }
        Some(StreamingArgmaxResult {
            best_token_id,
            max_logit,
        })
    }
    pub fn apply_output_norm_inplace(
        &self,
        index: &P::Index,
        hidden: &mut [f32],
        emb_dim: usize,
    ) -> bool {
        let descriptor = P::output_norm(index);
        let info = match descriptor.as_ref() {
            Some(i) => i,
            None => return true,
        };
        let n_embd = P::embedding_dim(index);
        let n = emb_dim.min(n_embd).min(hidden.len());
        let mut norm_w = [0f32; MAX_HIDDEN_DIM];
        let dequant_n = if let Some(raw) = self.host.fetch_tensor_raw_bytes(info) {
            dequant_norm_from_raw(raw, info, &mut norm_w)
        } else if let Some(mmap) = self.host.model_bytes() {
            dequant_norm_row_into(mmap, P::data_start(index), info, &mut norm_w)
        } else {
            0
        };
        if dequant_n < n {
            return false;
        }
        rms_norm_inplace(&mut hidden[..n], &norm_w[..n], RMS_NORM_EPS);
        true
    }
}
