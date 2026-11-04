use super::*;
impl<P: BrowserProjectionProvider> BrowserProjectionEngine<'_, P> {
    pub async fn dispatch_output_argmax_chunked_async_mc8_fused(
        &self,
        index: &P::Index,
        hidden: &[f32],
        emb_dim: usize,
        chunk_logits: &mut [f32],
        max_chunks: u32,
        sieve_mask: Option<&P::Mask>,
    ) -> Option<StreamingArgmaxResult> {
        let resources = self.host.resources();
        let info = P::logits(index)?;
        let (n_in, vocab_size) = crate::ordinary::matmul_dims(&info);
        if n_in == 0 || vocab_size == 0 || n_in > emb_dim || n_in > hidden.len() {
            return None;
        }
        if chunk_logits.len() < VOCAB_CHUNK_ROWS || !self.host.buffers_ready() {
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
            // Phase 5.3: bind the resident output-projection sub-range (zero per-token upload)
            // when available; otherwise fall back to per-chunk fetch + write_buffer. The chunk
            // byte offset is 256-aligned because VOCAB_CHUNK_ROWS is a multiple of 256.
            let ok = if let Some(buf) = resources.mc8_logits_resident_buf {
                let row_bytes = resources.mc8_logits_row_bytes as u64;
                let weight = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: row_start as u64 * row_bytes,
                    size: std::num::NonZeroU64::new(chunk_rows as u64 * row_bytes),
                });
                self.dispatch_gemm_resident_chunk_async(
                    info.ggml_type,
                    info.dims[0] as u32,
                    weight,
                    (chunk_rows as u64 * row_bytes) as u32,
                    &hidden[..n_in],
                    &mut chunk_logits[..chunk_rows],
                    n_in,
                    chunk_rows,
                )
                .await
            } else {
                let raw = etalon_model::ggml::fetch_tensor_row_range_bytes(
                    mmap,
                    P::data_start(index),
                    &info,
                    row_start,
                    chunk_rows,
                )
                .ok()?;
                self.dispatch_gemm_raw_into_async(
                    &info,
                    raw,
                    &hidden[..n_in],
                    &mut chunk_logits[..chunk_rows],
                    n_in,
                    chunk_rows,
                )
                .await
            };
            if !ok {
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
}
