use super::*;
impl TrunkNvmeReader {
    /// Evaluate a row-addressable output projection without materialising a
    /// logits vector.  The decoder scans one vocabulary row at a time and
    /// retains only the current winner, so a 248k-token vocabulary does not
    /// require a multi-megabyte hot-path allocation.
    pub fn argmax_rows(
        &mut self,
        info: &TensorInfo,
        input: &[f32],
        raw_scratch: &mut [u8],
        row_scratch: &mut [f32],
    ) -> Result<StreamedArgmax, TrunkNvmeError> {
        let mut winner = [StreamedArgmax {
            token_id: 0,
            logit: f32::NEG_INFINITY,
        }];
        self.topk_rows(info, input, raw_scratch, row_scratch, &mut winner)?;
        Ok(winner[0])
    }

    /// Stream the same projection retaining the `out.len()` highest logits,
    /// sorted descending.  Lower token IDs win exact ties, matching the
    /// ascending row scan.
    pub fn topk_rows(
        &mut self,
        info: &TensorInfo,
        input: &[f32],
        raw_scratch: &mut [u8],
        row_scratch: &mut [f32],
        out: &mut [StreamedArgmax],
    ) -> Result<(), TrunkNvmeError> {
        if info.n_dims != 2 || info.dims[0] == 0 || info.dims[1] == 0 {
            return Err(TrunkNvmeError::InvalidTensor);
        }
        let width = info.dims[0] as usize;
        let rows = info.dims[1] as usize;
        if input.len() < width || row_scratch.len() < width {
            return Err(TrunkNvmeError::OutputTooSmall);
        }
        if rows > u32::MAX as usize {
            return Err(TrunkNvmeError::RowOutOfRange);
        }
        if out.is_empty() {
            return Ok(());
        }
        for slot in out.iter_mut() {
            *slot = StreamedArgmax {
                token_id: 0,
                logit: f32::NEG_INFINITY,
            };
        }
        let row_bytes = etalon_model::ggml::ggml_row_bytes(info.ggml_type, width)
            .ok_or(TrunkNvmeError::InvalidTensor)?;
        if let Some(map) = self.map.as_ref() {
            // Parallel local top-k per chunk, then a stable merge.  Chunk
            // order is ascending row, and equal logits keep insertion order,
            // so the merge preserves the sequential ascending-row tie-break.
            let start = usize::try_from(
                self.tensor_data_start
                    .checked_add(info.byte_offset)
                    .ok_or(TrunkNvmeError::OffsetOverflow)?,
            )
            .map_err(|_| TrunkNvmeError::OffsetOverflow)?;
            let span = rows
                .checked_mul(row_bytes)
                .ok_or(TrunkNvmeError::OffsetOverflow)?;
            let bytes = map
                .get(
                    start
                        ..start
                            .checked_add(span)
                            .ok_or(TrunkNvmeError::OffsetOverflow)?,
                )
                .ok_or(TrunkNvmeError::Io)?;
            self.stats.rows += rows as u64;
            self.stats.bytes += span as u64;
            if self
                .parallel
                .topk(bytes, info.ggml_type, width, row_bytes, input, rows, out)?
            {
                return Ok(());
            }
            for row in 0..rows {
                etalon_model::ggml::dequantize_row_into(
                    &bytes[row * row_bytes..(row + 1) * row_bytes],
                    info.ggml_type,
                    width,
                    &mut row_scratch[..width],
                )
                .map_err(|_| TrunkNvmeError::Dequantization)?;
                parallel::insert(
                    out,
                    StreamedArgmax {
                        token_id: row as u32,
                        logit: dot_f32(&row_scratch[..width], input),
                    },
                );
            }
            return Ok(());
        }
        self.read_row_run(
            info,
            0,
            rows,
            raw_scratch,
            row_scratch,
            |row, row_values| {
                let logit = dot_f32(row_values, input);
                if logit <= out[out.len() - 1].logit {
                    return Ok(());
                }
                let mut slot = out.len();
                while slot > 0 && logit > out[slot - 1].logit {
                    slot -= 1;
                }
                for index in (slot + 1..out.len()).rev() {
                    out[index] = out[index - 1];
                }
                out[slot] = StreamedArgmax {
                    token_id: row as u32,
                    logit,
                };
                Ok(())
            },
        )?;
        Ok(())
    }

    /// Report each `target_rows` token's logit and its rank across the whole
    /// projection (rank 0 = argmax).  Diagnostics for comparing against a
    /// reference distribution: the target rows are read directly, then one
    /// streamed pass counts how many vocabulary logits beat each target.
    /// `out[i]` receives `(logit, rank)` for `target_rows[i]`.
    pub fn logits_and_ranks(
        &mut self,
        info: &TensorInfo,
        input: &[f32],
        target_rows: &[u32],
        raw_scratch: &mut [u8],
        row_scratch: &mut [f32],
        out: &mut [(f32, u64)],
    ) -> Result<(), TrunkNvmeError> {
        if info.n_dims != 2 || info.dims[0] == 0 || info.dims[1] == 0 {
            return Err(TrunkNvmeError::InvalidTensor);
        }
        let width = info.dims[0] as usize;
        let rows = info.dims[1] as usize;
        if input.len() < width || row_scratch.len() < width || out.len() < target_rows.len() {
            return Err(TrunkNvmeError::OutputTooSmall);
        }
        for (slot, &target) in target_rows.iter().enumerate() {
            self.read_row_into(info, target as usize, raw_scratch, row_scratch)?;
            let logit = dot_f32(&row_scratch[..width], input);
            out[slot] = (logit, 0);
        }
        self.read_row_run(
            info,
            0,
            rows,
            raw_scratch,
            row_scratch,
            |_row, row_values| {
                let logit = dot_f32(row_values, input);
                for slot in out.iter_mut() {
                    if logit > slot.0 {
                        slot.1 += 1;
                    }
                }
                Ok(())
            },
        )
    }
}
