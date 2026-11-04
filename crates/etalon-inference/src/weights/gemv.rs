use super::*;
impl TrunkNvmeReader {
    /// Execute a bounded streamed matrix-vector product over consecutive
    /// flattened rows. Each row is read, dequantized, dotted, then discarded.
    pub fn gemv_rows_into(
        &mut self,
        info: &TensorInfo,
        first_row: usize,
        rows: usize,
        input: &[f32],
        output: &mut [f32],
        raw_scratch: &mut [u8],
        row_scratch: &mut [f32],
    ) -> Result<(), TrunkNvmeError> {
        let width = info.dims[0] as usize;
        if input.len() < width || output.len() < rows || row_scratch.len() < width {
            return Err(TrunkNvmeError::OutputTooSmall);
        }
        let row_bytes = etalon_model::ggml::ggml_row_bytes(info.ggml_type, width)
            .ok_or(TrunkNvmeError::InvalidTensor)?;
        if let Some(map) = self.map.as_ref() {
            // Mapped + parallel: each worker dequantizes rows into its own
            // scratch.  Output slots are disjoint, so results are identical
            // to the sequential path regardless of chunking or thread count.
            let start = usize::try_from(
                self.tensor_data_start
                    .checked_add(info.byte_offset)
                    .and_then(|base| base.checked_add((first_row as u64) * row_bytes as u64))
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
            if self.parallel.gemv(
                bytes,
                info.ggml_type,
                width,
                row_bytes,
                input,
                &mut output[..rows],
            )? {
                return Ok(());
            }
            for (row, slot) in output[..rows].iter_mut().enumerate() {
                etalon_model::ggml::dequantize_row_into(
                    &bytes[row * row_bytes..(row + 1) * row_bytes],
                    info.ggml_type,
                    width,
                    &mut row_scratch[..width],
                )
                .map_err(|_| TrunkNvmeError::Dequantization)?;
                *slot = dot_f32(&row_scratch[..width], input);
            }
            return Ok(());
        }
        self.read_row_run(
            info,
            first_row,
            rows,
            raw_scratch,
            row_scratch,
            |row, row_values| {
                output[row - first_row] = dot_f32(row_values, input);
                Ok(())
            },
        )
    }

    /// Execute one selected plane of an encoded `[in, out, expert]` tensor.
    /// Only the selected expert plane is read; other expert weights remain
    /// untouched on disk regardless of the model's expert count or top-k.
    pub fn gemv_expert_plane_into(
        &mut self,
        info: &TensorInfo,
        expert: usize,
        input: &[f32],
        output: &mut [f32],
        raw_scratch: &mut [u8],
        row_scratch: &mut [f32],
    ) -> Result<usize, TrunkNvmeError> {
        if info.n_dims != 3 || info.dims[1] == 0 || info.dims[2] == 0 {
            return Err(TrunkNvmeError::InvalidTensor);
        }
        let rows_per_expert = info.dims[1] as usize;
        if expert >= info.dims[2] as usize {
            return Err(TrunkNvmeError::RowOutOfRange);
        }
        let first_row = expert
            .checked_mul(rows_per_expert)
            .ok_or(TrunkNvmeError::OffsetOverflow)?;
        self.gemv_rows_into(
            info,
            first_row,
            rows_per_expert,
            input,
            output,
            raw_scratch,
            row_scratch,
        )?;
        Ok(rows_per_expert)
    }
}
