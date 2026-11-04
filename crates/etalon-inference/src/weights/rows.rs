use super::*;
impl TrunkNvmeReader {
    /// Opens the resource.
    pub fn open(source: &Path, tensor_data_start: u64) -> Result<Self, TrunkNvmeError> {
        let file = File::open(source).map_err(|_| TrunkNvmeError::Io)?;
        Ok(Self {
            file: source::WeightCursor::File(file),
            map: None,
            tensor_data_start,
            chunk: vec![0u8; TRUNK_CHUNK_BYTES],
            stats: TrunkIoStats::default(),
            parallel: parallel::ParallelScratch::new(),
        })
    }

    /// Open the same source through a read-only OS mapping.  Page residency
    /// is the OS's decision; this reader still owns no decoded weights.
    pub fn open_mapped(source: &Path, tensor_data_start: u64) -> Result<Self, TrunkNvmeError> {
        Self::open_mapped_with_budget(
            source,
            tensor_data_start,
            parallel::ParallelScratch::required_bytes()?,
        )
    }

    pub fn open_mapped_with_budget(
        source: &Path,
        tensor_data_start: u64,
        byte_budget: usize,
    ) -> Result<Self, TrunkNvmeError> {
        let file = File::open(source).map_err(|_| TrunkNvmeError::Io)?;
        // Safety: read-only mapping of a file the decode contract treats as
        // immutable for the session, identical to gguf_bridge's memmap2 use.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|_| TrunkNvmeError::Io)?;
        let mut parallel = parallel::ParallelScratch::new();
        parallel.prepare(byte_budget)?;
        Ok(Self {
            file: source::WeightCursor::File(file),
            map: Some(source::WeightMap::File(map)),
            tensor_data_start,
            chunk: Vec::new(),
            stats: TrunkIoStats::default(),
            parallel,
        })
    }

    /// Consume the retained immutable source; no pathname is reopened.
    pub fn from_lease(
        lease: std::sync::Arc<etalon_model_io::lease::ModelLease>,
        tensor_data_start: u64,
        mapped: bool,
        scratch_budget: usize,
    ) -> Result<Self, TrunkNvmeError> {
        if tensor_data_start >= lease.identity().byte_len {
            return Err(TrunkNvmeError::InvalidTensor);
        }
        let mut parallel = parallel::ParallelScratch::new();
        if mapped { parallel.prepare(scratch_budget)?; }
        if !mapped && scratch_budget < TRUNK_CHUNK_BYTES {
            return Err(TrunkNvmeError::RawScratchTooSmall);
        }
        Ok(Self {
            file: source::WeightCursor::Lease(std::io::Cursor::new(source::LeaseBytes(lease.clone()))),
            map: mapped.then(|| source::WeightMap::Lease(lease)),
            tensor_data_start,
            chunk: if mapped { Vec::new() } else { vec![0; TRUNK_CHUNK_BYTES] },
            stats: TrunkIoStats::default(),
            parallel,
        })
    }

    /// Whether row access is served from the page-cache mapping.
    pub fn is_mapped(&self) -> bool {
        self.map.is_some()
    }

    pub fn mapped_bytes(&self) -> Option<&[u8]> {
        self.map.as_deref()
    }

    /// Cumulative physical I/O performed by this reader.
    pub fn io_stats(&self) -> TrunkIoStats {
        self.stats
    }

    /// Read and dequantize one flattened tensor row. For an expert tensor,
    /// `row` includes its expert-plane offset; no entire expert tensor is read.
    pub fn read_row_into(
        &mut self,
        info: &TensorInfo,
        row: usize,
        raw_scratch: &mut [u8],
        out: &mut [f32],
    ) -> Result<usize, TrunkNvmeError> {
        if info.n_dims == 0 || info.dims[0] == 0 {
            return Err(TrunkNvmeError::InvalidTensor);
        }
        let width = info.dims[0] as usize;
        let row_bytes = etalon_model::ggml::ggml_row_bytes(info.ggml_type, width)
            .ok_or(TrunkNvmeError::InvalidTensor)?;
        if raw_scratch.len() < row_bytes {
            return Err(TrunkNvmeError::RawScratchTooSmall);
        }
        if out.len() < width {
            return Err(TrunkNvmeError::OutputTooSmall);
        }
        let mut total_rows = 1usize;
        for dim in info.dims.iter().take(info.n_dims as usize).skip(1) {
            total_rows = total_rows
                .checked_mul(*dim as usize)
                .ok_or(TrunkNvmeError::OffsetOverflow)?;
        }
        if row >= total_rows {
            return Err(TrunkNvmeError::RowOutOfRange);
        }
        let relative = info
            .byte_offset
            .checked_add(
                (row as u64)
                    .checked_mul(row_bytes as u64)
                    .ok_or(TrunkNvmeError::OffsetOverflow)?,
            )
            .ok_or(TrunkNvmeError::OffsetOverflow)?;
        let offset = self
            .tensor_data_start
            .checked_add(relative)
            .ok_or(TrunkNvmeError::OffsetOverflow)?;
        let mapped = match self.map.as_ref() {
            Some(map) => {
                let start = usize::try_from(offset).map_err(|_| TrunkNvmeError::OffsetOverflow)?;
                let end = start
                    .checked_add(row_bytes)
                    .ok_or(TrunkNvmeError::OffsetOverflow)?;
                Some(map.get(start..end).ok_or(TrunkNvmeError::Io)?)
            }
            None => None,
        };
        let raw: &[u8] = match mapped {
            Some(bytes) => bytes,
            None => {
                self.file
                    .seek(SeekFrom::Start(offset))
                    .map_err(|_| TrunkNvmeError::Io)?;
                self.file
                    .read_exact(&mut raw_scratch[..row_bytes])
                    .map_err(|_| TrunkNvmeError::Io)?;
                self.stats.reads += 1;
                &raw_scratch[..row_bytes]
            }
        };
        self.stats.rows += 1;
        self.stats.bytes += row_bytes as u64;
        etalon_model::ggml::dequantize_row_into(raw, info.ggml_type, width, out)
            .map_err(|_| TrunkNvmeError::Dequantization)
    }

    /// Read `rows` consecutive flattened rows starting at `first_row` through
    /// the internal chunk buffer: one file read per chunk, one dequantization
    /// per row.  `visit` receives each dequantized row in order.
    pub(super) fn read_row_run<F>(
        &mut self,
        info: &TensorInfo,
        first_row: usize,
        rows: usize,
        raw_scratch: &mut [u8],
        row_scratch: &mut [f32],
        mut visit: F,
    ) -> Result<(), TrunkNvmeError>
    where
        F: FnMut(usize, &[f32]) -> Result<(), TrunkNvmeError>,
    {
        if info.n_dims == 0 || info.dims[0] == 0 {
            return Err(TrunkNvmeError::InvalidTensor);
        }
        let width = info.dims[0] as usize;
        let row_bytes = etalon_model::ggml::ggml_row_bytes(info.ggml_type, width)
            .ok_or(TrunkNvmeError::InvalidTensor)?;
        if row_scratch.len() < width {
            return Err(TrunkNvmeError::OutputTooSmall);
        }
        let mut total_rows = 1usize;
        for dim in info.dims.iter().take(info.n_dims as usize).skip(1) {
            total_rows = total_rows
                .checked_mul(*dim as usize)
                .ok_or(TrunkNvmeError::OffsetOverflow)?;
        }
        let end = first_row
            .checked_add(rows)
            .ok_or(TrunkNvmeError::OffsetOverflow)?;
        if end > total_rows {
            return Err(TrunkNvmeError::RowOutOfRange);
        }
        if let Some(map) = self.map.as_ref() {
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
            for index in 0..rows {
                let row_start = index * row_bytes;
                etalon_model::ggml::dequantize_row_into(
                    &bytes[row_start..row_start + row_bytes],
                    info.ggml_type,
                    width,
                    &mut row_scratch[..width],
                )
                .map_err(|_| TrunkNvmeError::Dequantization)?;
                visit(first_row + index, &row_scratch[..width])?;
            }
            return Ok(());
        }
        if row_bytes > self.chunk.len() || row_bytes > raw_scratch.len() {
            // Fall back to one read per row when the fixed chunk cannot hold it.
            for row in first_row..end {
                self.read_row_into(info, row, raw_scratch, row_scratch)?;
                visit(row, &row_scratch[..width])?;
            }
            return Ok(());
        }
        let rows_per_chunk = self.chunk.len() / row_bytes;
        let mut row = first_row;
        while row < end {
            let take = rows_per_chunk.min(end - row);
            let relative = info
                .byte_offset
                .checked_add(
                    (row as u64)
                        .checked_mul(row_bytes as u64)
                        .ok_or(TrunkNvmeError::OffsetOverflow)?,
                )
                .ok_or(TrunkNvmeError::OffsetOverflow)?;
            let offset = self
                .tensor_data_start
                .checked_add(relative)
                .ok_or(TrunkNvmeError::OffsetOverflow)?;
            let span = take
                .checked_mul(row_bytes)
                .ok_or(TrunkNvmeError::OffsetOverflow)?;
            self.file
                .seek(SeekFrom::Start(offset))
                .map_err(|_| TrunkNvmeError::Io)?;
            self.file
                .read_exact(&mut self.chunk[..span])
                .map_err(|_| TrunkNvmeError::Io)?;
            self.stats.reads += 1;
            self.stats.rows += take as u64;
            self.stats.bytes += span as u64;
            for index in 0..take {
                let start = index * row_bytes;
                etalon_model::ggml::dequantize_row_into(
                    &self.chunk[start..start + row_bytes],
                    info.ggml_type,
                    width,
                    &mut row_scratch[..width],
                )
                .map_err(|_| TrunkNvmeError::Dequantization)?;
                visit(row + index, &row_scratch[..width])?;
            }
            row += take;
        }
        Ok(())
    }
}
