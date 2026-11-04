use super::*;
use rayon::prelude::*;

const WIDTH: usize = 65_536;
const CHUNKS: usize = 512;
const TOP_K: usize = 128;
const ROWS_PER_CHUNK: usize = 2048;

pub(super) struct ParallelScratch {
    workers: Vec<f32>,
    partials: Vec<StreamedArgmax>,
}

impl ParallelScratch {
    pub(super) fn new() -> Self {
        Self {
            workers: Vec::new(),
            partials: Vec::new(),
        }
    }

    pub(super) fn required_bytes() -> Result<usize, TrunkNvmeError> {
        rayon::current_num_threads()
            .checked_mul(WIDTH)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(CHUNKS * TOP_K * core::mem::size_of::<StreamedArgmax>()))
            .ok_or(TrunkNvmeError::OffsetOverflow)
    }

    pub(super) fn prepare(&mut self, byte_budget: usize) -> Result<(), TrunkNvmeError> {
        let workers = rayon::current_num_threads();
        let floats = workers
            .checked_mul(WIDTH)
            .ok_or(TrunkNvmeError::OffsetOverflow)?;
        let bytes = floats
            .checked_mul(4)
            .and_then(|n| n.checked_add(CHUNKS * TOP_K * core::mem::size_of::<StreamedArgmax>()))
            .ok_or(TrunkNvmeError::OffsetOverflow)?;
        if bytes > byte_budget {
            return Err(TrunkNvmeError::RawScratchTooSmall);
        }
        let mut worker_storage = Vec::new();
        worker_storage
            .try_reserve_exact(floats)
            .map_err(|_| TrunkNvmeError::RawScratchTooSmall)?;
        worker_storage.resize(floats, 0.0);
        let mut partials = Vec::new();
        partials
            .try_reserve_exact(CHUNKS * TOP_K)
            .map_err(|_| TrunkNvmeError::RawScratchTooSmall)?;
        partials.resize(CHUNKS * TOP_K, empty());
        self.workers = worker_storage;
        self.partials = partials;
        Ok(())
    }

    pub(super) fn gemv(
        &mut self,
        bytes: &[u8],
        kind: u32,
        width: usize,
        row_bytes: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<bool, TrunkNvmeError> {
        if width > WIDTH || self.workers.is_empty() || output.is_empty() {
            return Ok(false);
        }
        let worker_count = self.workers.len() / WIDTH;
        let rows_per_worker = output.len().div_ceil(worker_count);
        output
            .par_chunks_mut(rows_per_worker)
            .zip(self.workers.par_chunks_mut(WIDTH))
            .enumerate()
            .try_for_each(|(worker, (out, scratch))| {
                for (index, slot) in out.iter_mut().enumerate() {
                    let start = (worker * rows_per_worker + index) * row_bytes;
                    etalon_model::ggml::dequantize_row_into(
                        &bytes[start..start + row_bytes],
                        kind,
                        width,
                        &mut scratch[..width],
                    )
                    .map_err(|_| TrunkNvmeError::Dequantization)?;
                    *slot = dot_f32(&scratch[..width], input);
                }
                Ok::<(), TrunkNvmeError>(())
            })?;
        Ok(true)
    }

    pub(super) fn topk(
        &mut self,
        bytes: &[u8],
        kind: u32,
        width: usize,
        row_bytes: usize,
        input: &[f32],
        rows: usize,
        out: &mut [StreamedArgmax],
    ) -> Result<bool, TrunkNvmeError> {
        let k = out.len();
        let chunks = rows.div_ceil(ROWS_PER_CHUNK);
        if width > WIDTH || k > TOP_K || chunks > CHUNKS || self.workers.is_empty() {
            return Ok(false);
        }
        let workers = self.workers.len() / WIDTH;
        let chunks_per_worker = chunks.div_ceil(workers);
        let partials = &mut self.partials[..chunks * k];
        partials.fill(empty());
        partials
            .par_chunks_mut(chunks_per_worker * k)
            .zip(self.workers.par_chunks_mut(WIDTH))
            .enumerate()
            .try_for_each(|(worker, (locals, scratch))| {
                for (local_index, local) in locals.chunks_mut(k).enumerate() {
                    let chunk = worker * chunks_per_worker + local_index;
                    for row in chunk * ROWS_PER_CHUNK..((chunk + 1) * ROWS_PER_CHUNK).min(rows) {
                        let start = row * row_bytes;
                        etalon_model::ggml::dequantize_row_into(
                            &bytes[start..start + row_bytes],
                            kind,
                            width,
                            &mut scratch[..width],
                        )
                        .map_err(|_| TrunkNvmeError::Dequantization)?;
                        insert(
                            local,
                            StreamedArgmax {
                                token_id: row as u32,
                                logit: dot_f32(&scratch[..width], input),
                            },
                        );
                    }
                }
                Ok::<(), TrunkNvmeError>(())
            })?;
        for &candidate in partials.iter() {
            if candidate.logit != f32::NEG_INFINITY {
                insert(out, candidate);
            }
        }
        Ok(true)
    }
}

fn empty() -> StreamedArgmax {
    StreamedArgmax {
        token_id: 0,
        logit: f32::NEG_INFINITY,
    }
}

pub(super) fn insert(out: &mut [StreamedArgmax], candidate: StreamedArgmax) {
    if candidate.logit <= out[out.len() - 1].logit {
        return;
    }
    let mut slot = out.len();
    while slot > 0 && candidate.logit > out[slot - 1].logit {
        slot -= 1;
    }
    if slot == out.len() {
        return;
    }
    for index in (slot + 1..out.len()).rev() {
        out[index] = out[index - 1];
    }
    out[slot] = candidate;
}
