//! Bounded model-weight row reads, mapped access, and matrix execution.
//!
//! Callers supply tensor descriptors and the source data offset. Model-family
//! adapters select and admit their weight sources; this layer performs row I/O
//! and numerical work without requiring a Qwen architecture. A mapped address
//! reservation does not imply that the model is resident in physical memory.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use etalon_model::inference::TensorInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrunkNvmeError {
    InvalidTensor,
    RowOutOfRange,
    RawScratchTooSmall,
    OutputTooSmall,
    OffsetOverflow,
    Io,
    Dequantization,
}

/// Exact winner from a streamed vocabulary projection.  The token id is a
/// flattened row number in the output tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamedArgmax {
    pub token_id: u32,
    pub logit: f32,
}

/// Cumulative physical I/O accounting for one reader.  `reads` counts file
/// read calls (chunked), `rows` counts model rows consumed, `bytes` the raw
/// encoded weight bytes pulled from storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrunkIoStats {
    pub rows: u64,
    pub reads: u64,
    pub bytes: u64,
}

impl core::fmt::Display for TrunkNvmeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTensor => write!(f, "trunk tensor is not row-addressable"),
            Self::RowOutOfRange => write!(f, "trunk tensor row is out of range"),
            Self::RawScratchTooSmall => write!(f, "trunk raw row scratch is too small"),
            Self::OutputTooSmall => write!(f, "trunk row output is too small"),
            Self::OffsetOverflow => write!(f, "trunk row byte offset overflowed"),
            Self::Io => write!(f, "trunk row I/O failed"),
            Self::Dequantization => write!(f, "trunk row dequantization failed"),
        }
    }
}

impl std::error::Error for TrunkNvmeError {}

/// One encoded-weight source cursor. It owns no model payload; caller buffers
/// bound each individual read/dequantization operation.  A fixed 4 MiB read
/// chunk amortizes syscalls across consecutive rows without holding model
/// weights; it carries raw bytes only.
///
/// In mapped mode (`open_mapped`) the file is exposed through a read-only
/// `memmap2` view so row access slices the OS page cache instead of issuing
/// syscalls.  `stats.reads` then counts real syscalls (zero), while rows and
/// bytes still record the logical data volume touched.
pub struct TrunkNvmeReader {
    file: source::WeightCursor,
    map: Option<source::WeightMap>,
    tensor_data_start: u64,
    chunk: Vec<u8>,
    stats: TrunkIoStats,
    parallel: parallel::ParallelScratch,
}

const TRUNK_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// f32 dot product accumulated through eight independent lanes so LLVM can
/// vectorize the multiply-accumulate; a strict left-to-right fold cannot be
/// reassociated under IEEE rules and stays scalar.  Lane order changes the
/// result's last-ulp rounding only.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];
    let mut acc = [0.0f32; 8];
    let mut index = 0;
    while index + 8 <= n {
        for lane in 0..8 {
            acc[lane] += a[index + lane] * b[index + lane];
        }
        index += 8;
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    while index < n {
        sum += a[index] * b[index];
        index += 1;
    }
    sum
}

pub(crate) mod source;
mod gemv;
mod parallel;
mod projection;
#[allow(unsafe_code)]
mod rows;
#[cfg(test)]
mod tests;
