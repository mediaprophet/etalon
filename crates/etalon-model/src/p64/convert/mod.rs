//! Cold P64 numerical conversion and compatibility serialization.
use super::build::write::write_entry as write_tensor_entry;
use super::integrity::crc32c;
use super::layout::*;
use crate::inference::TensorInfo as GgufTensorInfo;
pub mod codecs;
mod converted;
mod ffn;
mod legacy;
mod quantized;
mod vision;
pub use converted::compile_converted;
pub use ffn::{requantize_ffn, FfnQuant};
pub use legacy::{compile_legacy, LegacyInput, LegacyTensor};
pub use quantized::{transcode_quantized, QuantizedTensor, SafetensorQuantization};
pub use vision::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertLayout {
    F16Expand,
    Q4kSoa,
}
#[derive(Clone, Copy, Debug)]
pub struct ConversionBudget {
    pub output_bytes: usize,
    pub scratch_bytes: usize,
}
impl Default for ConversionBudget {
    fn default() -> Self {
        Self {
            output_bytes: u32::MAX as usize,
            scratch_bytes: 256 * 1024 * 1024,
        }
    }
}
impl ConversionBudget {
    fn output(self, bytes: usize) -> Result<(), String> {
        if bytes > self.output_bytes {
            return Err(format!(
                "p64: output budget exceeded: {bytes} > {}",
                self.output_bytes
            ));
        }
        Ok(())
    }
    fn scratch(self, bytes: usize) -> Result<(), String> {
        if bytes > self.scratch_bytes {
            return Err(format!(
                "p64: scratch budget exceeded: {bytes} > {}",
                self.scratch_bytes
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct ConversionInput<'a> {
    pub tensor_data_start: usize,
    pub tensors: &'a [(u16, u16, u64, GgufTensorInfo)],
    pub names: &'a [String],
    pub hparams: P64HParams,
    pub tokenizer: &'a [u8],
    pub header_reserved: [u8; 20],
    pub manifold_bytes: &'a [u8],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscodeReport {
    pub n_tensors: usize,
    pub bytes_written: usize,
    pub largest_tensor_bytes: usize,
    pub total_tensor_bytes: usize,
    pub peak_working_bytes: usize,
}
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}
fn q_hash(s: &str) -> u64 {
    s.as_bytes().iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

fn workspace_sum(values: &[usize]) -> Result<usize, String> {
    values
        .iter()
        .try_fold(0usize, |n, v| n.checked_add(*v))
        .ok_or_else(|| "p64: workspace overflow".into())
}
fn names_len<'a>(mut names: impl Iterator<Item = &'a str>) -> Result<usize, String> {
    names
        .try_fold(1usize, |n, s| n.checked_add(s.len())?.checked_add(1))
        .ok_or_else(|| "p64: string table overflow".into())
}
fn write_zeros(out: &mut impl std::io::Write, mut count: usize) -> Result<(), String> {
    let zeros = [0u8; 4096];
    while count > 0 {
        let n = count.min(zeros.len());
        out.write_all(&zeros[..n]).map_err(|e| e.to_string())?;
        count -= n;
    }
    Ok(())
}
