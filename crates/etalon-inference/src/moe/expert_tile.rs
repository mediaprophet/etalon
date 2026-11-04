//! Generic row-addressed gate/up/down tiles. Descriptor format names remain
//! historical Qwen identifiers for compatibility with existing HMC artifacts.

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
pub const QWEN_EXPERT_TILE_FORMAT: &str = "qualia.qwen4exp.expert-tile.v1";
pub const QWEN_EXPERT_TILE_SUFFIX: &str = ".qwen-expert.hmc";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertPlane {
    pub dims: [u64; 4],
    pub n_dims: u32,
    pub ggml_type: u32,
    pub source_byte_offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertTileDescriptor {
    pub format: String,
    pub layer: u16,
    pub expert: u16,
    pub source_file_name: String,
    pub source_byte_len: u64,
    pub gate: ExpertPlane,
    pub up: ExpertPlane,
    pub down: ExpertPlane,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertTileAdmission {
    pub cache_used_bytes: u64,
    pub requested_bytes: u64,
    pub cache_limit_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertTileError {
    InvalidTensor,
    ExpertOutOfRange,
    OffsetOverflow,
    SourceIo,
    OutputExists,
    OutputPath,
    CacheBudgetExceeded,
    Bundle,
    Serialize,
    Write,
    InvalidPackage,
    BufferTooSmall,
}

impl core::fmt::Display for ExpertTileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTensor => write!(
                f,
                "expert tensor is not a supported [input, output, expert] GGML plane"
            ),
            Self::ExpertOutOfRange => {
                write!(f, "expert id is outside the tensor's expert dimension")
            }
            Self::OffsetOverflow => write!(f, "expert tile byte range overflowed"),
            Self::SourceIo => write!(f, "could not read source GGUF expert plane"),
            Self::OutputExists => write!(f, "refusing to overwrite an existing Qwen expert tile"),
            Self::OutputPath => write!(f, "Qwen expert tile output path is invalid"),
            Self::CacheBudgetExceeded => write!(
                f,
                "Qwen expert tile would exceed the configured cache budget"
            ),
            Self::Bundle => write!(f, "could not build Qwen expert HMC tile"),
            Self::Serialize => write!(f, "could not serialize Qwen expert tile descriptor"),
            Self::Write => write!(f, "could not write Qwen expert tile"),
            Self::InvalidPackage => write!(f, "Qwen expert tile HMC is invalid"),
            Self::BufferTooSmall => write!(f, "Qwen expert tile caller buffer is too small"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertTilePlaneKind {
    Gate,
    Up,
    Down,
}

/// A validated C:-resident expert tile. Its mapping is bounded by one explicit
/// promotion artifact, never the original multi-gigabyte GGUF.
pub struct ExpertTileReader {
    mapped: Mmap,
    pub descriptor: ExpertTileDescriptor,
    gate_offset: usize,
    up_offset: usize,
    down_offset: usize,
}

impl std::error::Error for ExpertTileError {}

impl ExpertTileReader {
    pub fn from_mapping(
        mapped: Mmap,
        descriptor: ExpertTileDescriptor,
        offsets: [usize; 3],
    ) -> Result<Self, ExpertTileError> {
        for (plane, offset) in [descriptor.gate, descriptor.up, descriptor.down]
            .iter()
            .zip(offsets)
        {
            if plane.n_dims != 2 || plane.dims[0] == 0 || plane.dims[1] == 0 || plane.bytes == 0 {
                return Err(ExpertTileError::InvalidPackage);
            }
            let bytes =
                usize::try_from(plane.bytes).map_err(|_| ExpertTileError::OffsetOverflow)?;
            if offset
                .checked_add(bytes)
                .filter(|end| *end <= mapped.len())
                .is_none()
            {
                return Err(ExpertTileError::InvalidPackage);
            }
        }
        Ok(Self {
            mapped,
            descriptor,
            gate_offset: offsets[0],
            up_offset: offsets[1],
            down_offset: offsets[2],
        })
    }
    fn plane(&self, kind: ExpertTilePlaneKind) -> (ExpertPlane, usize) {
        match kind {
            ExpertTilePlaneKind::Gate => (self.descriptor.gate, self.gate_offset),
            ExpertTilePlaneKind::Up => (self.descriptor.up, self.up_offset),
            ExpertTilePlaneKind::Down => (self.descriptor.down, self.down_offset),
        }
    }

    /// Compute a cached expert plane directly from its C: HMC segment. Caller
    /// storage bounds every dequantized row and output vector.
    pub fn gemv_into(
        &self,
        kind: ExpertTilePlaneKind,
        input: &[f32],
        out: &mut [f32],
        row_scratch: &mut [f32],
    ) -> Result<usize, ExpertTileError> {
        let (plane, offset) = self.plane(kind);
        let width = plane.dims[0] as usize;
        let rows = plane.dims[1] as usize;
        let row_bytes = etalon_model::ggml::ggml_row_bytes(plane.ggml_type, width)
            .ok_or(ExpertTileError::InvalidPackage)?;
        if input.len() < width || out.len() < rows || row_scratch.len() < width {
            return Err(ExpertTileError::BufferTooSmall);
        }
        for row in 0..rows {
            let start = offset
                .checked_add(
                    row.checked_mul(row_bytes)
                        .ok_or(ExpertTileError::OffsetOverflow)?,
                )
                .ok_or(ExpertTileError::OffsetOverflow)?;
            let end = start
                .checked_add(row_bytes)
                .ok_or(ExpertTileError::OffsetOverflow)?;
            if end > self.mapped.len() {
                return Err(ExpertTileError::InvalidPackage);
            }
            etalon_model::ggml::dequantize_row_into(
                &self.mapped[start..end],
                plane.ggml_type,
                width,
                &mut row_scratch[..width],
            )
            .map_err(|_| ExpertTileError::InvalidPackage)?;
            let mut sum = 0.0f32;
            for index in 0..width {
                sum += row_scratch[index] * input[index];
            }
            out[row] = sum;
        }
        Ok(rows)
    }
}
