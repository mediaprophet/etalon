//! Pure operation IR and tensor descriptors decoupled from database storage.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ForgeError {
    #[error("Emission error: {0}")]
    Emission(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Hardware unavailable: {0}")]
    GpuUnavailable(String),
}

/// Primitive tensor data type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U32,
    I8,
    U8,
    Q4K,
    Q8_0,
}

/// 64-byte aligned GPU descriptor container for P64 weight records.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct P64GpuWords64 {
    pub words: [u32; 16],
}

/// Validated multi-dimensional tensor view descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorViewDesc {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub block_size: usize,
}

impl TensorViewDesc {
    pub fn new_contiguous(shape: Vec<usize>, dtype: DType) -> Self {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1usize;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride = stride.saturating_mul(dim);
        }
        strides.reverse();
        Self {
            dtype,
            shape,
            strides,
            block_size: 1,
        }
    }
}

/// Core computational operators lowerable across backends.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    MatMul { tc: bool, transpose_b: bool },
    Gemv { tc: bool },
    ElementwiseAdd,
    ElementwiseMul,
    Silu,
    RmsNorm { eps: f32 },
    RoPE { theta: f32 },
    PrefixScan,
    RadixHistogram { shift: u32 },
    RadixScatter { shift: u32 },
}
