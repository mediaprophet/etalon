//! Pure operation IR and tensor descriptors decoupled from database storage.

use std::collections::BTreeSet;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::ir_builtins::BuiltinKernel;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ForgeError {
    #[error("unknown kernel: {0}")]
    UnknownKernel(String),
    #[error("invalid kernel: {0}")]
    InvalidKernel(String),
    #[error("invalid schedule: {0}")]
    InvalidSchedule(String),
    #[error("WGSL emission: {0}")]
    Emission(String),
    #[error("WGSL parse: {0}")]
    WgslParse(String),
    #[error("WGSL validation: {0}")]
    WgslValidation(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("GPU unavailable: {0}")]
    GpuUnavailable(String),
    #[error("GPU validation: {0}")]
    GpuValidation(String),
    #[error("device lost: {0}")]
    DeviceLost(String),
    #[error("oracle mismatch: {0}")]
    OracleMismatch(String),
    #[error("serialization: {0}")]
    Serialization(String),
    #[error("I/O: {0}")]
    Io(String),
}

impl From<std::io::Error> for ForgeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for ForgeError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

/// Primitive tensor data type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct P64GpuWords64 {
    pub lanes: [[u32; 4]; 4],
}

impl P64GpuWords64 {
    pub fn from_words(words: [u32; 16]) -> Self {
        Self {
            lanes: [
                [words[0], words[1], words[2], words[3]],
                [words[4], words[5], words[6], words[7]],
                [words[8], words[9], words[10], words[11]],
                [words[12], words[13], words[14], words[15]],
            ],
        }
    }

    pub fn words(&self) -> &[u32; 16] {
        bytemuck::cast_ref(self)
    }

    pub fn from_u64_fields(fields: [u64; 8]) -> Self {
        let mut words = [0u32; 16];
        for (index, field) in fields.into_iter().enumerate() {
            words[index * 2] = field as u32;
            words[index * 2 + 1] = (field >> 32) as u32;
        }
        Self::from_words(words)
    }

    pub fn u64_field(&self, index: usize) -> Option<u64> {
        if index >= 8 {
            return None;
        }
        let words = self.words();
        Some(words[index * 2] as u64 | ((words[index * 2 + 1] as u64) << 32))
    }
}

/// Validated multi-dimensional tensor view descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    F32,
    U32,
    I32,
    U64Words,
}

impl ScalarType {
    pub const fn wgsl_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64Words => "vec2<u32>",
        }
    }

    pub const fn size_bytes(self) -> u32 {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::U64Words => 8,
        }
    }

    pub const fn alignment_bytes(self) -> u32 {
        self.size_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferElement {
    Scalar(ScalarType),
    AffineParams,
    P64Words64,
    AccelerationStructure,
}

impl BufferElement {
    pub const fn size_bytes(self) -> u32 {
        match self {
            Self::Scalar(value) => value.size_bytes(),
            Self::AffineParams => 16,
            Self::P64Words64 => 64,
            Self::AccelerationStructure => 0,
        }
    }

    pub const fn alignment_bytes(self) -> u32 {
        match self {
            Self::Scalar(value) => value.alignment_bytes(),
            Self::AffineParams | Self::P64Words64 => 16,
            Self::AccelerationStructure => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferAccess {
    StorageRead,
    StorageReadWrite,
    Uniform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedLen {
    Fixed(u32),
    WorkgroupSize,
}

impl SharedLen {
    pub const fn resolve(self, workgroup_size: u32) -> u32 {
        match self {
            Self::Fixed(value) => value,
            Self::WorkgroupSize => workgroup_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SharedMemorySpec {
    pub name: String,
    pub element: ScalarType,
    pub length: SharedLen,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferSpec {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub element: BufferElement,
    pub access: BufferAccess,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubgroupReduceOp {
    Add,
    Mul,
    Min,
    Max,
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicClass {
    Subgroup,
    CooperativeMatrix,
    RayTracing,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intrinsic {
    SubgroupReduce {
        op: SubgroupReduceOp,
    },
    SubgroupShuffle {
        delta: u32,
    },
    SubgroupBallot,
    CoopMatMul {
        m: u32,
        n: u32,
        k: u32,
    },
    RayQuery {
        acceleration_structure: String,
        origin: String,
        direction: String,
        t_min: String,
        t_max: String,
        destination: String,
    },
}

impl Intrinsic {
    pub const fn class(&self) -> IntrinsicClass {
        match self {
            Self::SubgroupReduce { .. } | Self::SubgroupShuffle { .. } | Self::SubgroupBallot => {
                IntrinsicClass::Subgroup
            }
            Self::CoopMatMul { .. } => IntrinsicClass::CooperativeMatrix,
            Self::RayQuery { .. } => IntrinsicClass::RayTracing,
        }
    }
}

/// Core computational operators lowerable across backends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Load {
        buffer: String,
        index: String,
        destination: String,
    },
    Store {
        buffer: String,
        index: String,
        value: String,
    },
    Mul {
        left: String,
        right: String,
        destination: String,
    },
    Add {
        left: String,
        right: String,
        destination: String,
    },
    Fma {
        a: String,
        b: String,
        c: String,
        destination: String,
    },
    Intrinsic(Intrinsic),
    StructLoad {
        buffer: String,
        field: String,
        destination: String,
    },
    Loop {
        induction_var: String,
        start: String,
        end: String,
        step: String,
        body: Vec<Op>,
    },
    DotProduct {
        left_buffer: String,
        left_base: String,
        right_buffer: String,
        right_base: String,
        len: String,
        destination: String,
    },
    Relu {
        operand: String,
        destination: String,
    },
    Gelu {
        operand: String,
        destination: String,
    },
    MatrixMultiply {
        left_buffer: String,
        right_buffer: String,
        destination: String,
        m: String,
        n: String,
        k: String,
    },
    Barrier,
    // High-level macro operators for direct `emit_kernel` dispatch
    MatMul {
        tc: bool,
        transpose_b: bool,
    },
    Gemv {
        tc: bool,
    },
    ElementwiseAdd,
    ElementwiseMul,
    Silu,
    RmsNorm {
        eps: f32,
    },
    RoPE {
        theta: f32,
    },
    PrefixScan,
    RadixHistogram {
        shift: u32,
    },
    RadixScatter {
        shift: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelSpec {
    pub id: String,
    pub semantic_version: u32,
    pub entry_point: String,
    pub description: String,
    pub buffers: Vec<BufferSpec>,
    pub ops: Vec<Op>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shared_memory: Vec<SharedMemorySpec>,
}

impl KernelSpec {
    pub fn validate(&self) -> Result<(), ForgeError> {
        if self.id.is_empty() || self.entry_point.is_empty() {
            return Err(ForgeError::InvalidKernel(
                "kernel id and entry point must not be empty".to_string(),
            ));
        }
        if !is_identifier(&self.entry_point) {
            return Err(ForgeError::InvalidKernel(format!(
                "entry point {:?} is not a WGSL identifier",
                self.entry_point
            )));
        }
        let mut bindings = BTreeSet::new();
        let mut names = BTreeSet::new();
        for buffer in &self.buffers {
            if !is_identifier(&buffer.name) {
                return Err(ForgeError::InvalidKernel(format!(
                    "buffer name {:?} is not a WGSL identifier",
                    buffer.name
                )));
            }
            if !bindings.insert((buffer.group, buffer.binding)) {
                return Err(ForgeError::InvalidKernel(format!(
                    "duplicate binding @group({}) @binding({})",
                    buffer.group, buffer.binding
                )));
            }
            if !names.insert(buffer.name.as_str()) {
                return Err(ForgeError::InvalidKernel(format!(
                    "duplicate buffer name {:?}",
                    buffer.name
                )));
            }
        }
        crate::ir_builtins::validate_builtin_shape(self)
    }

    pub fn semantic_hash(&self) -> Result<String, ForgeError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    pub fn required_intrinsics(&self) -> Vec<Intrinsic> {
        let mut found = Vec::new();
        collect_intrinsics(&self.ops, &mut found);
        found
    }
}

fn collect_intrinsics(ops: &[Op], out: &mut Vec<Intrinsic>) {
    for op in ops {
        match op {
            Op::Intrinsic(intrinsic) => out.push(intrinsic.clone()),
            Op::Loop { body, .. } => collect_intrinsics(body, out),
            _ => {}
        }
    }
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|value| value == '_' || value.is_ascii_alphanumeric())
}
