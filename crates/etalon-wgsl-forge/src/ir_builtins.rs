//! Built-in kernel specifications and shape validation for Etalon WGSL Forge.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::ir::{
    BufferAccess, BufferElement, BufferSpec, ForgeError, Intrinsic, KernelSpec, Op, ScalarType,
    SharedLen, SharedMemorySpec,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuiltinKernel {
    AffineF32,
    FusedFfn,
    P64Project,
    TopK,
    RayProbe,
    TernaryGemv,
    Gemm,
    Gemv,
    Fft,
}

impl BuiltinKernel {
    pub const ALL: [Self; 9] = [
        Self::AffineF32,
        Self::FusedFfn,
        Self::P64Project,
        Self::TopK,
        Self::RayProbe,
        Self::TernaryGemv,
        Self::Gemm,
        Self::Gemv,
        Self::Fft,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::AffineF32 => "affine-f32",
            Self::FusedFfn => "fused-ffn",
            Self::P64Project => "p64-project",
            Self::TopK => "topk",
            Self::RayProbe => "ray-probe",
            Self::TernaryGemv => "ternary-gemv",
            Self::Gemm => "gemm",
            Self::Gemv => "gemv",
            Self::Fft => "fft",
        }
    }

    pub const fn has_gpu_oracle(self) -> bool {
        true
    }

    pub fn spec(self) -> KernelSpec {
        match self {
            Self::AffineF32 => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 1,
                entry_point: "affine_f32".to_string(),
                description: "out[i] = input[i] * scale + bias".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "input",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "output",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                    buf(
                        2,
                        "params",
                        BufferElement::AffineParams,
                        BufferAccess::Uniform,
                    ),
                ],
                ops: vec![
                    Op::StructLoad {
                        buffer: "params".to_string(),
                        field: "scale".to_string(),
                        destination: "scale".to_string(),
                    },
                    Op::StructLoad {
                        buffer: "params".to_string(),
                        field: "bias".to_string(),
                        destination: "bias".to_string(),
                    },
                    Op::Load {
                        buffer: "input".to_string(),
                        index: "global_id".to_string(),
                        destination: "in_val".to_string(),
                    },
                    Op::Fma {
                        a: "in_val".to_string(),
                        b: "scale".to_string(),
                        c: "bias".to_string(),
                        destination: "out_val".to_string(),
                    },
                    Op::Store {
                        buffer: "output".to_string(),
                        index: "global_id".to_string(),
                        value: "out_val".to_string(),
                    },
                ],
                shared_memory: Vec::new(),
            },
            Self::FusedFfn => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 2,
                entry_point: "fused_ffn".to_string(),
                description: "out[o] = sum_h w2[o,h] * gelu(sum_i w1[h,i] * in[i])".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "input",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "w1",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        2,
                        "w2",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        3,
                        "output",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                    buf(
                        4,
                        "params",
                        BufferElement::AffineParams,
                        BufferAccess::Uniform,
                    ),
                ],
                ops: vec![Op::Gelu {
                    operand: "hv".to_string(),
                    destination: "g".to_string(),
                }],
                shared_memory: Vec::new(),
            },
            Self::P64Project => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 2,
                entry_point: "p64_project".to_string(),
                description: "out[r] = sum_w weights[w] * f32(p64[r].word[w])".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "input",
                        BufferElement::P64Words64,
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "weights",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        2,
                        "output",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                ],
                ops: Vec::new(),
                shared_memory: Vec::new(),
            },
            Self::TopK => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 1,
                entry_point: "topk".to_string(),
                description: "out[block*k + i] = i-th largest of input[block]".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "input",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "output",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                    buf(
                        2,
                        "params",
                        BufferElement::AffineParams,
                        BufferAccess::Uniform,
                    ),
                ],
                ops: vec![Op::Barrier],
                shared_memory: vec![
                    shared("s_val", ScalarType::F32),
                    shared("s_idx", ScalarType::U32),
                    shared("r_val", ScalarType::F32),
                    shared("r_idx", ScalarType::U32),
                ],
            },
            Self::RayProbe => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 1,
                entry_point: "ray_probe".to_string(),
                description: "hits[i] = ray_query(scene, rays[i]) committed t or -1".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "scene",
                        BufferElement::AccelerationStructure,
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "rays",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        2,
                        "hits",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                ],
                ops: vec![Op::Intrinsic(Intrinsic::RayQuery {
                    acceleration_structure: "scene".to_string(),
                    origin: "origin".to_string(),
                    direction: "direction".to_string(),
                    t_min: "t_min".to_string(),
                    t_max: "t_max".to_string(),
                    destination: "hit_t".to_string(),
                })],
                shared_memory: Vec::new(),
            },
            Self::TernaryGemv => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 1,
                entry_point: "ternary_gemv".to_string(),
                description: "out[o] = scale[o] * sum_i ternary(w[o,i]) * x[i]".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "x",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "w_packed",
                        BufferElement::Scalar(ScalarType::U32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        2,
                        "scale",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        3,
                        "output",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                    buf(
                        4,
                        "params",
                        BufferElement::AffineParams,
                        BufferAccess::Uniform,
                    ),
                ],
                ops: Vec::new(),
                shared_memory: Vec::new(),
            },
            Self::Gemm => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 1,
                entry_point: "gemm".to_string(),
                description: "C[i][j] = sum_k A[i*K+k] * B[k*N+j]".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "a",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "b",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        2,
                        "c",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                    buf(
                        3,
                        "params",
                        BufferElement::AffineParams,
                        BufferAccess::Uniform,
                    ),
                ],
                ops: Vec::new(),
                shared_memory: Vec::new(),
            },
            Self::Gemv => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 1,
                entry_point: "gemv".to_string(),
                description: "y[i] = sum_j a[i*N+j] * x[j]".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "a",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "x",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        2,
                        "y",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                    buf(
                        3,
                        "params",
                        BufferElement::AffineParams,
                        BufferAccess::Uniform,
                    ),
                ],
                ops: Vec::new(),
                shared_memory: Vec::new(),
            },
            Self::Fft => KernelSpec {
                id: self.name().to_string(),
                semantic_version: 1,
                entry_point: "fft".to_string(),
                description: "out = forward DFT(in) via workgroup radix-2 DIT".to_string(),
                buffers: vec![
                    buf(
                        0,
                        "input",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageRead,
                    ),
                    buf(
                        1,
                        "output",
                        BufferElement::Scalar(ScalarType::F32),
                        BufferAccess::StorageReadWrite,
                    ),
                    buf(
                        2,
                        "params",
                        BufferElement::AffineParams,
                        BufferAccess::Uniform,
                    ),
                ],
                ops: vec![Op::Barrier],
                shared_memory: vec![
                    shared("s_re", ScalarType::F32),
                    shared("s_im", ScalarType::F32),
                ],
            },
        }
    }
}

fn buf(binding: u32, name: &str, element: BufferElement, access: BufferAccess) -> BufferSpec {
    BufferSpec {
        group: 0,
        binding,
        name: name.to_string(),
        element,
        access,
    }
}

fn shared(name: &str, element: ScalarType) -> SharedMemorySpec {
    SharedMemorySpec {
        name: name.to_string(),
        element,
        length: SharedLen::WorkgroupSize,
    }
}

impl FromStr for BuiltinKernel {
    type Err = ForgeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "affine-f32" | "affine_f32" | "affine" => Ok(Self::AffineF32),
            "fused-ffn" | "fused_ffn" | "ffn" => Ok(Self::FusedFfn),
            "p64-project" | "p64_project" | "p64" => Ok(Self::P64Project),
            "topk" | "top-k" | "top_k" => Ok(Self::TopK),
            "ray-probe" | "ray_probe" | "rayquery" | "ray_query" | "rt" => Ok(Self::RayProbe),
            "ternary-gemv" | "ternary_gemv" | "ternary" | "bitnet" => Ok(Self::TernaryGemv),
            "gemm" | "matmul" | "dense-gemm" | "dense_gemm" => Ok(Self::Gemm),
            "gemv" | "matvec" => Ok(Self::Gemv),
            "fft" | "dft" | "radix2" | "radix-2" => Ok(Self::Fft),
            other => Err(ForgeError::UnknownKernel(other.to_string())),
        }
    }
}

pub(crate) fn validate_builtin_shape(spec: &KernelSpec) -> Result<(), ForgeError> {
    if spec.id == BuiltinKernel::AffineF32.name() || spec.id == BuiltinKernel::TopK.name() {
        let expected = [
            (
                "input",
                0,
                BufferElement::Scalar(ScalarType::F32),
                BufferAccess::StorageRead,
            ),
            (
                "output",
                1,
                BufferElement::Scalar(ScalarType::F32),
                BufferAccess::StorageReadWrite,
            ),
            (
                "params",
                2,
                BufferElement::AffineParams,
                BufferAccess::Uniform,
            ),
        ];
        check_bindings(&spec.id, &spec.buffers, &expected)?;
    } else if spec.id == BuiltinKernel::FusedFfn.name() {
        let expected = [
            (
                "input",
                0,
                BufferElement::Scalar(ScalarType::F32),
                BufferAccess::StorageRead,
            ),
            (
                "w1",
                1,
                BufferElement::Scalar(ScalarType::F32),
                BufferAccess::StorageRead,
            ),
            (
                "w2",
                2,
                BufferElement::Scalar(ScalarType::F32),
                BufferAccess::StorageRead,
            ),
            (
                "output",
                3,
                BufferElement::Scalar(ScalarType::F32),
                BufferAccess::StorageReadWrite,
            ),
            (
                "params",
                4,
                BufferElement::AffineParams,
                BufferAccess::Uniform,
            ),
        ];
        check_bindings(&spec.id, &spec.buffers, &expected)?;
    }
    Ok(())
}

fn check_bindings(
    kernel_name: &str,
    buffers: &[BufferSpec],
    expected: &[(&str, u32, BufferElement, BufferAccess)],
) -> Result<(), ForgeError> {
    for &(name, binding, element, access) in expected {
        let Some(buffer) = buffers
            .iter()
            .find(|candidate| candidate.group == 0 && candidate.binding == binding)
        else {
            return Err(ForgeError::InvalidKernel(format!(
                "{kernel_name} requires group 0 binding {binding}"
            )));
        };
        if buffer.name != name || buffer.element != element || buffer.access != access {
            return Err(ForgeError::InvalidKernel(format!(
                "{kernel_name} binding {binding} must be {name:?} / {element:?} / {access:?}"
            )));
        }
    }
    Ok(())
}
