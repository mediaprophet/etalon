use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::intrinsics::Intrinsic;
use crate::numerical::ForgeError;

pub use crate::P64GpuWords64;

pub use crate::{BufferAccess, BufferElement, BufferSpec, ScalarType, SharedLen, SharedMemorySpec};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// Workgroup execution + shared-memory barrier (WGSL `workgroupBarrier()`).
    Barrier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSpec {
    pub id: String,
    pub semantic_version: u32,
    pub entry_point: String,
    pub description: String,
    pub buffers: Vec<BufferSpec>,
    pub ops: Vec<Op>,
    /// Workgroup-shared arrays the kernel uses. Skipped during serialization
    /// when empty so existing kernels' semantic hashes are unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shared_memory: Vec<SharedMemorySpec>,
}

impl KernelSpec {
    /// Validates the kernel spec.
    pub fn validate(&self) -> Result<(), ForgeError> {
        super::builtin::iq::validate(self)?;
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

        // Validate specific kernel shapes based on their name for backwards compatibility or hardcoded constraints
        if self.id == BuiltinKernel::AffineF32.name() {
            validate_affine_buffers(&self.buffers)?;
        }

        if self.id == BuiltinKernel::FusedFfn.name() {
            validate_fused_ffn_buffers(&self.buffers)?;
        }

        if self.id == BuiltinKernel::TopK.name() {
            validate_topk_buffers(&self.buffers)?;
        }

        Ok(())
    }

    /// Content hash of the kernel spec.
    pub fn semantic_hash(&self) -> Result<String, ForgeError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    /// Hardware intrinsics this kernel requires, gathered recursively through
    /// loop bodies. The capability checker uses these to prune schedules on
    /// adapters lacking the necessary hardware (plan §6).
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

fn validate_affine_buffers(buffers: &[BufferSpec]) -> Result<(), ForgeError> {
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
    for (name, binding, element, access) in expected {
        let Some(buffer) = buffers
            .iter()
            .find(|candidate| candidate.group == 0 && candidate.binding == binding)
        else {
            return Err(ForgeError::InvalidKernel(format!(
                "affine_f32 requires group 0 binding {binding}"
            )));
        };
        if buffer.name != name || buffer.element != element || buffer.access != access {
            return Err(ForgeError::InvalidKernel(format!(
                "affine_f32 binding {binding} must be {name:?} / {element:?} / {access:?}"
            )));
        }
    }
    Ok(())
}

fn validate_fused_ffn_buffers(buffers: &[BufferSpec]) -> Result<(), ForgeError> {
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
    for (name, binding, element, access) in expected {
        let Some(buffer) = buffers
            .iter()
            .find(|candidate| candidate.group == 0 && candidate.binding == binding)
        else {
            return Err(ForgeError::InvalidKernel(format!(
                "fused_ffn requires group 0 binding {binding}"
            )));
        };
        if buffer.name != name || buffer.element != element || buffer.access != access {
            return Err(ForgeError::InvalidKernel(format!(
                "fused_ffn binding {binding} must be {name:?} / {element:?} / {access:?}"
            )));
        }
    }
    Ok(())
}

fn validate_topk_buffers(buffers: &[BufferSpec]) -> Result<(), ForgeError> {
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
    for (name, binding, element, access) in expected {
        let Some(buffer) = buffers
            .iter()
            .find(|candidate| candidate.group == 0 && candidate.binding == binding)
        else {
            return Err(ForgeError::InvalidKernel(format!(
                "topk requires group 0 binding {binding}"
            )));
        };
        if buffer.name != name || buffer.element != element || buffer.access != access {
            return Err(ForgeError::InvalidKernel(format!(
                "topk binding {binding} must be {name:?} / {element:?} / {access:?}"
            )));
        }
    }
    Ok(())
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|value| value == '_' || value.is_ascii_alphanumeric())
}

pub use super::builtin::BuiltinKernel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_kernel_is_typed_and_hash_stable() {
        let kernel = BuiltinKernel::AffineF32.spec();
        kernel.validate().expect("valid builtin");
        assert_eq!(
            kernel.semantic_hash().unwrap(),
            kernel.semantic_hash().unwrap()
        );
        assert_eq!(ScalarType::U64Words.wgsl_name(), "vec2<u32>");
        assert_eq!(BufferElement::P64Words64.size_bytes(), 64);
        assert_eq!(size_of::<P64GpuWords64>(), 64);
        assert_eq!(align_of::<P64GpuWords64>(), 16);
        let fields = [
            0,
            1,
            u32::MAX as u64 + 1,
            u64::MAX,
            0x0123_4567_89AB_CDEF,
            5,
            6,
            7,
        ];
        let record = P64GpuWords64::from_u64_fields(fields);
        for (index, expected) in fields.into_iter().enumerate() {
            assert_eq!(record.u64_field(index), Some(expected));
        }
        assert_eq!(record.u64_field(8), None);
    }

    #[test]
    fn duplicate_bindings_are_rejected() {
        let mut kernel = BuiltinKernel::AffineF32.spec();
        kernel.buffers[1].binding = 0;
        assert!(kernel.validate().is_err());
    }
}
