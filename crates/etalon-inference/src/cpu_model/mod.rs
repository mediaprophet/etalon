//! Borrowed CPU model bindings. This numerical API supplies no authority.
mod bindings;
mod config;
mod kernels;
pub mod sequence;

pub use bindings::{empty_layer, prepare_dense_gguf_into, BindingLimits, PackedTensor};
pub use config::{inspect_dense_gguf, DenseGgufConfig};
pub use kernels::{GgmlCpuKernels, MAX_INPUT_WIDTH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseCpuError {
    Metadata(etalon_model::gguf::ParseError),
    MissingMetadata,
    DuplicateMetadata,
    UnsupportedProfile,
    InvalidDimensions,
    MissingTensor,
    DuplicateTensor,
    UnsupportedTensor,
    Payload,
    Alignment,
    ArithmeticOverflow,
    Capacity,
    Budget,
    Kernel,
}
impl From<etalon_model::gguf::ParseError> for DenseCpuError {
    fn from(value: etalon_model::gguf::ParseError) -> Self {
        Self::Metadata(value)
    }
}
impl core::fmt::Display for DenseCpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CPU GGUF: {self:?}")
    }
}
impl core::error::Error for DenseCpuError {}

#[cfg(test)]
mod tests;

pub type GgufCpuError = DenseCpuError;
mod native;
pub use native::prepare_dense_p64_into;
