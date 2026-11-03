//! Borrowed native CPU GGUF bindings. This numerical API supplies no authority.
mod bindings;
mod config;
mod kernels;

pub use bindings::{empty_layer, prepare_dense_gguf_into, BindingLimits, PackedTensor};
pub use config::{inspect_dense_gguf, DenseGgufConfig};
pub use kernels::{GgmlCpuKernels, MAX_INPUT_WIDTH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufCpuError {
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
impl From<etalon_model::gguf::ParseError> for GgufCpuError {
    fn from(value: etalon_model::gguf::ParseError) -> Self {
        Self::Metadata(value)
    }
}
impl core::fmt::Display for GgufCpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CPU GGUF: {self:?}")
    }
}
impl core::error::Error for GgufCpuError {}

#[cfg(test)]
mod tests;
