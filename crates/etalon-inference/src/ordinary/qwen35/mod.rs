//! Explicitly admitted Qwen3.5 linear-layer execution over caller-owned buffers.
mod execute;
pub mod fixture;
pub mod full_attention;
pub mod model_execute;
pub mod model_plan;
pub mod state;
pub mod weights;
mod workspace;

use etalon_inference_kernel::recurrent::QwenDeltaError;
use etalon_model::inference::{
    qwen_roles::{Projection, Qwen35RoleError},
    TensorInfo,
};
pub use execute::{execute_normalized, ordinary_rms_norm, zero_centered_rms_norm};
pub use workspace::WorkspaceLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Shape,
    Overflow,
    Workspace,
    UnsupportedProjection,
    Provider,
    Numerical(QwenDeltaError),
    Roles(Qwen35RoleError),
}
impl From<QwenDeltaError> for Error {
    fn from(value: QwenDeltaError) -> Self {
        Self::Numerical(value)
    }
}

/// GEMM must honor `row_scale`, including its numerical interpretation, or reject admission.
/// Calls may modify only `out`; provider state, live hidden, and recurrent state are not exposed.
pub trait Host {
    fn admits(&self, projection: &Projection) -> bool;
    fn gemm(&self, projection: &Projection, input: &[f32], out: &mut [f32]) -> Result<(), Error>;
    /// Decode the exact tensor into `out`. Convolution weights are channel-major,
    /// with oldest-to-current kernel taps. Return an error for unsupported encodings.
    fn read(&self, tensor: &TensorInfo, out: &mut [f32]) -> Result<(), Error>;
}

fn finite(values: &[f32]) -> Result<(), Error> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::Numerical(QwenDeltaError::NonFinite))
    }
}
