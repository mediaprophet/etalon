//! Optional Qualcomm QNN HTP inference for trusted sensing models.
//!
//! This requests a heterogeneous tensor accelerator, not a general-purpose
//! DSP kernel. ORT CPU graph fallback is disabled, but this is not hardware
//! attestation and does not describe execution inside the native provider.
//! Runtime and HTP libraries, drivers and a suitable device are host supplied.
//! ORT and provider allocations are outside these input limits; a hard memory
//! or no-egress boundary requires a separately isolated host process. Mapped
//! drives, reparse points, library dependencies and native code are not
//! controlled by lexical path validation. No files are downloaded.
//! Quantized model I/O compatibility with CPU fallback disabled is unverified;
//! the QNN provider's graph I/O quantization option is left at its default.

#![cfg(not(target_arch = "wasm32"))]
#![forbid(unsafe_code)]

use std::path::Path;
pub use etalon_sense_npu::{NpuError, NpuRuntime};
use ort::{ep, session::Session, value::{TensorElementType, TensorRef, ValueType}};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub max_model_bytes: usize,
    pub max_input_elements: usize,
    pub max_output_elements: usize,
}

#[derive(Debug)]
pub enum QnnError {
    UnsupportedPlatform,
    InvalidBackendPath,
    InvalidLimits,
    ModelSize,
    ModelInterface,
    InputSize,
    OutputSize,
    Runtime(String),
    RuntimePanicked,
    Faulted,
    DeploymentAdmission,
}

/// A synchronous fixed-shape f32 model session. Provider work may block.
pub struct QnnHtpSession {
    session: Session,
    input_elements: usize,
    output_elements: usize,
    faulted: bool,
}

impl QnnHtpSession {
    /// Admit the exact QNN artifact set and requested ORT path. The path match
    /// is consistency only; global ORT reuse and native dependencies still
    /// require the host's protected artifact and process initialization policy.
    pub fn prepare_from_deployment(runtime_handle: &NpuRuntime,
        deployment: &etalon_sense_deployment::AdmittedDeployment<'_, '_, '_>,
        model: &[u8], limits: ExecutionLimits) -> Result<Self, QnnError> {
        if deployment.provider_kind() != etalon_sense_deployment::ProviderKind::QnnHtp
            || runtime_handle.requested_library() != deployment.runtime_path() {
            return Err(QnnError::DeploymentAdmission);
        }
        deployment.revalidate_authority().map_err(|_| QnnError::DeploymentAdmission)?;
        let provider = deployment.provider_path().ok_or(QnnError::DeploymentAdmission)?;
        Self::prepare(runtime_handle, provider, model, limits)
    }

    /// The caller must initialize the ORT runtime first, and authorize both
    /// the trusted self-contained model and the selected HTP library.
    pub fn prepare(runtime_handle: &NpuRuntime, backend_library: &Path, model: &[u8],
        limits: ExecutionLimits) -> Result<Self, QnnError> {
        if !cfg!(all(target_arch = "aarch64", any(target_os = "windows", target_os = "linux", target_os = "android"))) {
            return Err(QnnError::UnsupportedPlatform);
        }
        if !local_path_syntax(backend_library) { return Err(QnnError::InvalidBackendPath); }
        let path = backend_library.to_str().filter(|s| !s.contains('\0')).ok_or(QnnError::InvalidBackendPath)?;
        let file = backend_library.file_name().and_then(|name| name.to_str()).ok_or(QnnError::InvalidBackendPath)?;
        if !file.eq_ignore_ascii_case(if cfg!(windows) { "QnnHtp.dll" } else { "libQnnHtp.so" })
            || !backend_library.is_file()
        { return Err(QnnError::InvalidBackendPath); }
        if limits.max_model_bytes == 0 || limits.max_input_elements == 0 || limits.max_output_elements == 0 {
            return Err(QnnError::InvalidLimits);
        }
        if model.is_empty() || model.len() > limits.max_model_bytes { return Err(QnnError::ModelSize); }
        let session = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut builder = Session::builder(runtime_handle.environment()).map_err(runtime)?
                .with_no_environment_execution_providers().map_err(runtime)?
                .with_execution_providers([ep::QNN::default().with_backend_path(path).build().error_on_failure()]).map_err(runtime)?
                .with_disable_cpu_fallback().map_err(runtime)?;
            builder.commit_from_memory(model).map_err(runtime)
        })).map_err(|_| QnnError::RuntimePanicked)??;
        if session.inputs().len() != 1 || session.outputs().len() != 1 { return Err(QnnError::ModelInterface); }
        let input_elements = fixed_f32_vector(session.inputs()[0].dtype(), limits.max_input_elements)?;
        let output_elements = fixed_f32_vector(session.outputs()[0].dtype(), limits.max_output_elements)?;
        Ok(Self { session, input_elements, output_elements, faulted: false })
    }

    pub fn input_elements(&self) -> usize { self.input_elements }
    pub fn output_elements(&self) -> usize { self.output_elements }

    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), QnnError> {
        output.fill(0.0);
        if self.faulted { return Err(QnnError::Faulted); }
        if input.len() != self.input_elements { return Err(QnnError::InputSize); }
        if output.len() != self.output_elements { return Err(QnnError::OutputSize); }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let tensor = TensorRef::from_array_view(([self.input_elements], input)).map_err(runtime)?;
            let values = self.session.run(ort::inputs![tensor]).map_err(runtime)?;
            if values.len() != 1 { return Err(QnnError::ModelInterface); }
            let (shape, data) = values[0].try_extract_tensor::<f32>().map_err(runtime)?;
            if shape.len() != 1 || shape[0] != self.output_elements as i64 || data.len() != output.len() {
                return Err(QnnError::ModelInterface);
            }
            output.copy_from_slice(data);
            Ok(())
        })).unwrap_or(Err(QnnError::RuntimePanicked));
        if result.is_err() { self.faulted = true; output.fill(0.0); }
        result
    }
}

fn fixed_f32_vector(value: &ValueType, limit: usize) -> Result<usize, QnnError> {
    match value {
        ValueType::Tensor { ty: TensorElementType::Float32, shape, .. } if shape.len() == 1 => {
            let count = usize::try_from(shape[0]).map_err(|_| QnnError::ModelInterface)?;
            if count == 0 || count > limit { return Err(QnnError::ModelInterface); }
            Ok(count)
        }
        _ => Err(QnnError::ModelInterface),
    }
}

#[cfg(windows)]
fn local_path_syntax(path: &Path) -> bool {
    use std::path::{Component, Prefix};
    path.is_absolute() && matches!(path.components().next(),
        Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_)))
}

#[cfg(not(windows))]
fn local_path_syntax(path: &Path) -> bool { path.is_absolute() }

fn runtime(error: impl core::fmt::Display) -> QnnError { QnnError::Runtime(error.to_string()) }
