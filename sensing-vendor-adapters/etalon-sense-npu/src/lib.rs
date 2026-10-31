//! OpenVINO NPU execution for caller-supplied sensing models.
//!
//! The caller must initialize `ort` with a trusted ONNX Runtime library built
//! with OpenVINO support before creating a session. OpenVINO and a compatible
//! NPU driver must be installed separately. No runtime or model is downloaded.
//! ORT telemetry is disabled; native provider network behavior is outside this
//! library's control and must be isolated by the host when required.
//! Windows UNC and device paths are rejected before filesystem probing; mapped
//! drives, reparse points, and native dependency loading require host isolation.
//! Model, input, and output sizes are capped, but ORT and its provider manage
//! additional allocations outside these caps. Use an externally resource-
//! limited process when a hard memory budget is required. This implementation
//! is not a `LiveProcessingProvider` and does not claim hot zero allocation.
//!
//! The model and its semantic interpretation are host-authorized inputs.
//! Requesting `NPU` and disabling ORT CPU fallback does not constitute hardware
//! attestation or validate a model's predictions.

#![cfg(not(target_arch = "wasm32"))]

use ort::{
    ep,
    session::Session,
    value::{TensorElementType, TensorRef, ValueType},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub max_model_bytes: usize,
    pub max_input_elements: usize,
    pub max_output_elements: usize,
}

#[derive(Debug)]
pub enum NpuError {
    InvalidLimits,
    ModelSize,
    ModelInterface,
    InputSize,
    OutputSize,
    Runtime(String),
    RuntimeAlreadyInitialized,
    InvalidRuntimePath,
    RuntimePanicked,
    Faulted,
    DeploymentAdmission,
}

/// Token for the process-global ORT environment. The library must be trusted;
/// loading a native library runs code in this process. ORT reuses its first
/// loaded library, so callers must invoke this before any other `ort` use.
pub struct NpuRuntime {
    environment: ort::environment::Environment,
    requested_library: std::path::PathBuf,
}

impl NpuRuntime {
    pub fn initialize(library: &std::path::Path) -> Result<Self, NpuError> {
        if !local_path_syntax(library) || !library.is_file() {
            return Err(NpuError::InvalidRuntimePath);
        }
        std::panic::catch_unwind(|| {
            let environment = ort::init_from(library)
                .map_err(runtime)?
                .with_telemetry(false)
                .build()
                .map_err(runtime)?;
            Ok(Self {
                environment,
                requested_library: library.to_owned(),
            })
        })
        .map_err(|_| NpuError::RuntimePanicked)?
    }

    /// Retained environment shared by explicit NPU and QNN sessions.
    pub fn environment(&self) -> &ort::environment::Environment {
        &self.environment
    }

    /// The requested path only. ORT's global loader may have reused a library;
    /// this is not an observation of the loaded binary's identity.
    pub fn requested_library(&self) -> &std::path::Path {
        &self.requested_library
    }

    /// Use a separately admitted artifact set. The host must keep its root and
    /// artifact lease protected through all runtime use and initialize before
    /// any other ORT API. No loaded-library or provider attestation is minted.
    pub fn initialize_from_deployment(
        deployment: &etalon_sense_deployment::AdmittedDeployment<'_, '_, '_>,
    ) -> Result<Self, NpuError> {
        deployment
            .revalidate_authority()
            .map_err(|_| NpuError::DeploymentAdmission)?;
        Self::initialize(deployment.runtime_path())
    }
}

#[cfg(windows)]
fn local_path_syntax(path: &std::path::Path) -> bool {
    use std::path::{Component, Prefix};
    path.is_absolute()
        && matches!(path.components().next(),
        Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_)))
}

#[cfg(not(windows))]
fn local_path_syntax(path: &std::path::Path) -> bool {
    path.is_absolute()
}

/// One fixed-shape, single-input, single-output f32 inference session.
/// `&mut self` serializes inference and session destruction.
pub struct OpenVinoNpuSession {
    session: Session,
    input_elements: usize,
    output_elements: usize,
    faulted: bool,
}

impl OpenVinoNpuSession {
    /// Check admission immediately before model preparation. Native runtime
    /// loading, driver identity and no-egress enforcement remain host duties.
    pub fn prepare_from_deployment(
        runtime_handle: &NpuRuntime,
        deployment: &etalon_sense_deployment::AdmittedDeployment<'_, '_, '_>,
        model: &[u8],
        limits: ExecutionLimits,
    ) -> Result<Self, NpuError> {
        if deployment.provider_kind() != etalon_sense_deployment::ProviderKind::OpenVinoNpu
            || runtime_handle.requested_library() != deployment.runtime_path()
        {
            return Err(NpuError::DeploymentAdmission);
        }
        deployment
            .revalidate_authority()
            .map_err(|_| NpuError::DeploymentAdmission)?;
        Self::prepare(runtime_handle, model, limits)
    }

    /// The caller supplies the initialized runtime and a trusted self-contained
    /// model. ORT may execute setup code while loading
    /// it; the byte limit alone does not sandbox its internal allocations.
    pub fn prepare(
        runtime_handle: &NpuRuntime,
        model: &[u8],
        limits: ExecutionLimits,
    ) -> Result<Self, NpuError> {
        if limits.max_model_bytes == 0
            || limits.max_input_elements == 0
            || limits.max_output_elements == 0
        {
            return Err(NpuError::InvalidLimits);
        }
        if model.is_empty() || model.len() > limits.max_model_bytes {
            return Err(NpuError::ModelSize);
        }
        let session = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut builder = Session::builder(runtime_handle.environment())
                .map_err(runtime)?
                .with_no_environment_execution_providers()
                .map_err(runtime)?
                .with_execution_providers([ep::OpenVINO::default()
                    .with_device_type("NPU")
                    .build()
                    .error_on_failure()])
                .map_err(runtime)?
                .with_disable_cpu_fallback()
                .map_err(runtime)?;
            builder.commit_from_memory(model).map_err(runtime)
        }))
        .map_err(|_| NpuError::RuntimePanicked)??;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return Err(NpuError::ModelInterface);
        }
        let input_elements =
            fixed_f32_vector(session.inputs()[0].dtype(), limits.max_input_elements)?;
        let output_elements =
            fixed_f32_vector(session.outputs()[0].dtype(), limits.max_output_elements)?;
        Ok(Self {
            session,
            input_elements,
            output_elements,
            faulted: false,
        })
    }

    pub fn input_elements(&self) -> usize {
        self.input_elements
    }
    pub fn output_elements(&self) -> usize {
        self.output_elements
    }

    /// Caller supplies exact-size vectors. Output is cleared on any error.
    pub fn run(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), NpuError> {
        output.fill(0.0);
        if self.faulted {
            return Err(NpuError::Faulted);
        }
        if input.len() != self.input_elements {
            return Err(NpuError::InputSize);
        }
        if output.len() != self.output_elements {
            return Err(NpuError::OutputSize);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let tensor =
                TensorRef::from_array_view(([self.input_elements], input)).map_err(runtime)?;
            let values = self.session.run(ort::inputs![tensor]).map_err(runtime)?;
            if values.len() != 1 {
                return Err(NpuError::ModelInterface);
            }
            let (shape, data) = values[0].try_extract_tensor::<f32>().map_err(runtime)?;
            if shape.len() != 1
                || shape[0] != self.output_elements as i64
                || data.len() != output.len()
            {
                return Err(NpuError::ModelInterface);
            }
            output.copy_from_slice(data);
            Ok(())
        }))
        .unwrap_or(Err(NpuError::RuntimePanicked));
        if result.is_err() {
            self.faulted = true;
            output.fill(0.0);
        }
        result
    }
}

fn fixed_f32_vector(value: &ValueType, limit: usize) -> Result<usize, NpuError> {
    match value {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } if shape.len() == 1 => {
            let count = usize::try_from(shape[0]).map_err(|_| NpuError::ModelInterface)?;
            if count == 0 || count > limit {
                return Err(NpuError::ModelInterface);
            }
            Ok(count)
        }
        _ => Err(NpuError::ModelInterface),
    }
}

fn runtime(error: impl core::fmt::Display) -> NpuError {
    NpuError::Runtime(error.to_string())
}

#[cfg(all(test, windows))]
mod path_tests {
    use super::local_path_syntax;
    use std::path::Path;

    #[test]
    fn rejects_unc_and_device_paths_before_probe() {
        assert!(!local_path_syntax(Path::new(
            r"\\server\share\onnxruntime.dll"
        )));
        assert!(!local_path_syntax(Path::new(
            r"\\?\UNC\server\share\onnxruntime.dll"
        )));
        assert!(!local_path_syntax(Path::new(r"\\.\device\onnxruntime.dll")));
        assert!(!local_path_syntax(Path::new(r"C:relative\onnxruntime.dll")));
        assert!(local_path_syntax(Path::new(r"C:\local\onnxruntime.dll")));
        assert!(local_path_syntax(Path::new(
            r"\\?\C:\local\onnxruntime.dll"
        )));
    }
}
