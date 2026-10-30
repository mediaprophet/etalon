//! Naga module validation, native compiler validation, and CPU oracle drift checking.

use serde::{Deserialize, Serialize};

use crate::emit::TargetBackend;
use crate::ir::{ForgeError, KernelSpec};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub source_hash: String,
    pub entry_points: Vec<String>,
    pub binding_count: usize,
    pub naga_validated: bool,
    pub native_tool_validated: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OracleTolerance {
    pub absolute: f32,
    pub relative: f32,
}

impl Default for OracleTolerance {
    fn default() -> Self {
        Self {
            absolute: 1.0e-6,
            relative: 1.0e-5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub compared: usize,
    pub mismatch_count: usize,
    pub first_mismatch: Option<(usize, f32, f32)>,
    pub max_absolute_error: f32,
    pub max_relative_error: f32,
}

impl ComparisonReport {
    pub const fn passed(&self) -> bool {
        self.mismatch_count == 0
    }
}

/// Validates a WGSL module via Naga's front-end parser and IR validator.
pub fn validate_wgsl(source: &str) -> Result<ValidationReport, ForgeError> {
    #[cfg(feature = "naga-validate")]
    {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|error| ForgeError::WgslParse(error.emit_to_string(source)))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .map_err(|error| ForgeError::WgslValidation(format!("{error:?}")))?;

        let mut entry_points = module
            .entry_points
            .iter()
            .map(|entry| entry.name.clone())
            .collect::<Vec<_>>();
        entry_points.sort();
        let binding_count = module
            .global_variables
            .iter()
            .filter(|(_, variable)| variable.binding.is_some())
            .count();
        Ok(ValidationReport {
            source_hash: blake3::hash(source.as_bytes()).to_hex().to_string(),
            entry_points,
            binding_count,
            naga_validated: true,
            native_tool_validated: None,
        })
    }
    #[cfg(not(feature = "naga-validate"))]
    {
        Ok(ValidationReport {
            source_hash: blake3::hash(source.as_bytes()).to_hex().to_string(),
            entry_points: Vec::new(),
            binding_count: 0,
            naga_validated: false,
            native_tool_validated: None,
        })
    }
}

/// Certifies a WGSL kernel source string via Naga validation.
pub fn certify_kernel(source: &str) -> Result<ValidationReport, ForgeError> {
    validate_wgsl(source)
}

/// Validates a non-WGSL shader by invoking the offline compiler (`ptxas`, `dxc`, or `xcrun`).
pub fn validate_native(
    source: &str,
    target: TargetBackend,
    kernel: Option<&KernelSpec>,
) -> Result<ValidationReport, ForgeError> {
    use std::process::Command;

    let tool_name = match target {
        TargetBackend::Ptx => "ptxas",
        TargetBackend::Hlsl => "dxc",
        TargetBackend::Msl => "xcrun",
        _ => {
            return Err(ForgeError::WgslValidation(
                "Not a native target supported by offline validation".to_string(),
            ))
        }
    };

    let temp_file = tempfile::Builder::new()
        .prefix("etalon-validate-")
        .suffix(match target {
            TargetBackend::Ptx => ".ptx",
            TargetBackend::Hlsl => ".hlsl",
            TargetBackend::Msl => ".metal",
            _ => "",
        })
        .tempfile()
        .map_err(|e| ForgeError::Io(format!("Failed to create temp file: {e}")))?;

    let temp_path = temp_file.path().to_path_buf();
    std::fs::write(&temp_path, source)
        .map_err(|e| ForgeError::Io(format!("Failed to write to temp file: {e}")))?;

    let mut cmd = Command::new(tool_name);
    match target {
        TargetBackend::Ptx => {
            if Command::new("ptxas").arg("--version").output().is_err() {
                return Err(ForgeError::WgslValidation(
                    "ptxas not found in PATH".to_string(),
                ));
            }
            cmd.arg(&temp_path).arg("-c");
        }
        TargetBackend::Hlsl => {
            if Command::new("dxc").arg("--help").output().is_err() {
                return Err(ForgeError::WgslValidation(
                    "dxc not found in PATH".to_string(),
                ));
            }
            cmd.arg("-T").arg("cs_6_0").arg(&temp_path);
        }
        TargetBackend::Msl => {
            if Command::new("xcrun").arg("--version").output().is_err() {
                return Err(ForgeError::WgslValidation(
                    "xcrun not found in PATH".to_string(),
                ));
            }
            cmd.arg("-sdk")
                .arg("macosx")
                .arg("metal")
                .arg("-c")
                .arg(&temp_path);
        }
        _ => {}
    }

    let output = cmd
        .output()
        .map_err(|e| ForgeError::Io(format!("Failed to execute validation tool: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ForgeError::WgslValidation(format!(
            "{tool_name} validation failed: {stderr}"
        )));
    }

    let (entry_points, binding_count) = match kernel {
        Some(k) => (vec![k.entry_point.clone()], k.buffers.len()),
        None => (Vec::new(), 0),
    };
    Ok(ValidationReport {
        source_hash: blake3::hash(source.as_bytes()).to_hex().to_string(),
        entry_points,
        binding_count,
        naga_validated: false,
        native_tool_validated: Some(tool_name.to_string()),
    })
}

/// Compares actual GPU f32 outputs against a CPU reference within `tolerance`.
pub fn compare_f32(
    expected: &[f32],
    actual: &[f32],
    tolerance: OracleTolerance,
) -> Result<ComparisonReport, ForgeError> {
    if expected.len() != actual.len() {
        return Err(ForgeError::OracleMismatch(format!(
            "output length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        )));
    }
    let mut mismatch_count = 0usize;
    let mut first_mismatch = None;
    let mut max_absolute_error = 0.0f32;
    let mut max_relative_error = 0.0f32;

    for (idx, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        let abs_err = (exp - act).abs();
        let rel_err = if exp.abs() > 1.0e-12 {
            abs_err / exp.abs()
        } else {
            0.0
        };
        if abs_err > max_absolute_error {
            max_absolute_error = abs_err;
        }
        if rel_err > max_relative_error {
            max_relative_error = rel_err;
        }
        let limit = tolerance.absolute + tolerance.relative * exp.abs();
        if !act.is_finite() || abs_err > limit {
            mismatch_count += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((idx, exp, act));
            }
        }
    }

    Ok(ComparisonReport {
        compared: expected.len(),
        mismatch_count,
        first_mismatch,
        max_absolute_error,
        max_relative_error,
    })
}

/// Reference CPU row-major dense GEMM `C[M x N] = A[M x K] * B[K x N]`.
pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

/// Reference CPU row-major dense GEMV `y[M] = A[M x N] * x[N]`.
pub fn gemv_cpu(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    for i in 0..m {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += a[i * n + j] * x[j];
        }
        y[i] = acc;
    }
    y
}
