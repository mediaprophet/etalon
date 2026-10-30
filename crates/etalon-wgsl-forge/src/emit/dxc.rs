//! HLSL-to-SPIR-V compilation via DirectXShaderCompiler (DXC) with RAII temp cleanup.
//!
//! Includes Runbook 01 Bug 1.4 fix: uses a thread-unique `tempfile::Builder::new().prefix("etalon-dxc-").tempdir()`
//! with automatic RAII cleanup on drop or panic (Rule 0-B).

use crate::ir::ForgeError;
#[cfg(feature = "dxc")]
use std::process::Command;

/// Compiles HLSL source to SPIR-V bytecodes using the DirectXShaderCompiler (DXC).
#[cfg(feature = "dxc")]
pub fn compile_hlsl_to_spirv(hlsl_source: &str, entry_point: &str) -> Result<Vec<u8>, ForgeError> {
    // Thread-unique temporary directory with RAII cleanup (Rule 0-B).
    let temp_dir = tempfile::Builder::new()
        .prefix("etalon-dxc-")
        .tempdir()
        .map_err(|e| {
            ForgeError::Emission(format!(
                "Failed to create temporary directory for DXC: {e:?}"
            ))
        })?;
    let in_path = temp_dir.path().join("shader.hlsl");
    let out_path = temp_dir.path().join("shader.spv");

    std::fs::write(&in_path, hlsl_source)
        .map_err(|e| ForgeError::Emission(format!("Failed to write HLSL temp file: {e:?}")))?;

    let in_path_str = in_path
        .to_str()
        .ok_or_else(|| ForgeError::Emission("Non-UTF8 temporary input path for DXC".to_string()))?;
    let out_path_str = out_path.to_str().ok_or_else(|| {
        ForgeError::Emission("Non-UTF8 temporary output path for DXC".to_string())
    })?;

    let dxc_path = resolve_dxc_cli_path();

    let output = Command::new(&dxc_path)
        .arg("-spirv")
        .arg("-fspv-target-env=vulkan1.1")
        .arg("-fvk-use-dx-layout")
        .arg("-O3")
        .arg("-T")
        .arg("cs_6_0")
        .arg("-E")
        .arg(entry_point)
        .arg(in_path_str)
        .arg("-Fo")
        .arg(out_path_str)
        .output()
        .map_err(|e| {
            ForgeError::Emission(format!("Failed to execute DXC CLI at {dxc_path}: {e:?}"))
        })?;

    if !output.status.success() {
        let err_str = String::from_utf8_lossy(&output.stderr);
        let out_str = String::from_utf8_lossy(&output.stdout);
        return Err(ForgeError::Emission(format!(
            "DXC Compilation Failed:\nStdout: {out_str}\nStderr: {err_str}"
        )));
    }

    let spirv_blob = std::fs::read(&out_path)
        .map_err(|e| ForgeError::Emission(format!("Failed to read SPIRV output: {e:?}")))?;
    // TempDir auto-cleans up in_path and out_path upon drop.
    Ok(spirv_blob)
}

#[cfg(feature = "dxc")]
fn resolve_dxc_cli_path() -> String {
    for env_key in ["ETALON_DXC_CLI_PATH", "QUALIA_DXC_CLI_PATH"] {
        if let Ok(p) = std::env::var(env_key) {
            if !p.trim().is_empty() {
                return p;
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let dxc_exe = dir.join("dxc.exe");
            if dxc_exe.exists() {
                return dxc_exe.to_string_lossy().into_owned();
            }
        }
    }
    "dxc".to_string()
}

/// Fallback when DXC feature is disabled.
#[cfg(not(feature = "dxc"))]
pub fn compile_hlsl_to_spirv(
    _hlsl_source: &str,
    _entry_point: &str,
) -> Result<Vec<u8>, ForgeError> {
    Err(ForgeError::Emission(
        "HLSL compilation requires the 'dxc' feature".to_string(),
    ))
}
