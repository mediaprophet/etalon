use std::{path::PathBuf, process::Command};

use crate::{emit::emit_cuda_c, ForgeError, GeneratedShader, KernelSpec, Schedule};

fn architecture(source: &str, compute_capability: (u32, u32)) -> Result<String, ForgeError> {
    if source.trim().is_empty() || source.contains('\0') {
        return Err(ForgeError::Emission(
            "CUDA source must be nonempty and contain no NUL bytes".into(),
        ));
    }
    let (major, minor) = compute_capability;
    if !(1..=99).contains(&major) || minor > 9 {
        return Err(ForgeError::Emission(format!(
            "Invalid CUDA compute capability {major}.{minor}"
        )));
    }
    Ok(format!("compute_{major}{minor}"))
}

fn compiler_path() -> PathBuf {
    let executable = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    if let Some(root) = std::env::var_os("CUDA_PATH").filter(|root| !root.is_empty()) {
        let path = PathBuf::from(root).join("bin").join(executable);
        if path.is_file() {
            // NVCC cannot locate its toolkit headers through Windows verbatim paths.
            if let Ok(path) = std::path::absolute(path) {
                return path;
            }
        }
    }
    PathBuf::from(executable)
}

/// Compiles trusted CUDA-C source to PTX for an explicit compute capability.
/// Requires a local CUDA Toolkit; resolves NVCC through `CUDA_PATH/bin`, then `PATH`.
pub fn compile_cuda_to_ptx(
    source: &str,
    compute_capability: (u32, u32),
) -> Result<String, ForgeError> {
    let architecture = architecture(source, compute_capability)?;
    let directory = tempfile::Builder::new()
        .prefix("etalon-nvcc-")
        .tempdir()
        .map_err(|error| ForgeError::Emission(format!("Create NVCC workspace: {error}")))?;
    let input = directory.path().join("kernel.cu");
    let output = directory.path().join("kernel.ptx");
    std::fs::write(&input, source)
        .map_err(|error| ForgeError::Emission(format!("Write CUDA source: {error}")))?;
    let compiler = compiler_path();
    let result = Command::new(&compiler)
        .arg("--ptx")
        .arg("--std=c++17")
        .arg(format!("--gpu-architecture={architecture}"))
        .arg("--output-file")
        .arg(&output)
        .arg(&input)
        .current_dir(directory.path())
        .output()
        .map_err(|error| {
            ForgeError::Emission(format!(
                "Execute NVCC at {}: {error}; install the CUDA Toolkit and set CUDA_PATH or PATH",
                compiler.display()
            ))
        })?;
    if !result.status.success() {
        return Err(ForgeError::Emission(format!(
            "NVCC compilation failed for {architecture} ({}):\nstdout:\n{}\nstderr:\n{}",
            result.status,
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr),
        )));
    }
    let ptx = std::fs::read_to_string(&output)
        .map_err(|error| ForgeError::Emission(format!("Read NVCC PTX output: {error}")))?;
    if ![".version", ".target", ".address_size"]
        .iter()
        .all(|directive| {
            ptx.lines()
                .any(|line| line.trim_start().starts_with(*directive))
        })
    {
        return Err(ForgeError::Emission(
            "NVCC output is missing required PTX directives".into(),
        ));
    }
    Ok(ptx)
}

/// Emits CUDA-C and compiles it to PTX for the selected native CUDA device.
pub fn emit_shader_ptx(
    kernel: &KernelSpec,
    schedule: Schedule,
    compute_capability: (u32, u32),
) -> Result<GeneratedShader, ForgeError> {
    let mut generated = emit_cuda_c(kernel, schedule)?;
    generated.source = compile_cuda_to_ptx(&generated.source, compute_capability)?;
    generated.source_hash = blake3::hash(generated.source.as_bytes())
        .to_hex()
        .to_string();
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinKernel;

    #[test]
    fn invalid_source_and_architecture_fail_before_compiler_access() {
        for source in ["", " \r\n\t", "kernel\0truncated"] {
            let error = compile_cuda_to_ptx(source, (8, 6)).unwrap_err();
            assert!(error.to_string().contains("CUDA source"));
        }
        for capability in [(0, 0), (8, 10), (100, 0), (u32::MAX, u32::MAX)] {
            let error = compile_cuda_to_ptx("source", capability).unwrap_err();
            assert!(error.to_string().contains("compute capability"));
        }
        assert_eq!(architecture("source", (8, 6)).unwrap(), "compute_86");
        assert_eq!(architecture("source", (12, 0)).unwrap(), "compute_120");
    }

    #[test]
    #[ignore = "requires a local CUDA Toolkit with compute_86 support"]
    fn nvcc_compiles_affine_and_reports_invalid_syntax() {
        let shader = emit_shader_ptx(
            &BuiltinKernel::AffineF32.spec(),
            Schedule::default(),
            (8, 6),
        )
        .unwrap();
        assert!(shader.source.contains(".entry affine_f32"));
        assert!(shader.source.contains(".target sm_86"));
        assert_eq!(
            shader.source_hash,
            blake3::hash(shader.source.as_bytes()).to_hex().to_string()
        );
        let error = compile_cuda_to_ptx(
            "extern \"C\" __global__ void invalid_kernel( { syntax_error; }",
            (8, 6),
        )
        .unwrap_err();
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("NVCC compilation failed"));
        assert!(diagnostic.contains("kernel.cu"));
        assert!(diagnostic.contains("error"));
    }
}
