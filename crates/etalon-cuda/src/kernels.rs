//! Checked CUDA operations over host slices and privately owned device buffers.

use std::mem::size_of;

use crate::runtime::{CudaArgument, CudaContext, CudaError, CudaModule};
use etalon_wgsl_forge::{emit_shader_ptx, BuiltinKernel, Schedule};

const BLOCK_SIZE: u32 = 256;

fn element_count(len: usize) -> Result<u32, CudaError> {
    u32::try_from(len).map_err(|_| CudaError::InvalidArgument("element count exceeds u32"))
}

fn byte_length(len: usize) -> Result<usize, CudaError> {
    len.checked_mul(size_of::<f32>())
        .ok_or(CudaError::InvalidArgument("buffer byte length overflow"))
}

fn matvec_shape(matrix_len: usize, columns: usize, rows: usize) -> Result<(u32, u32), CudaError> {
    if columns == 0 {
        return Err(CudaError::InvalidArgument("matrix columns must be nonzero"));
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(CudaError::InvalidArgument("matrix dimensions overflow"))?;
    if matrix_len != expected {
        return Err(CudaError::InvalidArgument(
            "matrix dimensions do not match storage",
        ));
    }
    element_count(expected)?;
    byte_length(expected)?;
    Ok((element_count(rows)?, element_count(columns)?))
}

fn iq_count(len: usize) -> Result<u32, CudaError> {
    if !len.is_multiple_of(2) {
        return Err(CudaError::InvalidArgument(
            "IQ input must contain complete pairs",
        ));
    }
    byte_length(len)?;
    element_count(len)?;
    element_count(len / 2)
}

fn grid(count: u32) -> [u32; 3] {
    [count.div_ceil(BLOCK_SIZE), 1, 1]
}

/// Reusable built-in CUDA kernels. Each operation uploads, dispatches, and reads back.
#[derive(Debug)]
pub struct CudaKernels {
    affine: CudaModule,
    matvec: CudaModule,
    iq_power: CudaModule,
    context: CudaContext,
}

impl CudaKernels {
    /// Compiles Forge's built-ins for this device; requires the CUDA toolkit's `nvcc`.
    pub fn new(context: CudaContext) -> Result<Self, CudaError> {
        let compile = |builtin: BuiltinKernel| {
            let shader = emit_shader_ptx(
                &builtin.spec(),
                Schedule::from_1d(BLOCK_SIZE, 1, 1),
                context.device_info().compute_capability,
            )
            .map_err(|error| CudaError::Compilation(error.to_string()))?;
            // Only fixed Forge kernels with the checked launch contracts below are admitted.
            unsafe { context.load_ptx(&shader.source) }
        };
        Ok(Self {
            affine: compile(BuiltinKernel::AffineF32)?,
            matvec: compile(BuiltinKernel::Gemv)?,
            iq_power: compile(BuiltinKernel::IqPowerF32)?,
            context,
        })
    }

    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    /// Forge routing facts matched against the caller's selected device PCI bus ID.
    pub fn forge_capabilities(
        &self,
        selected_pci_bus_id: &str,
        portable_target: etalon_wgsl_forge::TargetBackend,
    ) -> etalon_wgsl_forge::RouteCapabilities {
        etalon_wgsl_forge::RouteCapabilities {
            cuda: etalon_wgsl_forge::CudaRouteSupport {
                driver_available: true,
                compiler_available: true,
                selected_device_matches: !selected_pci_bus_id.is_empty()
                    && self
                        .context
                        .device_info()
                        .pci_bus_id
                        .eq_ignore_ascii_case(selected_pci_bus_id),
                compute_capability: Some(self.context.device_info().compute_capability),
            },
            portable_target,
        }
    }

    /// Computes `input * scale + bias` with Forge's FP32 affine kernel.
    pub fn affine_f32(&self, input: &[f32], scale: f32, bias: f32) -> Result<Vec<f32>, CudaError> {
        let count = element_count(input.len())?;
        let bytes = byte_length(input.len())?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let source = self.context.allocate(bytes)?;
        let destination = self.context.allocate(bytes)?;
        source.write(bytemuck::cast_slice(input))?;
        let kernel = self.affine.kernel("affine_f32")?;
        let parameters = [count, scale.to_bits(), bias.to_bits(), 0];
        // Argument types match the fixed kernel; both allocations cover every checked index.
        unsafe {
            kernel.launch(
                grid(count),
                [BLOCK_SIZE, 1, 1],
                0,
                &[
                    CudaArgument::Buffer(&source),
                    CudaArgument::Buffer(&destination),
                    CudaArgument::Bytes(bytemuck::cast_slice(&parameters)),
                ],
            )?;
        }
        let mut output = vec![0.0; input.len()];
        destination.read(bytemuck::cast_slice_mut(&mut output))?;
        Ok(output)
    }

    /// Multiplies a row-major FP32 matrix by a vector, accumulating each row in order.
    pub fn matvec_f32(
        &self,
        matrix: &[f32],
        vector: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let (row_count, columns) = matvec_shape(matrix.len(), vector.len(), rows)?;
        if row_count == 0 {
            return Ok(Vec::new());
        }
        let matrix_buffer = self.context.allocate(byte_length(matrix.len())?)?;
        let vector_buffer = self.context.allocate(byte_length(vector.len())?)?;
        let destination = self.context.allocate(byte_length(rows)?)?;
        let dimensions = self.context.allocate(2 * size_of::<u32>())?;
        matrix_buffer.write(bytemuck::cast_slice(matrix))?;
        vector_buffer.write(bytemuck::cast_slice(vector))?;
        dimensions.write(bytemuck::cast_slice(&[row_count, columns]))?;
        let kernel = self.matvec.kernel("gemv_f32")?;
        // Exact u32-bounded dimensions establish Forge's row/column bounds and argument ABI.
        unsafe {
            kernel.launch(
                grid(row_count),
                [BLOCK_SIZE, 1, 1],
                0,
                &[
                    CudaArgument::Buffer(&matrix_buffer),
                    CudaArgument::Buffer(&vector_buffer),
                    CudaArgument::Buffer(&destination),
                    CudaArgument::Buffer(&dimensions),
                ],
            )?;
        }
        let mut output = vec![0.0; rows];
        destination.read(bytemuck::cast_slice_mut(&mut output))?;
        Ok(output)
    }

    /// Computes `I * I + Q * Q` for interleaved FP32 IQ pairs.
    pub fn iq_power_f32(&self, interleaved_iq: &[f32]) -> Result<Vec<f32>, CudaError> {
        let count = iq_count(interleaved_iq.len())?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let source = self.context.allocate(byte_length(interleaved_iq.len())?)?;
        let destination = self.context.allocate(byte_length(count as usize)?)?;
        source.write(bytemuck::cast_slice(interleaved_iq))?;
        let kernel = self.iq_power.kernel("iq_power_f32")?;
        let parameters = [count, 0, 0, 0];
        // Complete pairs and exact output capacity cover every access by the fixed kernel.
        unsafe {
            kernel.launch(
                grid(count),
                [BLOCK_SIZE, 1, 1],
                0,
                &[
                    CudaArgument::Buffer(&source),
                    CudaArgument::Buffer(&destination),
                    CudaArgument::Bytes(bytemuck::cast_slice(&parameters)),
                ],
            )?;
        }
        let mut output = vec![0.0; count as usize];
        destination.read(bytemuck::cast_slice_mut(&mut output))?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_dimensions_reject_mismatch_and_overflow() {
        assert!(matvec_shape(5, 2, 3).is_err());
        assert!(matvec_shape(0, 0, 0).is_err());
        assert!(matvec_shape(0, 2, usize::MAX).is_err());
        assert!(byte_length(usize::MAX).is_err());
        assert!(iq_count(3).is_err());
        assert_eq!(matvec_shape(6, 2, 3).unwrap(), (3, 2));
        assert_eq!(matvec_shape(0, 3, 0).unwrap(), (0, 3));
        assert_eq!(iq_count(0).unwrap(), 0);
        assert_eq!(iq_count(1026).unwrap(), 513);
        assert_eq!(grid(513), [3, 1, 1]);
        #[cfg(target_pointer_width = "64")]
        {
            let beyond_u32 = u32::MAX as usize + 1;
            assert!(element_count(beyond_u32).is_err());
            assert!(matvec_shape(beyond_u32, beyond_u32 / 2, 2).is_err());
            assert!(iq_count(beyond_u32).is_err());
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-5 + expected.abs() * 1e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    #[ignore = "requires nvcc and a live NVIDIA CUDA device; select ordinal with ETALON_CUDA_DEVICE"]
    fn cuda_builtins_match_cpu_references() {
        let ordinal = std::env::var("ETALON_CUDA_DEVICE")
            .map(|value| {
                value
                    .parse::<i32>()
                    .expect("CUDA ordinal must be an integer")
            })
            .unwrap_or(0);
        let context = CudaContext::new(ordinal).unwrap();
        eprintln!(
            "CUDA built-in qualification adapter: {:?}",
            context.device_info()
        );
        let kernels = CudaKernels::new(context).unwrap();

        let input: Vec<f32> = (0..513).map(|i| (i as f32 - 251.0) / 32.0).collect();
        let expected: Vec<f32> = input.iter().map(|&value| value * 1.25 - 0.75).collect();
        assert_close(&kernels.affine_f32(&input, 1.25, -0.75).unwrap(), &expected);
        assert_close(&kernels.affine_f32(&[2.0], 3.0, -1.0).unwrap(), &[5.0]);

        let rows = 259;
        let columns = 17;
        let matrix: Vec<f32> = (0..rows * columns)
            .map(|i| ((i % 29) as f32 - 14.0) / 16.0)
            .collect();
        let vector: Vec<f32> = (0..columns).map(|i| (i as f32 - 8.0) / 8.0).collect();
        let expected: Vec<f32> = matrix
            .chunks_exact(columns)
            .map(|row| row.iter().zip(&vector).map(|(&a, &b)| a * b).sum())
            .collect();
        assert_close(
            &kernels.matvec_f32(&matrix, &vector, rows).unwrap(),
            &expected,
        );

        let iq: Vec<f32> = (0..1026).map(|i| ((i % 37) as f32 - 18.0) / 8.0).collect();
        let expected: Vec<f32> = iq
            .chunks_exact(2)
            .map(|pair| pair[0] * pair[0] + pair[1] * pair[1])
            .collect();
        assert_close(&kernels.iq_power_f32(&iq).unwrap(), &expected);
        assert!(kernels.affine_f32(&[], 1.0, 0.0).unwrap().is_empty());
        assert!(kernels.iq_power_f32(&[]).unwrap().is_empty());
        assert!(kernels.matvec_f32(&[], &[1.0], 0).unwrap().is_empty());
        assert!(kernels.matvec_f32(&matrix, &vector, rows - 1).is_err());
        assert!(kernels.iq_power_f32(&iq[..1025]).is_err());
    }
}
