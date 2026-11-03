//! Explicit CUDA execution of Forge's FP32 tensor kernels with resident weights.

use std::{
    mem::{size_of, size_of_val},
    rc::Rc,
};

use etalon_cuda::{CudaBuffer, CudaModule};
pub use etalon_cuda::{CudaContext, CudaDeviceInfo, CudaError};
use etalon_wgsl_forge::{emit_shader_ptx, BuiltinKernel, Schedule};

mod resident;
pub use resident::CudaPreparedTensor;
use resident::PreparedOperation;
#[cfg(test)]
mod tests;

const BLOCK_SIZE: u32 = 256;

fn checked_elements(elements: usize) -> Result<u32, CudaError> {
    if elements == 0 {
        return Err(CudaError::InvalidArgument(
            "tensor dimensions must be nonzero",
        ));
    }
    elements
        .checked_mul(size_of::<f32>())
        .ok_or(CudaError::InvalidArgument("tensor byte length overflows"))?;
    u32::try_from(elements).map_err(|_| CudaError::InvalidArgument("tensor indexing exceeds u32"))
}

fn matrix_elements(rows: usize, columns: usize) -> Result<usize, CudaError> {
    let count = rows
        .checked_mul(columns)
        .ok_or(CudaError::InvalidArgument("tensor dimensions overflow"))?;
    checked_elements(count)?;
    Ok(count)
}

fn validate_matrix(length: usize, rows: usize, columns: usize) -> Result<(), CudaError> {
    if length != matrix_elements(rows, columns)? {
        return Err(CudaError::InvalidArgument(
            "tensor shape does not match its storage",
        ));
    }
    Ok(())
}

fn upload(context: &CudaContext, values: &[f32]) -> Result<CudaBuffer, CudaError> {
    checked_elements(values.len())?;
    let buffer = context.allocate(size_of_val(values))?;
    buffer.write(bytemuck::cast_slice(values))?;
    Ok(buffer)
}

/// A selected CUDA device with compiled Forge GEMV, GEMM, and fused-FFN kernels.
#[derive(Debug)]
pub struct CudaTensorExecutor {
    context: CudaContext,
    gemv: Rc<CudaModule>,
    gemm: Rc<CudaModule>,
    fused_ffn: Rc<CudaModule>,
}

impl CudaTensorExecutor {
    /// Compile the shared Forge kernels for this device using the local CUDA toolkit.
    pub fn new(context: CudaContext) -> Result<Self, CudaError> {
        let compile = |builtin: BuiltinKernel| {
            let shader = emit_shader_ptx(
                &builtin.spec(),
                Schedule::from_1d(BLOCK_SIZE, 1, 1),
                context.device_info().compute_capability,
            )
            .map_err(|error| CudaError::Compilation(error.to_string()))?;
            // Fixed Forge kernels are admitted only through the checked shape/ABI paths below.
            unsafe { context.load_ptx(&shader.source).map(Rc::new) }
        };
        Ok(Self {
            gemv: compile(BuiltinKernel::Gemv)?,
            gemm: compile(BuiltinKernel::Gemm)?,
            fused_ffn: compile(BuiltinKernel::FusedFfn)?,
            context,
        })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.context.device_info()
    }

    /// Execute a nonempty row-major matrix-vector product and return GPU readback.
    pub fn gemv(&self, matrix: &[f32], vector: &[f32], rows: usize) -> Result<Vec<f32>, CudaError> {
        let mut prepared = self.prepare_gemv(matrix, rows, vector.len())?;
        let mut output = vec![0.0; prepared.output_len()];
        prepared.execute(vector, &mut output)?;
        Ok(output)
    }

    /// Multiply row-major `left[rows, inner]` and `right[inner, columns]` on CUDA.
    pub fn gemm(
        &self,
        left: &[f32],
        right: &[f32],
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<Vec<f32>, CudaError> {
        validate_matrix(left.len(), rows, inner)?;
        let mut prepared = self.prepare_gemm(right, rows, inner, columns)?;
        let mut output = vec![0.0; prepared.output_len()];
        prepared.execute(left, &mut output)?;
        Ok(output)
    }

    /// Retain `matrix[rows, columns]` and reuse input/output allocations across GEMV calls.
    pub fn prepare_gemv(
        &self,
        matrix: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<CudaPreparedTensor, CudaError> {
        validate_matrix(matrix.len(), rows, columns)?;
        let dimensions = self.context.allocate(2 * size_of::<u32>())?;
        dimensions.write(bytemuck::cast_slice(&[rows as u32, columns as u32]))?;
        CudaPreparedTensor::new(
            self.context.clone(),
            Rc::clone(&self.gemv),
            PreparedOperation::Gemv { dimensions },
            vec![upload(&self.context, matrix)?],
            columns,
            rows,
        )
    }

    /// Retain `right[inner, columns]` for repeated fixed-shape left-matrix batches.
    pub fn prepare_gemm(
        &self,
        right: &[f32],
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<CudaPreparedTensor, CudaError> {
        validate_matrix(right.len(), inner, columns)?;
        let input_len = matrix_elements(rows, inner)?;
        let output_len = matrix_elements(rows, columns)?;
        let dimensions = self.context.allocate(3 * size_of::<u32>())?;
        dimensions.write(bytemuck::cast_slice(&[
            rows as u32,
            columns as u32,
            inner as u32,
        ]))?;
        CudaPreparedTensor::new(
            self.context.clone(),
            Rc::clone(&self.gemm),
            PreparedOperation::Gemm { dimensions },
            vec![upload(&self.context, right)?],
            input_len,
            output_len,
        )
    }

    /// Retain `w1[hidden, input]` and `w2[output, hidden]` for `w2 * GELU(w1 * x)`.
    /// Uses Forge's tanh GELU approximation, with no biases or residual addition.
    pub fn prepare_fused_ffn(
        &self,
        w1: &[f32],
        w2: &[f32],
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Result<CudaPreparedTensor, CudaError> {
        validate_matrix(w1.len(), hidden_size, input_size)?;
        validate_matrix(w2.len(), output_size, hidden_size)?;
        CudaPreparedTensor::new(
            self.context.clone(),
            Rc::clone(&self.fused_ffn),
            PreparedOperation::FusedFfn {
                parameters: [input_size as u32, hidden_size as u32, output_size as u32, 0],
            },
            vec![upload(&self.context, w1)?, upload(&self.context, w2)?],
            input_size,
            output_size,
        )
    }
}
