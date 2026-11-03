use super::{checked_elements, CudaContext, CudaDeviceInfo, CudaError, BLOCK_SIZE};
use etalon_cuda::{CudaArgument, CudaBuffer, CudaLaunchWorkspace, CudaModule};
use std::{mem::size_of, rc::Rc};

#[derive(Debug)]
pub(super) enum PreparedOperation {
    Gemv { dimensions: CudaBuffer },
    Gemm { dimensions: CudaBuffer },
    FusedFfn { parameters: [u32; 4] },
}

/// A fixed-shape CUDA operation retaining its weights and device allocations.
/// Execution uploads only its input and reads back into caller-provided output storage.
#[derive(Debug)]
pub struct CudaPreparedTensor {
    context: CudaContext,
    module: Rc<CudaModule>,
    operation: PreparedOperation,
    weights: Vec<CudaBuffer>,
    input: CudaBuffer,
    output: CudaBuffer,
    input_len: usize,
    output_len: usize,
    groups: u32,
    workspace: CudaLaunchWorkspace,
}

impl CudaPreparedTensor {
    pub(super) fn new(
        context: CudaContext,
        module: Rc<CudaModule>,
        operation: PreparedOperation,
        weights: Vec<CudaBuffer>,
        input_len: usize,
        output_len: usize,
    ) -> Result<Self, CudaError> {
        checked_elements(input_len)?;
        let outputs = checked_elements(output_len)?;
        Ok(Self {
            module,
            operation,
            weights,
            input: context.allocate(input_len * size_of::<f32>())?,
            output: context.allocate(output_len * size_of::<f32>())?,
            context,
            input_len,
            output_len,
            groups: outputs.div_ceil(BLOCK_SIZE),
            workspace: CudaLaunchWorkspace::new(5),
        })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.context.device_info()
    }
    pub fn input_len(&self) -> usize {
        self.input_len
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    /// Execute with exact input/output lengths. On error, caller output is cleared.
    pub fn execute(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), CudaError> {
        output.fill(0.0);
        if input.len() != self.input_len || output.len() != self.output_len {
            return Err(CudaError::InvalidArgument(
                "prepared tensor input/output length mismatch",
            ));
        }
        let result = (|| {
            self.input.write(bytemuck::cast_slice(input))?;
            match &self.operation {
                PreparedOperation::Gemv { dimensions } => {
                    let kernel = self.module.kernel_cstr(c"gemv_f32")?;
                    // Exact matrix/vector lengths cover Forge's u32 GEMV indexing and argument ABI.
                    unsafe {
                        kernel.launch_with_workspace(
                            [self.groups, 1, 1],
                            [BLOCK_SIZE, 1, 1],
                            0,
                            &[
                                CudaArgument::Buffer(&self.weights[0]),
                                CudaArgument::Buffer(&self.input),
                                CudaArgument::Buffer(&self.output),
                                CudaArgument::Buffer(dimensions),
                            ],
                            &mut self.workspace,
                        )?;
                    }
                }
                PreparedOperation::Gemm { dimensions } => {
                    let kernel = self.module.kernel_cstr(c"gemm_f32")?;
                    // All matrix products fit u32; buffers exactly match this fixed ABI.
                    unsafe {
                        kernel.launch_with_workspace(
                            [self.groups, 1, 1],
                            [BLOCK_SIZE, 1, 1],
                            0,
                            &[
                                CudaArgument::Buffer(&self.input),
                                CudaArgument::Buffer(&self.weights[0]),
                                CudaArgument::Buffer(&self.output),
                                CudaArgument::Buffer(dimensions),
                            ],
                            &mut self.workspace,
                        )?;
                    }
                }
                PreparedOperation::FusedFfn { parameters } => {
                    let kernel = self.module.kernel_cstr(c"fused_ffn")?;
                    // Weight products fit u32 and parameters have Forge's 16-byte FfnParams ABI.
                    unsafe {
                        kernel.launch_with_workspace(
                            [self.groups, 1, 1],
                            [BLOCK_SIZE, 1, 1],
                            0,
                            &[
                                CudaArgument::Buffer(&self.input),
                                CudaArgument::Buffer(&self.weights[0]),
                                CudaArgument::Buffer(&self.weights[1]),
                                CudaArgument::Buffer(&self.output),
                                CudaArgument::Bytes(bytemuck::cast_slice(parameters)),
                            ],
                            &mut self.workspace,
                        )?;
                    }
                }
            }
            self.output.read(bytemuck::cast_slice_mut(output))
        })();
        if result.is_err() {
            output.fill(0.0);
        }
        result
    }
}
