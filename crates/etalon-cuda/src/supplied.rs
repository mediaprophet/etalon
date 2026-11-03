//! Explicit native compilation and reusable, ABI-checked supplied kernel dispatch.

use crate::{CudaArgument, CudaContext, CudaError, CudaKernel, CudaLaunchWorkspace, CudaModule};

/// The host-side ABI and minimum allocation size of one supplied kernel parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaParameter {
    Buffer { minimum_bytes: usize },
    U32,
    F32,
    Bytes { size: usize },
}

/// A compiled native program retained for repeated kernel lookup and dispatch.
#[derive(Debug)]
pub struct CudaProgram {
    module: CudaModule,
}

impl CudaProgram {
    /// Compile trusted CUDA-C through Forge for this context's device architecture.
    ///
    /// # Safety
    /// The source is native code. The caller must trust it and uphold the module
    /// and launch contracts of [`CudaContext::load_ptx`] and [`CudaKernel::launch`].
    pub unsafe fn compile(context: &CudaContext, source: &str) -> Result<Self, CudaError> {
        let ptx = etalon_wgsl_forge::compile_cuda_to_ptx(
            source,
            context.device_info().compute_capability,
        )
        .map_err(|error| CudaError::Compilation(error.to_string()))?;
        Ok(Self {
            module: unsafe { context.load_ptx(&ptx) }?,
        })
    }

    /// Load a previously compiled PTX artifact without invoking the toolkit.
    ///
    /// # Safety
    /// The contract of [`CudaContext::load_ptx`] applies.
    pub unsafe fn from_ptx(context: &CudaContext, ptx: &str) -> Result<Self, CudaError> {
        Ok(Self {
            module: unsafe { context.load_ptx(ptx) }?,
        })
    }

    /// Prepare reusable argument storage for a caller-declared ABI.
    /// The declaration is checked at dispatch, but is not inferred from native code.
    pub fn prepare(
        &self,
        name: &str,
        parameters: &[CudaParameter],
    ) -> Result<CudaPreparedKernel<'_>, CudaError> {
        for parameter in parameters {
            if matches!(
                parameter,
                CudaParameter::Buffer { minimum_bytes: 0 } | CudaParameter::Bytes { size: 0 }
            ) {
                return Err(CudaError::InvalidArgument(
                    "CUDA parameter sizes must be nonzero",
                ));
            }
        }
        Ok(CudaPreparedKernel {
            kernel: self.module.kernel(name)?,
            parameters: parameters.to_vec(),
            workspace: CudaLaunchWorkspace::new(parameters.len()),
        })
    }
}

/// Reusable dispatch state borrowing its program to keep the module alive.
#[derive(Debug)]
pub struct CudaPreparedKernel<'a> {
    kernel: CudaKernel<'a>,
    parameters: Vec<CudaParameter>,
    workspace: CudaLaunchWorkspace,
}

fn validate_arguments(
    parameters: &[CudaParameter],
    args: &[CudaArgument<'_>],
) -> Result<(), CudaError> {
    if parameters.len() != args.len() {
        return Err(CudaError::InvalidArgument(
            "CUDA argument count does not match declared ABI",
        ));
    }
    for (parameter, argument) in parameters.iter().zip(args) {
        let valid = match (parameter, argument) {
            (CudaParameter::Buffer { minimum_bytes }, CudaArgument::Buffer(buffer)) => {
                buffer.len() >= *minimum_bytes
            }
            (CudaParameter::U32, CudaArgument::U32(_))
            | (CudaParameter::F32, CudaArgument::F32(_)) => true,
            (CudaParameter::Bytes { size }, CudaArgument::Bytes(bytes)) => bytes.len() == *size,
            _ => false,
        };
        if !valid {
            return Err(CudaError::InvalidArgument(
                "CUDA argument does not match declared ABI",
            ));
        }
    }
    Ok(())
}

impl CudaPreparedKernel<'_> {
    /// Validate declared argument types and capacities, dispatch, and wait for completion.
    ///
    /// # Safety
    /// The declaration must match the actual kernel ABI, and every contract of
    /// [`CudaKernel::launch`] must hold. Minimum buffer sizes alone do not validate
    /// accesses determined by the arguments or by the supplied native code.
    pub unsafe fn launch(
        &mut self,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory: u32,
        args: &[CudaArgument<'_>],
    ) -> Result<(), CudaError> {
        validate_arguments(&self.parameters, args)?;
        unsafe {
            self.kernel
                .launch_with_workspace(grid, block, shared_memory, args, &mut self.workspace)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supplied_arguments_require_exact_types_counts_and_value_sizes() {
        let layout = [
            CudaParameter::U32,
            CudaParameter::F32,
            CudaParameter::Bytes { size: 16 },
        ];
        let bytes = [0u8; 16];
        let valid = [
            CudaArgument::U32(17),
            CudaArgument::F32(2.0),
            CudaArgument::Bytes(&bytes),
        ];
        assert!(validate_arguments(&layout, &valid).is_ok());
        assert!(validate_arguments(&layout, &valid[..2]).is_err());
        assert!(validate_arguments(&layout[..2], &valid).is_err());
        assert!(validate_arguments(&[CudaParameter::U32], &[CudaArgument::F32(0.0)]).is_err());
        assert!(validate_arguments(
            &[CudaParameter::Bytes { size: 16 }],
            &[CudaArgument::Bytes(&bytes[..15])]
        )
        .is_err());
    }

    #[test]
    #[ignore = "requires nvcc and a live NVIDIA CUDA device"]
    fn supplied_forge_program_reuses_dispatch_and_checks_abi() {
        use etalon_wgsl_forge::{emit_cuda_c, BuiltinKernel, Schedule};
        let ordinal = std::env::var("ETALON_CUDA_DEVICE")
            .map(|value| value.parse().expect("CUDA ordinal must be an integer"))
            .unwrap_or(0);
        let context = CudaContext::new(ordinal).unwrap();
        eprintln!("Supplied CUDA qualification: {:?}", context.device_info());
        let shader = emit_cuda_c(
            &BuiltinKernel::AffineF32.spec(),
            Schedule::from_1d(64, 1, 1),
        )
        .unwrap();
        let program = unsafe { CudaProgram::compile(&context, &shader.source) }.unwrap();
        let count = 131usize;
        let input = context.allocate(count * 4).unwrap();
        let output = context.allocate(count * 4).unwrap();
        let short = context.allocate(4).unwrap();
        let layout = [
            CudaParameter::Buffer {
                minimum_bytes: count * 4,
            },
            CudaParameter::Buffer {
                minimum_bytes: count * 4,
            },
            CudaParameter::Bytes { size: 16 },
        ];
        let mut prepared = program.prepare("affine_f32", &layout).unwrap();
        let params = [count as u32, 1.5f32.to_bits(), (-0.25f32).to_bits(), 0];
        let arguments = |buffer| {
            [
                CudaArgument::Buffer(buffer),
                CudaArgument::Buffer(&output),
                CudaArgument::Bytes(bytemuck::cast_slice(&params)),
            ]
        };
        let error =
            unsafe { prepared.launch([3, 1, 1], [64, 1, 1], 0, &arguments(&short)) }.unwrap_err();
        assert!(matches!(error, CudaError::InvalidArgument(_)));
        for iteration in 0..8 {
            let values: Vec<f32> = (0..count)
                .map(|i| (i as f32 - 50.0) / (iteration + 1) as f32)
                .collect();
            input.write(bytemuck::cast_slice(&values)).unwrap();
            unsafe { prepared.launch([3, 1, 1], [64, 1, 1], 0, &arguments(&input)) }.unwrap();
            let mut actual = vec![0f32; count];
            output.read(bytemuck::cast_slice_mut(&mut actual)).unwrap();
            for (actual, value) in actual.iter().zip(values) {
                assert!((actual - (value * 1.5 - 0.25)).abs() < 1e-5);
            }
        }
    }
}
