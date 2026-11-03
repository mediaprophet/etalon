//! Reusable signed-i16 IQ power using the shared Forge kernel.

use crate::{
    CudaArgument, CudaBuffer, CudaContext, CudaDeviceInfo, CudaError, CudaLaunchWorkspace,
    CudaModule,
};
use etalon_wgsl_forge::{emit_shader_ptx, BuiltinKernel, Schedule};

const BLOCK_SIZE: u32 = 64;

/// Fixed input/output allocations; CUDA driver allocations are outside the buffer budget.
#[derive(Debug)]
pub struct CudaIqPowerPlan {
    context: CudaContext,
    module: CudaModule,
    input: CudaBuffer,
    output: CudaBuffer,
    launch: CudaLaunchWorkspace,
    max_pairs: usize,
    completed_runs: u64,
    poisoned: bool,
}

fn buffer_bytes(max_pairs: usize) -> Result<u64, CudaError> {
    if max_pairs == 0 || u32::try_from(max_pairs).is_err() {
        return Err(CudaError::InvalidArgument(
            "CUDA IQ capacity must be in 1..=u32::MAX",
        ));
    }
    max_pairs
        .checked_mul(4)
        .ok_or(CudaError::InvalidArgument("CUDA IQ buffer size overflows"))?;
    (max_pairs as u64)
        .checked_mul(8)
        .ok_or(CudaError::InvalidArgument(
            "CUDA IQ buffer budget overflows",
        ))
}

fn validate_input(
    input: &[u8],
    scale: f32,
    output_len: usize,
    capacity: usize,
) -> Result<u32, CudaError> {
    if input.is_empty() || !input.len().is_multiple_of(4) {
        return Err(CudaError::InvalidArgument(
            "CUDA IQ input must contain complete signed-i16 pairs",
        ));
    }
    if !scale.is_finite() {
        return Err(CudaError::InvalidArgument("CUDA IQ scale must be finite"));
    }
    let count = input.len() / 4;
    if count > capacity || output_len != count {
        return Err(CudaError::InvalidArgument(
            "CUDA IQ input/output length exceeds prepared capacity",
        ));
    }
    u32::try_from(count).map_err(|_| CudaError::InvalidArgument("CUDA IQ pair count exceeds u32"))
}

impl CudaIqPowerPlan {
    pub fn new(
        context: CudaContext,
        max_pairs: usize,
        buffer_budget: u64,
    ) -> Result<Self, CudaError> {
        if buffer_bytes(max_pairs)? > buffer_budget {
            return Err(CudaError::InvalidArgument(
                "CUDA IQ device buffers exceed budget",
            ));
        }
        let shader = emit_shader_ptx(
            &BuiltinKernel::IqPowerI16.spec(),
            Schedule::from_1d(BLOCK_SIZE, 1, 1),
            context.device_info().compute_capability,
        )
        .map_err(|error| CudaError::Compilation(error.to_string()))?;
        // Only the fixed Forge kernel is loaded; execute validates its buffer and parameter ABI.
        let module = unsafe { context.load_ptx(&shader.source) }?;
        module.kernel_cstr(c"iq_power_i16")?;
        Ok(Self {
            input: context.allocate(max_pairs * 4)?,
            output: context.allocate(max_pairs * 4)?,
            launch: CudaLaunchWorkspace::new(3),
            context,
            module,
            max_pairs,
            completed_runs: 0,
            poisoned: false,
        })
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.context.device_info()
    }
    pub fn max_pairs(&self) -> usize {
        self.max_pairs
    }
    pub fn logical_buffer_bytes(&self) -> u64 {
        self.max_pairs as u64 * 8
    }
    pub fn completed_runs(&self) -> u64 {
        self.completed_runs
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Decode packed signed-i16 pairs on the GPU, scale, and compute FP32 power.
    /// Output is cleared on failure. Shorter reuse never reads the unused input tail.
    pub fn execute(
        &mut self,
        input: &[u8],
        big_endian: bool,
        scale: f32,
        output: &mut [f32],
    ) -> Result<usize, CudaError> {
        output.fill(0.0);
        let count = validate_input(input, scale, output.len(), self.max_pairs)?;
        if self.poisoned {
            return Err(CudaError::Poisoned);
        }
        let next_run = self
            .completed_runs
            .checked_add(1)
            .ok_or(CudaError::InvalidArgument("CUDA IQ sequence exhausted"))?;
        let result = (|| {
            self.input.write(input)?;
            let parameters = [scale.to_bits(), u32::from(big_endian), count, 0];
            let kernel = self.module.kernel_cstr(c"iq_power_i16")?;
            // The fixed kernel guards count; both buffers hold count words and params match its ABI.
            unsafe {
                kernel.launch_with_workspace(
                    [count.div_ceil(BLOCK_SIZE), 1, 1],
                    [BLOCK_SIZE, 1, 1],
                    0,
                    &[
                        CudaArgument::Buffer(&self.input),
                        CudaArgument::Buffer(&self.output),
                        CudaArgument::Bytes(bytemuck::cast_slice(&parameters)),
                    ],
                    &mut self.launch,
                )?;
            }
            self.output.read(bytemuck::cast_slice_mut(output))?;
            if output
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(CudaError::InvalidArgument(
                    "CUDA IQ power is outside the finite FP32 range",
                ));
            }
            Ok(count as usize)
        })();
        match result {
            Ok(count) => {
                self.completed_runs = next_run;
                Ok(count)
            }
            Err(error) => {
                output.fill(0.0);
                if matches!(error, CudaError::Driver { .. } | CudaError::Poisoned) {
                    self.poisoned = true;
                }
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_iq_preflight_checks_capacity_shapes_and_scale() {
        assert!(buffer_bytes(0).is_err());
        assert_eq!(buffer_bytes(65).unwrap(), 520);
        assert!(buffer_bytes(usize::MAX).is_err());
        assert_eq!(validate_input(&[0; 4], 1.0, 1, 1).unwrap(), 1);
        for (bytes, scale, output, capacity) in [
            (&[][..], 1.0, 0, 1),
            (&[0; 3][..], 1.0, 1, 1),
            (&[0; 4][..], f32::NAN, 1, 1),
            (&[0; 4][..], f32::INFINITY, 1, 1),
            (&[0; 8][..], 1.0, 2, 1),
            (&[0; 4][..], 1.0, 2, 2),
        ] {
            assert!(validate_input(bytes, scale, output, capacity).is_err());
        }
    }

    #[test]
    #[ignore = "requires NVIDIA CUDA and nvcc"]
    fn cuda_prepared_i16_iq_reuse_matches_cpu() {
        let ordinal = std::env::var("ETALON_CUDA_DEVICE")
            .map(|value| value.parse().unwrap())
            .unwrap_or(0);
        let context = CudaContext::new(ordinal).unwrap();
        eprintln!("CUDA packed IQ qualification: {:?}", context.device_info());
        let mut plan = CudaIqPowerPlan::new(context, 65, 65 * 8).unwrap();
        let samples: Vec<(i16, i16)> = (0..65)
            .map(|index| match index % 5 {
                0 => (i16::MIN, i16::MAX),
                1 => (-1, 1),
                2 => (0, 0),
                3 => (12345, -23456),
                _ => (-3, -4),
            })
            .collect();
        for big_endian in [false, true] {
            for scale in [0.0, 1.0, -0.25, 1.0 / 32768.0, 1.0e-20] {
                for count in [65, 1, 64, 3] {
                    let mut input = Vec::new();
                    for &(i, q) in &samples[..count] {
                        input.extend_from_slice(&if big_endian {
                            i.to_be_bytes()
                        } else {
                            i.to_le_bytes()
                        });
                        input.extend_from_slice(&if big_endian {
                            q.to_be_bytes()
                        } else {
                            q.to_le_bytes()
                        });
                    }
                    let mut output = vec![f32::NAN; count];
                    assert_eq!(
                        plan.execute(&input, big_endian, scale, &mut output)
                            .unwrap(),
                        count
                    );
                    for (&actual, &(i, q)) in output.iter().zip(&samples) {
                        let i = f32::from(i) * scale;
                        let q = f32::from(q) * scale;
                        let expected = i * i + q * q;
                        let tolerance =
                            expected.abs() * 4.0 * f32::EPSILON + 4.0 * f32::from_bits(1);
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "{actual} != {expected}"
                        );
                    }
                }
            }
        }
        let runs = plan.completed_runs();
        let mut output = [1.0];
        assert!(plan.execute(&[0; 8], false, 1.0, &mut output).is_err());
        assert_eq!(output, [0.0]);
        assert_eq!(plan.completed_runs(), runs);
        plan.execute(&[0; 4], false, 1.0, &mut output).unwrap();
        assert_eq!(output, [0.0]);
        assert!(!plan.is_poisoned());
        assert_eq!(plan.logical_buffer_bytes(), 65 * 8);
    }
}
