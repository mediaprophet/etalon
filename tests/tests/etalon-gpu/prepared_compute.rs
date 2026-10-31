use std::time::Duration;

use etalon::{PreparedComputeDescriptor, PreparedComputeError};
use etalon_test::{apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters};

const SHADER: &str = r"
struct Parameters { active: u32, addend: u32, padding: vec2<u32> }
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < parameters.active { output[id.x] = input[id.x] + parameters.addend; }
}
";

fn descriptor() -> PreparedComputeDescriptor<'static> {
    PreparedComputeDescriptor {
        shader: SHADER,
        entry_point: "main",
        input_size: 64,
        parameter_size: 16,
        output_size: 64,
        workgroups: [1; 3],
        buffer_budget: 288,
    }
}

fn diagnostic_flags() -> etalon::InstanceFlags {
    etalon::InstanceFlags::DEBUG
        | etalon::InstanceFlags::VALIDATION
        | etalon::InstanceFlags::GPU_BASED_VALIDATION
        | etalon::InstanceFlags::DEBUG_PRINTF
}

fn vulkan_parameters() -> TestParameters {
    TestParameters::default()
        .test_features_limits()
        .remove_instance_flags(diagnostic_flags() | etalon::InstanceFlags::STRICT_WEBGPU_COMPLIANCE)
        .skip(FailureCase::backend(etalon::Backends::all() - etalon::Backends::VULKAN))
}

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([PREPARED_COMPUTE_REUSE, PREPARED_COMPUTE_VALIDATION, PREPARED_COMPUTE_UNSUPPORTED]);
}

#[apply(gpu_test!)]
static PREPARED_COMPUTE_REUSE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(vulkan_parameters())
    .run_sync(|ctx| {
        let mut lane = ctx.adapter.prepare_compute(&descriptor()).unwrap();
        let mut parameters = [0_u8; 16];
        parameters[..4].copy_from_slice(&16_u32.to_le_bytes());
        parameters[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let mut output = [0_u8; 64];
        assert_eq!(lane.execute(&[3; 64], &parameters, &mut output, Duration::from_secs(10)), Ok(1));
        assert!(output.chunks_exact(4).all(|word| word == (0x03030303_u32 + 2).to_le_bytes()));

        parameters[..4].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(lane.execute(&7_u32.to_le_bytes(), &parameters, &mut output, Duration::from_secs(10)), Ok(2));
        assert_eq!(&output[..4], &9_u32.to_le_bytes());
        assert!(output[4..].iter().all(|&byte| byte == 0));

        output.fill(0xff);
        assert_eq!(lane.execute(&[0; 65], &parameters, &mut output, Duration::ZERO), Err(PreparedComputeError::Capacity));
        assert_eq!(output, [0; 64]);
        assert!(!lane.is_poisoned());
        assert_eq!(lane.completed_runs(), 2);
        assert_eq!(lane.execute(&[], &parameters[..12], &mut output, Duration::ZERO), Err(PreparedComputeError::InvalidDescriptor));

        parameters[..4].copy_from_slice(&16_u32.to_le_bytes());
        ctx.instance.poll_all(false);
        assert_eq!(lane.execute(&[], &parameters, &mut output, Duration::from_secs(10)), Ok(3));
        assert!(output.chunks_exact(4).all(|word| word == 2_u32.to_le_bytes()));
        assert!(!lane.is_poisoned());
        lane.retire().unwrap();
        assert!(lane.is_poisoned());
        output.fill(0xff);
        assert_eq!(lane.execute(&[], &parameters, &mut output, Duration::ZERO), Err(PreparedComputeError::Poisoned));
        assert_eq!(output, [0; 64]);
    });

#[apply(gpu_test!)]
static PREPARED_COMPUTE_VALIDATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(vulkan_parameters())
    .run_sync(|ctx| {
        let mut desc = descriptor();
        desc.buffer_budget -= 1;
        assert_eq!(ctx.adapter.prepare_compute(&desc).unwrap_err().reason, PreparedComputeError::Capacity);
        desc = descriptor();
        desc.shader = "invalid shader";
        assert_eq!(ctx.adapter.prepare_compute(&desc).unwrap_err().reason, PreparedComputeError::InvalidDescriptor);
        desc = descriptor();
        desc.parameter_size = 4;
        assert_eq!(ctx.adapter.prepare_compute(&desc).unwrap_err().reason, PreparedComputeError::InvalidDescriptor);
        desc = descriptor();
        desc.workgroups[0] = u32::MAX;
        assert_eq!(ctx.adapter.prepare_compute(&desc).unwrap_err().reason, PreparedComputeError::InvalidDescriptor);
    });

#[apply(gpu_test!)]
static PREPARED_COMPUTE_UNSUPPORTED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop().skip(FailureCase::backend(etalon::Backends::VULKAN)))
    .run_sync(|ctx| {
        assert_eq!(ctx.adapter.prepare_compute(&descriptor()).unwrap_err().reason, PreparedComputeError::UnsupportedBackend);
    });
