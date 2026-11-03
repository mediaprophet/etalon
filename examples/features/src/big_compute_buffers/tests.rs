use super::*;
use etalon_test::{apply, gpu_test, GpuTestConfiguration, TestParameters};

const TEST_INPUT: &[f32] = &[0.0; (1 << 27) / std::mem::size_of::<f32>() * 8];

#[apply(gpu_test!)]
pub static TWO_BUFFERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::BUFFER_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
            .instance_flags(etalon::InstanceFlags::GPU_BASED_VALIDATION)
            .downlevel_flags(etalon::DownlevelFlags::COMPUTE_SHADERS)
            .limits(etalon::Limits {
                max_buffer_size: MAX_BUFFER_SIZE,
                max_binding_array_elements_per_shader_stage: 8,
                ..Default::default()
            })
            .expect_fail(
                etalon_test::FailureCase::molten_vk()
                    // https://github.com/mediaprophet/etalon/issues/9184
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline")
                    // https://github.com/mediaprophet/etalon/issues/9849
                    .panic("assertion failed: produced"),
            ),
    )
    .run_async(|ctx| async move { assert_execute_gpu(&ctx.device, &ctx.queue, TEST_INPUT).await });

async fn assert_execute_gpu(device: &etalon::Device, queue: &etalon::Queue, input: &[f32]) {
    let expected_len = input.len();
    let produced = execute_gpu_inner(device, queue, input).await;

    assert_produced(&produced, expected_len);
}

fn assert_produced(produced: &[f32], expected_len: usize) {
    assert_eq!(produced.len(), expected_len);
    if let Some((index, value)) = produced
        .iter()
        .enumerate()
        .find(|(_, value)| **value != 1.0)
    {
        let elements_per_buffer = MAX_BUFFER_SIZE as usize / std::mem::size_of::<f32>();
        panic!(
            "assertion failed: produced values must all equal 1.0; first mismatch at element {index} \
             (buffer {}, offset {}): {value:?}, bits {:#010x}",
            index / elements_per_buffer,
            index % elements_per_buffer,
            value.to_bits(),
        );
    }
}
