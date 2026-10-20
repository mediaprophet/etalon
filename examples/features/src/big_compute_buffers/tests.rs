use super::*;
use etalon_test::{apply, gpu_test, GpuTestConfiguration, TestParameters};

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
    .run_async(|ctx| {
        // The test environment's GPU reports 134MB as the max storage buffer size.https://github.com/mediaprophet/etalon/actions/runs/11001397782/job/30546188996#step:12:1096
        const SIZE: usize = (1 << 27) / std::mem::size_of::<f32>() * 8;
        // 2 Buffers worth, of 0.0s.
        let input = &[0.0; SIZE];

        async move { assert_execute_gpu(&ctx.device, &ctx.queue, input).await }
    });

async fn assert_execute_gpu(device: &etalon::Device, queue: &etalon::Queue, input: &[f32]) {
    let expected_len = input.len();
    let produced = execute_gpu_inner(device, queue, input).await;

    assert_eq!(produced.len(), expected_len);
    assert!(produced.into_iter().all(|v| v == 1.0));
}
