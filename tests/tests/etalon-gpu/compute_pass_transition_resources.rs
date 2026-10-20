use etalon_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COMPUTE_PASS_TRANSITION_RESOURCES);
}

#[apply(gpu_test!)]
static COMPUTE_PASS_TRANSITION_RESOURCES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 128,
            usage: etalon::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        let mut pass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.transition_resources(
            [etalon::BufferTransition {
                buffer: &buffer,
                state: etalon::BufferUses::STORAGE_READ_WRITE,
            }]
            .into_iter(),
            core::iter::empty(),
        );

        drop(pass);

        ctx.queue.submit([encoder.finish()]);
    });
