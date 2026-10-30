use etalon::util::DeviceExt;
use etalon::{BufferDescriptor, BufferUsages, MapMode, PollType};
use etalon_test::{apply, fail_if, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};
use std::mem::size_of_val;

pub fn all_tests(vec: &mut Vec<etalon_test::GpuTestInitializer>) {
    vec.push(ARRAY_SIZE_OVERRIDES);
}

const SHADER: &str = r#"
    const testing_shared: array<i32, 1> = array(8);
    override n = testing_shared[0u];
    override m = testing_shared[0u];

    var<workgroup> arr: array<u32, n - 2>;

    @group(0) @binding(0)
    var<storage, read_write> output: array<u32>;

    // When `n` is overridden to 14 above, it will generate the type `array<u32, 14 - 2>` which
    // already exists in the program because of variable declaration. Ensures naga does not panic
    // when this happens.
    //
    // See https://github.com/mediaprophet/etalon/issues/6722 for more info.
    var<workgroup> testing0: array<u32, 12>;

    // Tests whether two overrides that are initialized by the same expression
    // crashes the unique types arena
    //
    // See https://github.com/mediaprophet/etalon/pull/6787#pullrequestreview-2576905294
    var<workgroup> testing1: array<u32, n>;
    var<workgroup> testing2: array<u32, m>;

    @compute @workgroup_size(1) fn main() {
        // 1d spiral
        for (var i = 0; i < n - 2; i++) {
            arr[i] = u32(n - 2 - i);
            if (i + 1 < (n + (n % 2)) / 2) {
                arr[i] -= 1u;
            }
        }
        var i = 0u;
        var j = 1u;
        while (i != j) {
            // non-commutative
            output[0] = output[0] * arr[i] + arr[i];
            j = i;
            i = arr[i];
        }
    }
"#;

#[apply(gpu_test!)]
static ARRAY_SIZE_OVERRIDES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(etalon::Limits::default())
            // https://github.com/mediaprophet/etalon/issues/9184
            .expect_fail(
                etalon_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline")
                    .unexpected_error("Unexpected Vulkan error: ERROR_INITIALIZATION_FAILED"),
            ),
    )
    .run_async(move |ctx| async move {
        array_size_overrides(&ctx, None, &[534], false).await;
        array_size_overrides(&ctx, Some(14), &[286480122], false).await;
        // Array size override evaluation test cases
    });

async fn array_size_overrides(
    ctx: &TestingContext,
    n: Option<u32>,
    out: &[u32],
    should_fail: bool,
) {
    let module = ctx
        .device
        .create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
    let pipeline_options = etalon::PipelineCompilationOptions {
        constants: &[("n", f64::from(n.unwrap_or(0)))],
        ..Default::default()
    };
    let compute_pipeline = fail_if(
        &ctx.device,
        should_fail,
        || {
            ctx.device
                .create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: if n.is_some() {
                        pipeline_options
                    } else {
                        etalon::PipelineCompilationOptions::default()
                    },
                    cache: None,
                })
        },
        None,
    );
    if should_fail {
        return;
    }
    let init: &[u32] = &[0];
    let init_size: u64 = size_of_val(init).try_into().unwrap();
    let buffer = DeviceExt::create_buffer_init(
        &ctx.device,
        &etalon::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(init),
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_DST
                | etalon::BufferUsages::COPY_SRC,
        },
    );
    let mapping_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("mapping buffer"),
        size: init_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group_entries = [etalon::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }];
        let bind_group = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &mapping_buffer, 0, init_size);
    ctx.queue.submit(Some(encoder.finish()));

    mapping_buffer.slice(..).map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

    let mapped = mapping_buffer.slice(..).get_mapped_range().unwrap();

    let typed: &[u32] = bytemuck::cast_slice(&mapped);
    assert_eq!(typed, out);
}
