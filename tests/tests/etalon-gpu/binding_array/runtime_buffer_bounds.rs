use std::num::{NonZeroU32, NonZeroU64};

use etalon::{util::DeviceExt, *};
use etalon_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(BINDING_ARRAY_RUNTIME_BUFFER_BOUNDS);
}

#[apply(gpu_test!)]
static BINDING_ARRAY_RUNTIME_BUFFER_BOUNDS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::BUFFER_BINDING_ARRAY | Features::STORAGE_RESOURCE_BINDING_ARRAY)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits {
                max_binding_array_elements_per_shader_stage: 8,
                ..Default::default()
            }),
    )
    .run_async(|ctx| async move {
        let source = r"
            enable etalon_binding_array;
            struct Data { inner: array<u32> }
            @group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 8>;
            @compute @workgroup_size(1)
            fn main(@builtin(workgroup_id) id: vec3u) {
                for (var index = 0u; index < 16u; index++) {
                    buffers[id.x].inner[index] = buffers[id.x].inner[index] + 1u;
                }
            }
        ";
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("runtime array length differs from binding array length"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let buffers: Vec<_> = (0..8u32)
            .map(|buffer| {
                let values: Vec<u32> = (0..16).map(|index| buffer * 100 + index).collect();
                ctx.device.create_buffer_init(&util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&values),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                })
            })
            .collect();
        let layout = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(64),
                    },
                    count: NonZeroU32::new(8),
                }],
            });
        let bindings: Vec<_> = buffers
            .iter()
            .map(Buffer::as_entire_buffer_binding)
            .collect();
        let group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::BufferArray(&bindings),
            }],
        });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let readback = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 8 * 64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(8, 1, 1);
        }
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.copy_buffer_to_buffer(buffer, 0, &readback, index as u64 * 64, 64);
        }
        ctx.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(MapMode::Read, |_| {});
        ctx.device.poll(PollType::wait_indefinitely()).unwrap();
        let mapped = readback.get_mapped_range(..).unwrap();
        let values: Vec<u32> = bytemuck::allocation::pod_collect_to_vec(&mapped);
        let expected: Vec<u32> = (0..8)
            .flat_map(|buffer| (0..16).map(move |index| buffer * 100 + index + 1))
            .collect();
        assert_eq!(values, expected);
    });
