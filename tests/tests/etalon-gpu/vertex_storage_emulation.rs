use etalon::util::DeviceExt;
use etalon_test::{apply, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<etalon_test::GpuTestInitializer>) {
    vec.push(VERTEX_STORAGE_INSTANCED);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    offset: [f32; 4],
    color: [f32; 4],
}

const WGSL: &str = r#"
struct InstanceData {
    offset: vec4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read> instances: array<InstanceData>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-0.4, -0.4),
        vec2<f32>(0.4, -0.4),
        vec2<f32>(0.0, 0.4),
    );
    let inst = instances[iid];
    var out: VsOut;
    out.pos = vec4<f32>(corners[vid] + inst.offset.xy, 0.0, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

#[apply(gpu_test!)]
static VERTEX_STORAGE_INSTANCED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().limits(etalon::Limits::downlevel_defaults()))
    .run_async(vertex_storage_instanced);

async fn vertex_storage_instanced(ctx: TestingContext) {
    let flags = ctx.adapter.get_downlevel_capabilities().flags;
    if !flags.intersects(
        etalon::DownlevelFlags::VERTEX_STORAGE
            | etalon::DownlevelFlags::READ_ONLY_STORAGE_EMULATION,
    ) {
        return;
    }

    let shader = ctx
        .device
        .create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("vertex_storage_instanced"),
            source: etalon::ShaderSource::Wgsl(WGSL.into()),
        });

    let instances = [
        InstanceData {
            offset: [-0.5, 0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        InstanceData {
            offset: [0.5, 0.0, 0.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],
        },
    ];
    let instance_buf = ctx
        .device
        .create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: None,
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::VERTEX,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let bg = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: instance_buf.as_entire_binding(),
        }],
    });

    let layout = ctx
        .device
        .create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::TextureFormat::Rgba8Unorm.into())],
            }),
            primitive: etalon::PrimitiveState::default(),
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

    let target = ctx.device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    let readback = ctx.device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: etalon::Operations {
                    load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                    store: etalon::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..2);
    }
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        etalon::TexelCopyBufferInfo {
            buffer: &readback,
            layout: etalon::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        etalon::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(etalon::MapMode::Read, |_| {});
    ctx.device
        .poll(etalon::PollType::wait_indefinitely())
        .unwrap();
    let data = slice.get_mapped_range().unwrap();
    // Row 2 (y=0 center): x=1 should be red (instance 0), x=3 should be green (instance 1).
    let row2 = &data[2 * 256..2 * 256 + 16];
    assert_eq!(&row2[4..8], &[255, 0, 0, 255]);
    assert_eq!(&row2[12..16], &[0, 255, 0, 255]);
}
