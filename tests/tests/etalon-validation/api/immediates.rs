//! Validation tests for `var<immediate>`

use etalon_test::fail;

const COMPUTE_SHADER: &str = "
    var<immediate> im: vec4<f32>;

    @group(0) @binding(0)
    var<storage, read_write> output: vec4<f32>;

    @compute @workgroup_size(1)
    fn main() {
        output = im;
    }
";

fn setup_compute() -> (
    etalon::Device,
    etalon::Queue,
    etalon::ComputePipeline,
    etalon::BindGroup,
) {
    let (device, queue) = etalon::Device::noop(&etalon::DeviceDescriptor {
        required_features: etalon::Features::IMMEDIATES,
        required_limits: etalon::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: None,
        source: etalon::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: None,
        entries: &[etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::COMPUTE,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 16,
    });

    let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 16,
        usage: etalon::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (device, queue, pipeline, bind_group)
}

#[test]
fn dispatch_without_setting_immediates_fails() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    fail(&device, || encoder.finish(), Some("not all immediate data required by the pipeline has been set via set_immediates (missing byte ranges: 0..16)"));
}

#[test]
fn dispatch_with_partial_immediates_fails() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 8]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    fail(&device, || encoder.finish(), Some("not all immediate data required by the pipeline has been set via set_immediates (missing byte ranges: 8..16)"));
}

#[test]
fn dispatch_with_all_immediates_set_succeeds() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    etalon_test::valid(&device, || encoder.finish());
}

#[test]
fn dispatch_with_incremental_immediates_succeeds() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 8]);
        pass.set_immediates(8, &[0u8; 8]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    etalon_test::valid(&device, || encoder.finish());
}

const STRUCT_SHADER: &str = "
    struct S {
        a: f32,
        // 12 bytes padding
        b: vec4<f32>,
    }
    var<immediate> im: S;

    @group(0) @binding(0)
    var<storage, read_write> output: vec4<f32>;

    @compute @workgroup_size(1)
    fn main() {
        output = im.b;
    }
";

#[test]
fn struct_padding_slots_not_required() {
    let (device, _q) = etalon::Device::noop(&etalon::DeviceDescriptor {
        required_features: etalon::Features::IMMEDIATES,
        required_limits: etalon::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: None,
        source: etalon::ShaderSource::Wgsl(STRUCT_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: None,
        entries: &[etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::COMPUTE,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 32,
    });

    let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 16,
        usage: etalon::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // skip padding at bytes 4..16
        pass.set_immediates(0, &[0u8; 4]);
        pass.set_immediates(16, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    etalon_test::valid(&device, || encoder.finish());
}

const NO_IMMEDIATES_SHADER: &str = "
    @group(0) @binding(0)
    var<storage, read_write> output: u32;

    @compute @workgroup_size(1)
    fn main() {
        output = 42u;
    }
";

#[test]
fn pipeline_without_immediates_needs_none() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let sm = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: None,
        source: etalon::ShaderSource::Wgsl(NO_IMMEDIATES_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: None,
        entries: &[etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::COMPUTE,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 4,
        usage: etalon::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    etalon_test::valid(&device, || encoder.finish());
}

#[test]
fn auto_layout_infers_immediate_size() {
    let (device, _q) = etalon::Device::noop(&etalon::DeviceDescriptor {
        required_features: etalon::Features::IMMEDIATES,
        required_limits: etalon::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: None,
        source: etalon::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 16,
        usage: etalon::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    etalon_test::valid(&device, || encoder.finish());
}

const RENDER_SHADER_MULTI_IMMEDIATES: &str = "
    enable f16;
    struct ImmediateA {
        @size(34)
        a1: f16,
    }
    var<immediate> im_a: ImmediateA;

    struct ImmediateB {
        b1: u32,
        b2: vec4<f32>,
    }
    var<immediate> im_b: ImmediateB;

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) index: u32,
    }

    @vertex fn vertex() -> VertexOutput {
        return VertexOutput(vec4f(1.0, 0.0, 0.0, 1.0), u32(im_a.a1));
    }

    @fragment fn fragment(
        @location(0) @interpolate(flat) ix: u32,
     ) -> @location(0) vec4f {
        return im_b.b2;
    }
";

fn begin_render_pass<'b, 'a: 'b>(
    output_texture_view: &'a etalon::TextureView,
    encoder: &'a mut etalon::CommandEncoder,
    f: impl Fn(&mut etalon::RenderPass<'b>),
) {
    let mut render_pass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(etalon::RenderPassColorAttachment {
            view: output_texture_view,
            depth_slice: None,
            resolve_target: None,
            ops: etalon::Operations {
                load: etalon::LoadOp::Clear(etalon::Color::default()),
                store: etalon::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    f(&mut render_pass);
}

fn setup_render() -> (etalon::Device, etalon::Queue, etalon::TextureView) {
    let (device, queue) = etalon::Device::noop(&etalon::DeviceDescriptor {
        required_features: etalon::Features::IMMEDIATES | etalon::Features::SHADER_F16,
        required_limits: etalon::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let output_texture = device.create_texture(&etalon::TextureDescriptor {
        size: etalon::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8UnormSrgb,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT,
        label: Some("Output Texture"),
        view_formats: &[],
    });
    let output_texture_view = output_texture.create_view(&Default::default());

    (device, queue, output_texture_view)
}

#[test]
fn render_multi_immediates_with_smaller_layout_immediate_size_fails() {
    let (device, _queue, output_texture_view) = setup_render();
    let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: etalon::ShaderSource::Wgsl(RENDER_SHADER_MULTI_IMMEDIATES.into()),
    });
    let create_pipeline = |immediate_size| {
        device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[],
                    immediate_size,
                }),
            ),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: None,
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: None,
                targets: &[Some(output_texture_view.texture().format().into())],
                compilation_options: Default::default(),
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    fail(&device, || create_pipeline(32), Some("Pipeline layout immediate size (32) must be >= the required immediate size (34) of the shader entry point"));

    fail(&device, || create_pipeline(34), Some("Immediate data has range bound 34 which is not aligned to IMMEDIATE_DATA_ALIGNMENT (4)"));

    etalon_test::valid(&device, || create_pipeline(36));
}

#[test]
fn render_multi_immediates_auto_layout_with_all_immediates_set_succeeds() {
    let (device, _queue, output_texture_view) = setup_render();
    let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: etalon::ShaderSource::Wgsl(RENDER_SHADER_MULTI_IMMEDIATES.into()),
    });
    let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: None,
        vertex: etalon::VertexState {
            module: &shader,
            entry_point: None,
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(etalon::FragmentState {
            module: &shader,
            entry_point: None,
            targets: &[Some(output_texture_view.texture().format().into())],
            compilation_options: Default::default(),
        }),
        primitive: etalon::PrimitiveState {
            topology: etalon::PrimitiveTopology::PointList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: etalon::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    fn do_encoding<'a>(
        encoder: &mut dyn etalon::util::RenderEncoder<'a>,
        pipeline: &'a etalon::RenderPipeline,
    ) {
        encoder.set_pipeline(pipeline);
        encoder.set_immediates(0, &[0u8; 4]);
        encoder.set_immediates(16, &[0u8; 16]);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = device.create_command_encoder(&Default::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        do_encoding(pass, &pipeline);
    });

    etalon_test::valid(&device, || encoder.finish());

    let mut encoder = device.create_command_encoder(&Default::default());
    let mut bundle_encoder =
        device.create_render_bundle_encoder(&etalon::RenderBundleEncoderDescriptor {
            color_formats: &[Some(output_texture_view.texture().format())],
            sample_count: 1,
            ..etalon::RenderBundleEncoderDescriptor::default()
        });
    do_encoding(&mut bundle_encoder, &pipeline);
    let bundle = bundle_encoder.finish(&etalon::RenderBundleDescriptor::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        pass.execute_bundles([&bundle]);
    });

    etalon_test::valid(&device, || encoder.finish());
}
