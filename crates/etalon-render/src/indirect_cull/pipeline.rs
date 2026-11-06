use super::*;
/// GPU-resident compute pipeline and indirect stream buffers for `INSTANCE_CULL_INDIRECT_WGSL`.
pub struct IndirectCullPipelineGpu {
    pub pipeline: etalon::ComputePipeline,
    pub bind_group_layout: etalon::BindGroupLayout,
    pub uniform_buf: etalon::Buffer,
    pub instance_buf: etalon::Buffer,
    pub draw_commands_buf: etalon::Buffer,
    pub visible_mesh_buf: etalon::Buffer,
    pub visible_holo_buf: etalon::Buffer,
    pub visible_surfel_buf: etalon::Buffer,
    pub counters_buf: etalon::Buffer,
    pub hzb_texture: etalon::Texture,
    pub hzb_view: etalon::TextureView,
    pub max_instances: u32,
    pub hzb_mip_count: u32,
}

impl IndirectCullPipelineGpu {
    /// Creates the GPU indirect culling compute pipeline, stream buffers, and HZB pyramid texture.
    pub fn new(device: &etalon::Device, max_instances: u32, hzb_base_dim: u32) -> Self {
        let cap = max_instances.max(64) as u64;
        let hzb_dim = hzb_base_dim.max(16).next_power_of_two();
        let hzb_mip_count = (hzb_dim.ilog2() + 1).min(8);

        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia_instance_cull_indirect_wgsl"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("../shaders/instance_cull_indirect.wgsl").into(),
            ),
        });

        let storage_entry = |binding: u32, read_only: bool| etalon::BindGroupLayoutEntry {
            binding,
            visibility: etalon::ShaderStages::COMPUTE,
            ty: etalon::BindingType::Buffer {
                ty: etalon::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("qualia_indirect_cull_bgl"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                etalon::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: false },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("qualia_indirect_cull_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("qualia_indirect_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_indirect_cull_uniforms"),
            size: std::mem::size_of::<CullUniformsGpu>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_indirect_cull_instances"),
            size: cap * std::mem::size_of::<InstanceCullDescriptor>() as u64,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_commands_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_indirect_draw_commands"),
            size: cap * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64,
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::INDIRECT
                | etalon::BufferUsages::COPY_DST
                | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visible_mesh_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_visible_mesh_instances"),
            size: cap * 4,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_holo_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_visible_holo_instances"),
            size: cap * 4,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_surfel_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_visible_surfel_instances"),
            size: cap * 4,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counters_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_indirect_stream_counters"),
            size: std::mem::size_of::<IndirectCullStreamCounts>() as u64,
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_DST
                | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hzb_texture = device.create_texture(&etalon::TextureDescriptor {
            label: Some("qualia_hzb_depth_pyramid"),
            size: etalon::Extent3d {
                width: hzb_dim,
                height: hzb_dim,
                depth_or_array_layers: 1,
            },
            mip_level_count: hzb_mip_count,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::R32Float,
            usage: etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let hzb_view = hzb_texture.create_view(&etalon::TextureViewDescriptor::default());

        Self {
            pipeline,
            bind_group_layout,
            uniform_buf,
            instance_buf,
            draw_commands_buf,
            visible_mesh_buf,
            visible_holo_buf,
            visible_surfel_buf,
            counters_buf,
            hzb_texture,
            hzb_view,
            max_instances: cap as u32,
            hzb_mip_count,
        }
    }
}
