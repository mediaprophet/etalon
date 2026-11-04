use super::*;
/// GPU-resident compute pipeline and indirect stream buffers for `INSTANCE_CULL_INDIRECT_WGSL`.
pub struct IndirectCullPipelineGpu {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buf: wgpu::Buffer,
    pub instance_buf: wgpu::Buffer,
    pub draw_commands_buf: wgpu::Buffer,
    pub visible_mesh_buf: wgpu::Buffer,
    pub visible_holo_buf: wgpu::Buffer,
    pub visible_surfel_buf: wgpu::Buffer,
    pub counters_buf: wgpu::Buffer,
    pub hzb_texture: wgpu::Texture,
    pub hzb_view: wgpu::TextureView,
    pub max_instances: u32,
    pub hzb_mip_count: u32,
}

impl IndirectCullPipelineGpu {
    /// Creates the GPU indirect culling compute pipeline, stream buffers, and HZB pyramid texture.
    pub fn new(device: &wgpu::Device, max_instances: u32, hzb_base_dim: u32) -> Self {
        let cap = max_instances.max(64) as u64;
        let hzb_dim = hzb_base_dim.max(16).next_power_of_two();
        let hzb_mip_count = (hzb_dim.ilog2() + 1).min(8);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qualia_instance_cull_indirect_wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/instance_cull_indirect.wgsl").into(),
            ),
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qualia_indirect_cull_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qualia_indirect_cull_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qualia_indirect_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qualia_indirect_cull_uniforms"),
            size: std::mem::size_of::<CullUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qualia_indirect_cull_instances"),
            size: cap * std::mem::size_of::<InstanceCullDescriptor>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_commands_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qualia_indirect_draw_commands"),
            size: cap * std::mem::size_of::<DrawIndexedIndirectCommand>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visible_mesh_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qualia_visible_mesh_instances"),
            size: cap * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_holo_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qualia_visible_holo_instances"),
            size: cap * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_surfel_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qualia_visible_surfel_instances"),
            size: cap * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counters_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qualia_indirect_stream_counters"),
            size: std::mem::size_of::<IndirectCullStreamCounts>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hzb_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qualia_hzb_depth_pyramid"),
            size: wgpu::Extent3d {
                width: hzb_dim,
                height: hzb_dim,
                depth_or_array_layers: 1,
            },
            mip_level_count: hzb_mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let hzb_view = hzb_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
