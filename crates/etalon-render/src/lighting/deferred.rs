/// Active GPU compute pipeline owner for Continuous EMF Spectral Deferred Resolve (`DEFERRED_RESOLVE_WGSL`).
pub struct DeferredSpectralResolvePipelineGpu {
    pub pipeline: etalon::ComputePipeline,
    pub frame_bgl: etalon::BindGroupLayout,
    pub gbuffer_bgl: etalon::BindGroupLayout,
    pub output_bgl: etalon::BindGroupLayout,
}

impl DeferredSpectralResolvePipelineGpu {
    pub fn new(device: &etalon::Device) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia_deferred_resolve_wgsl"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("../shaders/deferred_resolve.wgsl").into(),
            ),
        });

        let frame_bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("qualia_deferred_resolve_frame_bgl"),
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
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: false },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Sampler(etalon::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let gbuffer_bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("qualia_deferred_resolve_gbuffer_bgl"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: false },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Float { filterable: false },
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: etalon::ShaderStages::COMPUTE,
                    ty: etalon::BindingType::Texture {
                        sample_type: etalon::TextureSampleType::Depth,
                        view_dimension: etalon::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let output_bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("qualia_deferred_resolve_output_bgl"),
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::StorageTexture {
                    access: etalon::StorageTextureAccess::WriteOnly,
                    format: etalon::TextureFormat::Rgba16Float,
                    view_dimension: etalon::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("qualia_deferred_resolve_pl"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&gbuffer_bgl), Some(&output_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("qualia_deferred_resolve_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_deferred_resolve"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            frame_bgl,
            gbuffer_bgl,
            output_bgl,
        }
    }
}
