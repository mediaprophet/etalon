pub struct ProjectionsPipelines {
    #[cfg(not(target_arch = "wasm32"))]
    pub dual_gemv_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub dual_gemv_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub dual_gemv_mr_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub triple_gemv_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub triple_gemv_pipeline: wgpu::ComputePipeline,
}

pub(super) fn build(
    _device: &wgpu::Device,
    _native_pipeline_cache: &Option<wgpu::PipelineCache>,
    _options: super::PipelineOptions,
) -> ProjectionsPipelines {
    #[cfg(not(target_arch = "wasm32"))]
    let dual_gemv_bind_layout =
        _device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DualGemvBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    #[cfg(not(target_arch = "wasm32"))]
    let (dual_gemv_pipeline, dual_gemv_mr_pipeline) = {
        let module = _device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dual GEMV Shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::DUAL_GEMV_WGSL.into()),
        });
        let layout = _device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DualGemvPL"),
            bind_group_layouts: &[Some(&dual_gemv_bind_layout)],
            immediate_size: 0,
        });
        let dual = _device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dual GEMV Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("coop_gemv_dual"),
            compilation_options: Default::default(),
            cache: _native_pipeline_cache.as_ref(),
        });
        let dual_mr = _device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dual GEMV Multi-Row Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("coop_gemv_dual_mr"),
            compilation_options: Default::default(),
            cache: _native_pipeline_cache.as_ref(),
        });
        (dual, dual_mr)
    };

    // Triple Q+K+V GEMV (shared act, GQA) — resident mega-pass: 3 dispatches → 1.
    #[cfg(not(target_arch = "wasm32"))]
    let triple_gemv_bind_layout =
        _device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TripleGemvBGL"),
            entries: &[
                // 0 input, 1 Wq, 2 params, 3 out_q, 4 Wk, 5 out_k, 6 Wv, 7 out_v
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    #[cfg(not(target_arch = "wasm32"))]
    let triple_gemv_pipeline = {
        let module = _device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Triple GEMV Shader"),
            source: wgpu::ShaderSource::Wgsl(super::shaders::TRIPLE_GEMV_WGSL.into()),
        });
        let layout = _device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TripleGemvPL"),
            bind_group_layouts: &[Some(&triple_gemv_bind_layout)],
            immediate_size: 0,
        });
        _device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Triple GEMV Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("coop_gemv_triple"),
            compilation_options: Default::default(),
            cache: _native_pipeline_cache.as_ref(),
        })
    };

    ProjectionsPipelines {
        #[cfg(not(target_arch = "wasm32"))]
        dual_gemv_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        dual_gemv_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        dual_gemv_mr_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        triple_gemv_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        triple_gemv_pipeline,
    }
}
