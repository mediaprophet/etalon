#[cfg(target_arch = "wasm32")]
const MC8_UNIFORM_ALIGN: usize = 256;

pub struct CommonPipelines {
    pub embedding_pipeline: wgpu::ComputePipeline,
    pub attention_pipeline: wgpu::ComputePipeline,
    pub elem_rms_norm_pipeline: wgpu::ComputePipeline,
    pub elem_silu_mul_pipeline: wgpu::ComputePipeline,
    pub elem_add_residual_pipeline: wgpu::ComputePipeline,
    #[cfg(target_arch = "wasm32")]
    pub mc8_attn_bind_layout: wgpu::BindGroupLayout,
    #[cfg(target_arch = "wasm32")]
    pub mc8_elem_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub embedding_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub attention_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub elem_silu_mul_bind_layout: wgpu::BindGroupLayout,
}

pub(super) fn build(
    device: &wgpu::Device,
    native_pipeline_cache: &Option<wgpu::PipelineCache>,
    _options: super::PipelineOptions,
) -> CommonPipelines {
    let emb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Quantized Embedding Shader"),
        source: wgpu::ShaderSource::Wgsl(super::shaders::QUANTIZED_EMBEDDING_WGSL.into()),
    });
    let embedding_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Quantized Embedding Pipeline"),
        layout: None,
        module: &emb_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });

    let attn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Fused Attention Shader"),
        source: wgpu::ShaderSource::Wgsl(super::shaders::FUSED_ATTENTION_WGSL.into()),
    });
    #[cfg(target_arch = "wasm32")]
    let mc8_attn_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MC8AttnBGL"),
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
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(MC8_UNIFORM_ALIGN as u64),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    #[cfg(target_arch = "wasm32")]
    let mc8_attn_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MC8AttnPL"),
        bind_group_layouts: &[Some(&mc8_attn_bind_layout)],
        immediate_size: 0,
    });
    #[cfg(target_arch = "wasm32")]
    let attention_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Fused Attention Pipeline"),
        layout: Some(&mc8_attn_pipeline_layout),
        module: &attn_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(not(target_arch = "wasm32"))]
    let attention_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Fused Attention Pipeline"),
        layout: None,
        module: &attn_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });

    let elem_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Wasm Elementwise Shader"),
        source: wgpu::ShaderSource::Wgsl(super::shaders::WASM_ELEMENTWISE_WGSL.into()),
    });
    #[cfg(target_arch = "wasm32")]
    let mc8_elem_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MC8ElemBGL"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(MC8_UNIFORM_ALIGN as u64),
                },
                count: None,
            },
        ],
    });
    #[cfg(target_arch = "wasm32")]
    let mc8_elem_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MC8ElemPL"),
        bind_group_layouts: &[Some(&mc8_elem_bind_layout)],
        immediate_size: 0,
    });
    #[cfg(target_arch = "wasm32")]
    let elem_rms_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ElemRmsNorm"),
        layout: Some(&mc8_elem_pipeline_layout),
        module: &elem_shader,
        entry_point: Some("rms_norm_batch"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(target_arch = "wasm32")]
    let elem_silu_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ElemSiluMul"),
        layout: Some(&mc8_elem_pipeline_layout),
        module: &elem_shader,
        entry_point: Some("silu_mul_main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(target_arch = "wasm32")]
    let elem_add_residual_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElemAddResidual"),
            layout: Some(&mc8_elem_pipeline_layout),
            module: &elem_shader,
            entry_point: Some("add_residual_main"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
    #[cfg(not(target_arch = "wasm32"))]
    let elem_rms_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ElemRmsNorm"),
        layout: None,
        module: &elem_shader,
        entry_point: Some("rms_norm_batch"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(not(target_arch = "wasm32"))]
    let elem_silu_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ElemSiluMul"),
        layout: None,
        module: &elem_shader,
        entry_point: Some("silu_mul_main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(not(target_arch = "wasm32"))]
    let elem_add_residual_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElemAddResidual"),
            layout: None,
            module: &elem_shader,
            entry_point: Some("add_residual_main"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });

    #[cfg(not(target_arch = "wasm32"))]
    let embedding_bind_layout = embedding_pipeline.get_bind_group_layout(0);
    #[cfg(not(target_arch = "wasm32"))]
    let attention_bind_layout = attention_pipeline.get_bind_group_layout(0);
    #[cfg(not(target_arch = "wasm32"))]
    let elem_silu_mul_bind_layout = elem_silu_mul_pipeline.get_bind_group_layout(0);
    CommonPipelines {
        embedding_pipeline,
        attention_pipeline,
        elem_rms_norm_pipeline,
        elem_silu_mul_pipeline,
        elem_add_residual_pipeline,
        #[cfg(target_arch = "wasm32")]
        mc8_attn_bind_layout,
        #[cfg(target_arch = "wasm32")]
        mc8_elem_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        embedding_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        attention_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        elem_silu_mul_bind_layout,
    }
}
