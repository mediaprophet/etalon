#[cfg(target_arch = "wasm32")]
const MC8_UNIFORM_ALIGN: usize = 256;

pub struct GemvPipelines {
    pub pipeline: wgpu::ComputePipeline,
    #[cfg(target_arch = "wasm32")]
    pub mmv_q8_0_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_residual_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_residual_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_mr_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_residual_mr_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_warp_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub coop_gemv_residual_warp_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub mock_pipeline: wgpu::ComputePipeline,
    #[cfg(target_arch = "wasm32")]
    pub mc8_gemm_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub pipeline_bind_layout: wgpu::BindGroupLayout,
}

pub(super) fn build(
    device: &wgpu::Device,
    native_pipeline_cache: &Option<wgpu::PipelineCache>,
    options: super::PipelineOptions,
) -> GemvPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Fused Transformer Shader"),
        source: wgpu::ShaderSource::Wgsl(
            #[cfg(target_arch = "wasm32")]
            super::shaders::WASM_FUSED_TRANSFORMER_WGSL.into(),
            #[cfg(not(target_arch = "wasm32"))]
            super::shaders::FUSED_TRANSFORMER_WGSL.into(),
        ),
    });

    // Shared explicit 5-slot layout (bindings 0-3 + residual 4). Module-scope
    // `residual` in fused_transformer.wgsl requires binding 4 on every entry point.
    // Plain GEMV call sites bind a dummy residual (often the input buffer).
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_residual_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CoopGemvBGL"),
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
            ],
        });
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_bind_layout = coop_gemv_residual_bind_layout.clone();
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CoopGemvPL"),
            bind_group_layouts: &[Some(&coop_gemv_bind_layout)],
            immediate_size: 0,
        });
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_residual_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CoopGemvResidualPL"),
            bind_group_layouts: &[Some(&coop_gemv_residual_bind_layout)],
            immediate_size: 0,
        });

    #[cfg(target_arch = "wasm32")]
    let mc8_gemm_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MC8GemmBGL"),
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
        ],
    });
    #[cfg(target_arch = "wasm32")]
    let mc8_gemm_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MC8GemmPL"),
        bind_group_layouts: &[Some(&mc8_gemm_bind_layout)],
        immediate_size: 0,
    });

    #[cfg(target_arch = "wasm32")]
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Fused Transformer Pipeline (WASM)"),
        layout: Some(&mc8_gemm_pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(target_arch = "wasm32")]
    let mmv_q8_0_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MulMatVec Q8_0 Pipeline (WASM)"),
        layout: Some(&mc8_gemm_pipeline_layout),
        module: &shader,
        entry_point: Some("mul_mat_vec_q8_0"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(not(target_arch = "wasm32"))]
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Fused Transformer Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    // 0.0.21: second pipeline over the SAME shader module — the cooperative one-workgroup-per-row
    // GEMV. Auto-layout; same group-0 bindings as `main`. Native only.
    //
    // When the adapter advertises SUBGROUP, build the wave-reduction variant (`coop_gemv_sg` in
    // coop_gemv_subgroup.wgsl, concatenated after the base with `enable subgroups;`) which
    // replaces the 8-step barrier-synced shared-memory tree reduction with one `subgroupAdd` per
    // subgroup. Identical group-0 bindings + (n_out,1,1) dispatch → a drop-in for this field, so
    // every call site and the derived `coop_gemv_bind_layout` pick it up transparently. Adapters
    // without subgroups (and wasm) keep the universal shared-memory `coop_gemv`.
    // All coop GEMV entry points share the explicit 5-slot CoopGemvBGL so bind
    // groups are interchangeable across single-row / multi-row / residual / warp
    // (no exclusive-pipeline auto-layout traps).
    #[cfg(not(target_arch = "wasm32"))]
    let (coop_gemv_pipeline, coop_gemv_residual_pipeline) = if options.subgroups {
        // Note: do NOT inject `enable subgroups;` — naga/wgpu 29 rejects it.
        let sg_src = format!(
            "{}\n{}",
            super::shaders::FUSED_TRANSFORMER_WGSL,
            super::shaders::COOP_GEMV_SUBGROUP_WGSL,
        );
        let sg_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Coop GEMV Subgroup Shader"),
            source: wgpu::ShaderSource::Wgsl(sg_src.into()),
        });
        let gemv = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coop GEMV SG Pipeline"),
            layout: Some(&coop_gemv_pipeline_layout),
            module: &sg_module,
            entry_point: Some("coop_gemv_sg"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        let resid = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coop GEMV Residual SG Pipeline"),
            layout: Some(&coop_gemv_residual_pipeline_layout),
            module: &sg_module,
            entry_point: Some("coop_gemv_residual_sg"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        (gemv, resid)
    } else {
        let gemv = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coop GEMV Pipeline"),
            layout: Some(&coop_gemv_pipeline_layout),
            module: &shader,
            entry_point: Some("coop_gemv"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        let resid = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coop GEMV Residual Pipeline"),
            layout: Some(&coop_gemv_residual_pipeline_layout),
            module: &shader,
            entry_point: Some("coop_gemv_residual"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        (gemv, resid)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_mr_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Coop GEMV Multi-Row Pipeline"),
        layout: Some(&coop_gemv_pipeline_layout),
        module: &shader,
        entry_point: Some("coop_gemv_mr"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_residual_mr_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coop GEMV Residual Multi-Row Pipeline"),
            layout: Some(&coop_gemv_residual_pipeline_layout),
            module: &shader,
            entry_point: Some("coop_gemv_residual_mr"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_warp_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coop GEMV Warp Pipeline"),
            layout: Some(&coop_gemv_pipeline_layout),
            module: &shader,
            entry_point: Some("coop_gemv_warp"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
    #[cfg(not(target_arch = "wasm32"))]
    let coop_gemv_residual_warp_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Coop GEMV Residual Warp Pipeline"),
            layout: Some(&coop_gemv_residual_pipeline_layout),
            module: &shader,
            entry_point: Some("coop_gemv_residual_warp"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });

    #[cfg(not(target_arch = "wasm32"))]
    let mock_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mock Fused Contraction Shader"),
        source: wgpu::ShaderSource::Wgsl(super::shaders::FUSED_TENSOR_CONTRACTION_WGSL.into()),
    });
    #[cfg(not(target_arch = "wasm32"))]
    let mock_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mock Fused Contraction Pipeline"),
        layout: None,
        module: &mock_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: native_pipeline_cache.as_ref(),
    });

    #[cfg(not(target_arch = "wasm32"))]
    let pipeline_bind_layout = pipeline.get_bind_group_layout(0);
    GemvPipelines {
        pipeline,
        #[cfg(target_arch = "wasm32")]
        mmv_q8_0_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_residual_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_residual_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_mr_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_residual_mr_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_warp_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        coop_gemv_residual_warp_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        mock_pipeline,
        #[cfg(target_arch = "wasm32")]
        mc8_gemm_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        pipeline_bind_layout,
    }
}
