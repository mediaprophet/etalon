#[cfg(not(target_arch = "wasm32"))]
use crate::ordinary::gpu_params::GemmGpuParams;
#[cfg(target_arch = "wasm32")]
const MC8_UNIFORM_ALIGN: usize = 256;

pub struct FfnPipelines {
    #[cfg(target_arch = "wasm32")]
    pub mc8_ffn_fused_bind_layout: wgpu::BindGroupLayout,
    #[cfg(target_arch = "wasm32")]
    pub mc8_ffn_fused_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub ffn_fused_bind_layout: wgpu::BindGroupLayout,
    #[cfg(not(target_arch = "wasm32"))]
    pub ffn_fused_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub ffn_fused_coop_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub ffn_fused_mr_pipeline: wgpu::ComputePipeline,
    #[cfg(not(target_arch = "wasm32"))]
    pub ffn_fused_warp_pipeline: wgpu::ComputePipeline,
}

pub(super) fn build(
    device: &wgpu::Device,
    native_pipeline_cache: &Option<wgpu::PipelineCache>,
    #[allow(unused_variables)]
    options: super::PipelineOptions,
) -> FfnPipelines {
    #[cfg(target_arch = "wasm32")]
    let mc8_ffn_fused_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MC8FfnFusedBGL"),
            entries: &[
                // 0: ffn_input (normalized hidden, storage read)
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
                // 1: gate_words (quantized gate weight, storage read)
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
                // 2: up_words (quantized up weight, storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: params (GemmParams, dynamic uniform offset — gate's staged params)
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
                // 4: ffn_output (silu(gate)·up intermediate, storage read_write)
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
            ],
        });
    #[cfg(target_arch = "wasm32")]
    let mc8_ffn_fused_pipeline = {
        // Modular WGSL: shared scaffold + per-role dequant instances composed at runtime.
        let tpl = super::shaders::WASM_DEQUANT_TEMPLATE_WGSL;
        let gate_fns = tpl.replace("$W", "gate_words").replace("$S", "_gate");
        let up_fns = tpl.replace("$W", "up_words").replace("$S", "_up");
        let base = super::shaders::WASM_FUSED_FFN_WGSL;
        // Inject the per-role dequant math at the marker (between shared helpers and
        // the entry point) so declarations precede their uses.
        // WASM copy already has subgroup entry points stripped — no runtime filtering needed.
        let src = base.replace("// @@DEQUANT_FUNCTIONS@@", &format!("{gate_fns}\n{up_fns}"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FusedFFNExpansion"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC8FfnFusedPL"),
            bind_group_layouts: &[Some(&mc8_ffn_fused_bind_layout)],
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FusedFFNExpansionPipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("fused_ffn_expansion"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        })
    };

    // Native T-A1 — same fused_ffn.wgsl, static uniform (no dynamic offset).
    // Wired into resident_decode mega-pass when gate/up share a supported quant type.
    #[cfg(not(target_arch = "wasm32"))]
    let ffn_fused_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("NativeFfnFusedBGL"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<GemmGpuParams>() as u64,
                    ),
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
        ],
    });
    #[cfg(not(target_arch = "wasm32"))]
    let (
        ffn_fused_pipeline,
        ffn_fused_coop_pipeline,
        ffn_fused_mr_pipeline,
        ffn_fused_warp_pipeline,
    ) = {
        let tpl = super::shaders::DEQUANT_TEMPLATE_WGSL;
        let gate_fns = tpl.replace("$W", "gate_words").replace("$S", "_gate");
        let up_fns = tpl.replace("$W", "up_words").replace("$S", "_up");
        let base = super::shaders::FUSED_FFN_WGSL;
        let src = base.replace("// @@DEQUANT_FUNCTIONS@@", &format!("{gate_fns}\n{up_fns}"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NativeFusedFFNExpansion"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NativeFfnFusedPL"),
            bind_group_layouts: &[Some(&ffn_fused_bind_layout)],
            immediate_size: 0,
        });
        let naive = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NativeFusedFFNExpansionPipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("fused_ffn_expansion"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        // Prefer subgroup reduction when available (same as coop_gemv_sg).
        let coop_ep = if options.subgroups {
            "coop_fused_ffn_sg"
        } else {
            "coop_fused_ffn_expansion"
        };
        let coop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NativeCoopFusedFFNExpansionPipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(coop_ep),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        // Multi-row fused FFN (4 rows/WG, one K-sweep) — Q4_K_SOA 3B lever.
        let coop_mr = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NativeCoopFusedFFNMultiRowPipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("coop_fused_ffn_mr"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        // Warp fused FFN (32 thr/row, 8 cols/lane).
        let coop_warp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NativeCoopFusedFFNWarpPipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("coop_fused_ffn_warp"),
            compilation_options: Default::default(),
            cache: native_pipeline_cache.as_ref(),
        });
        (naive, coop, coop_mr, coop_warp)
    };

    FfnPipelines {
        #[cfg(target_arch = "wasm32")]
        mc8_ffn_fused_bind_layout,
        #[cfg(target_arch = "wasm32")]
        mc8_ffn_fused_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        ffn_fused_bind_layout,
        #[cfg(not(target_arch = "wasm32"))]
        ffn_fused_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        ffn_fused_coop_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        ffn_fused_mr_pipeline,
        #[cfg(not(target_arch = "wasm32"))]
        ffn_fused_warp_pipeline,
    }
}
