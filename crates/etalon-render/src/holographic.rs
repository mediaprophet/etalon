//! Holographic pass for a caller-owned camera binding layout.

pub const HOLOGRAPHIC_WGSL: &str = concat!(
    include_str!("shaders/spectral.wgsl"),
    include_str!("shaders/holographic.wgsl")
);

/// WebGPU pipeline wrapper for the Holographic 3D Engine.
pub struct HolographicPipelineGpu {
    pub pipeline: etalon::RenderPipeline,
    pub node_storage_layout: etalon::BindGroupLayout,
}

impl HolographicPipelineGpu {
    /// Creates the holographic render pipeline and storage layout.
    pub fn new(
        device: &etalon::Device,
        camera_layout: &etalon::BindGroupLayout,
        format: etalon::TextureFormat,
    ) -> Self {
        let node_storage_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                label: Some("qualia-holographic-node-storage-layout"),
                entries: &[etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::VERTEX_FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia-holographic-wgsl"),
            source: etalon::ShaderSource::Wgsl(HOLOGRAPHIC_WGSL.into()),
        });

        let layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("qualia-holographic-pipeline-layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&node_storage_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("qualia-holographic-pipeline"),
            layout: Some(&layout),
            vertex: etalon::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(etalon::ColorTargetState {
                    format,
                    blend: Some(etalon::BlendState::ALPHA_BLENDING),
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(etalon::DepthStencilState {
                format: etalon::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(etalon::CompareFunction::LessEqual),
                stencil: etalon::StencilState::default(),
                bias: etalon::DepthBiasState::default(),
            }),
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            node_storage_layout,
        }
    }

    /// Creates an Etalon decoupled volumetric ray-marching pipeline.
    pub fn new_etalon_volumetric(
        device: &etalon::Device,
        color_format: etalon::TextureFormat,
        depth_format: Option<etalon::TextureFormat>,
    ) -> super::volumetric::VolumetricPipeline {
        super::volumetric::VolumetricPipeline::new(device, color_format, depth_format)
    }
}
