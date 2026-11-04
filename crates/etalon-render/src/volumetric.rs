//! Volumetric ray-marcher and holographic projection pass.
//!
//! Evaluates procedural Signed Distance Fields (SDFs), constructive wave interference,
//! and Beer-Lambert absorption directly along camera rays using instanced proxy bounding volumes.

use bytemuck::{Pod, Zeroable};

/// Raw WGSL shader source for the procedural holographic volumetric engine.
pub const HOLOGRAPHIC_WGSL: &str = include_str!("shaders/holographic_standalone.wgsl");

/// 96-byte std140/std430 aligned GPU representation of a semantic holographic entity.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct HolographicNodeGpu {
    /// World space position xyz, bounding sphere / proxy scale w
    pub position_scale: [f32; 4],
    /// Half extents xyz, primitive type w (0: Sphere, 1: RoundedBox, 2: Capsule, 3: Metaball, 4: InterferenceBridge)
    pub half_extents_primitive: [f32; 4],
    /// Base frequency x, amplitude y, phase z, manifold curvature w
    pub frequency_amplitude: [f32; 4],
    /// 10D Manifold parameters: epistemic_q x, topological_spin y, temporal_decay z, density_threshold w
    pub manifold_tensor: [f32; 4],
    /// Target endpoint xyz (for relational bridges), coherence weight w
    pub connection_endpoint: [f32; 4],
    /// Emissive tint rgba
    pub emissive_color: [f32; 4],
}

const _: () = {
    assert!(core::mem::size_of::<HolographicNodeGpu>() == 96);
    assert!(core::mem::align_of::<HolographicNodeGpu>() == 16);
};

impl Default for HolographicNodeGpu {
    fn default() -> Self {
        Self {
            position_scale: [0.0, 0.0, 0.0, 1.0],
            half_extents_primitive: [0.5, 0.5, 0.5, 0.0],
            frequency_amplitude: [1.0, 1.0, 0.0, 0.0],
            manifold_tensor: [1.0, 0.0, 0.0, 0.5],
            connection_endpoint: [0.0, 0.0, 0.0, 0.0],
            emissive_color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Camera uniform (128 bytes, std140-aligned).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct CameraUniform {
    pub view_projection: [[f32; 4]; 4],
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub tensor_mode: u32,
    pub _padding0: [f32; 4],
    pub _padding1: [f32; 4],
    pub _padding2: [f32; 4],
}

const _: () = {
    assert!(core::mem::size_of::<CameraUniform>() == 128);
    assert!(core::mem::align_of::<CameraUniform>() == 16);
};

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
            tensor_mode: 0,
            _padding0: [0.0; 4],
            _padding1: [0.0; 4],
            _padding2: [0.0; 4],
        }
    }
}

/// Observer standpoint uniform (128 bytes, std140-aligned).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct ObserverStandpointUniform {
    pub standpoint_hash: [u32; 2],
    pub session_nonce: [u32; 2],
    pub epistemic_q: f32,
    pub t_slice: f32,
    pub t_window: f32,
    pub deontic_lane: u32,
    pub standpoint_class: u32,
    pub fabric_gate: u32,
    pub _padding0: [f32; 2],
    pub _padding1: [f32; 4],
    pub _padding2: [f32; 4],
    pub _padding3: [f32; 4],
    pub _padding4: [f32; 4],
    pub _padding5: [f32; 4],
}

const _: () = {
    assert!(core::mem::size_of::<ObserverStandpointUniform>() == 128);
    assert!(core::mem::align_of::<ObserverStandpointUniform>() == 16);
};

impl Default for ObserverStandpointUniform {
    fn default() -> Self {
        Self {
            standpoint_hash: [0, 0],
            session_nonce: [0, 0],
            epistemic_q: 1.0,
            t_slice: 0.0,
            t_window: 1.0,
            deontic_lane: 0,
            standpoint_class: 0,
            fabric_gate: 0,
            _padding0: [0.0; 2],
            _padding1: [0.0; 4],
            _padding2: [0.0; 4],
            _padding3: [0.0; 4],
            _padding4: [0.0; 4],
            _padding5: [0.0; 4],
        }
    }
}

/// Complete GPU volumetric ray-marching pipeline.
pub struct VolumetricPipeline {
    pub camera_buf: etalon::Buffer,
    pub observer_buf: etalon::Buffer,
    pub uniforms_layout: etalon::BindGroupLayout,
    pub nodes_layout: etalon::BindGroupLayout,
    pub uniforms_bind_group: etalon::BindGroup,
    pub pipeline: etalon::RenderPipeline,
}

impl VolumetricPipeline {
    pub fn new(
        device: &etalon::Device,
        color_format: etalon::TextureFormat,
        depth_format: Option<etalon::TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-holographic-shader"),
            source: etalon::ShaderSource::Wgsl(HOLOGRAPHIC_WGSL.into()),
        });

        let camera_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-holographic-camera"),
            size: core::mem::size_of::<CameraUniform>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let observer_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-holographic-observer"),
            size: core::mem::size_of::<ObserverStandpointUniform>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-holographic-uniforms-layout"),
            entries: &[
                etalon::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: core::num::NonZeroU64::new(core::mem::size_of::<
                            CameraUniform,
                        >()
                            as u64),
                    },
                    count: None,
                },
                etalon::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                    ty: etalon::BindingType::Buffer {
                        ty: etalon::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: core::num::NonZeroU64::new(core::mem::size_of::<
                            ObserverStandpointUniform,
                        >()
                            as u64),
                    },
                    count: None,
                },
            ],
        });

        let nodes_layout = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: Some("etalon-holographic-nodes-layout"),
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::VERTEX | etalon::ShaderStages::FRAGMENT,
                ty: etalon::BindingType::Buffer {
                    ty: etalon::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: core::num::NonZeroU64::new(core::mem::size_of::<
                        HolographicNodeGpu,
                    >() as u64),
                },
                count: None,
            }],
        });

        let uniforms_bind_group = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-holographic-uniforms-bind-group"),
            layout: &uniforms_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: observer_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("etalon-holographic-pipeline-layout"),
            bind_group_layouts: &[Some(&uniforms_layout), Some(&nodes_layout)],
            immediate_size: 0,
        });

        let depth_stencil = depth_format.map(|format| etalon::DepthStencilState {
            format,
            depth_write_enabled: Some(false),
            depth_compare: Some(etalon::CompareFunction::LessEqual),
            stencil: etalon::StencilState::default(),
            bias: etalon::DepthBiasState::default(),
        });

        let pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: Some("etalon-holographic-render-pipeline"),
            layout: Some(&pipeline_layout),
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
                targets: &[Some(etalon::ColorTargetState {
                    format: color_format,
                    blend: Some(etalon::BlendState::ALPHA_BLENDING),
                    write_mask: etalon::ColorWrites::ALL,
                })],
            }),
            primitive: etalon::PrimitiveState {
                topology: etalon::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: etalon::FrontFace::Ccw,
                cull_mode: Some(etalon::Face::Back),
                unclipped_depth: false,
                polygon_mode: etalon::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            camera_buf,
            observer_buf,
            uniforms_layout,
            nodes_layout,
            uniforms_bind_group,
            pipeline,
        }
    }

    pub fn update_camera(&self, queue: &etalon::Queue, camera: &CameraUniform) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));
    }

    pub fn update_observer(&self, queue: &etalon::Queue, observer: &ObserverStandpointUniform) {
        queue.write_buffer(&self.observer_buf, 0, bytemuck::bytes_of(observer));
    }

    /// Draw instanced bounding cuboids for `node_count` entities.
    pub fn render_nodes<'a>(
        &'a self,
        rpass: &mut etalon::RenderPass<'a>,
        nodes_bind_group: &'a etalon::BindGroup,
        node_count: u32,
    ) {
        if node_count == 0 {
            return;
        }
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.uniforms_bind_group, &[]);
        rpass.set_bind_group(1, nodes_bind_group, &[]);
        // 36 indices = 12 triangles for unit cuboid proxy
        rpass.draw(0..36, 0..node_count);
    }
}
