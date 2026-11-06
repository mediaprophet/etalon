/// 112-byte uniform block for `CLUSTER_LIGHT_WGSL` (`16 x 9 x 24 = 3456` frustum clusters).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterGridParamsGpu {
    pub grid_dim: [u32; 4],
    pub view_size: [f32; 4],
    pub inverse_projection: [[f32; 4]; 4],
    pub light_count: u32,
    pub max_lights_per_cluster: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: [(); 112] = [(); std::mem::size_of::<ClusterGridParamsGpu>()];

/// Active GPU compute pipeline owner for 3D Clustered Light Culling (`CLUSTER_LIGHT_WGSL`).
pub struct ClusterLightPipelineGpu {
    pub build_aabbs_pipeline: etalon::ComputePipeline,
    pub cull_lights_pipeline: etalon::ComputePipeline,
    pub bind_group_layout: etalon::BindGroupLayout,
    pub params_buf: etalon::Buffer,
    pub cluster_aabbs_buf: etalon::Buffer,
    pub point_lights_buf: etalon::Buffer,
    pub cluster_light_indices_buf: etalon::Buffer,
    pub cluster_grid_buf: etalon::Buffer,
}

impl ClusterLightPipelineGpu {
    pub const TOTAL_CLUSTERS: u32 = 16 * 9 * 24;
    pub const MAX_LIGHTS_PER_CLUSTER: u32 = 64;

    pub fn new(device: &etalon::Device, max_lights: u32) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("qualia_cluster_light_wgsl"),
            source: etalon::ShaderSource::Wgsl(include_str!("../shaders/cluster_light.wgsl").into()),
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
            label: Some("qualia_cluster_light_bgl"),
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
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });

        let layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: Some("qualia_cluster_light_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let build_aabbs_pipeline =
            device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                label: Some("qualia_cluster_build_aabbs"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("cluster_build_aabbs"),
                compilation_options: Default::default(),
                cache: None,
            });

        let cull_lights_pipeline =
            device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                label: Some("qualia_cluster_cull_lights"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("cluster_cull_lights"),
                compilation_options: Default::default(),
                cache: None,
            });

        let params_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_cluster_params_buf"),
            size: std::mem::size_of::<ClusterGridParamsGpu>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cluster_aabbs_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_cluster_aabbs_buf"),
            size: u64::from(Self::TOTAL_CLUSTERS) * 32,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let point_lights_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_cluster_point_lights_buf"),
            size: u64::from(max_lights.max(16)) * 32,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cluster_light_indices_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_cluster_light_indices_buf"),
            size: u64::from(Self::TOTAL_CLUSTERS * Self::MAX_LIGHTS_PER_CLUSTER) * 4,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cluster_grid_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("qualia_cluster_grid_buf"),
            size: u64::from(Self::TOTAL_CLUSTERS) * 8,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            build_aabbs_pipeline,
            cull_lights_pipeline,
            bind_group_layout,
            params_buf,
            cluster_aabbs_buf,
            point_lights_buf,
            cluster_light_indices_buf,
            cluster_grid_buf,
        }
    }
}
