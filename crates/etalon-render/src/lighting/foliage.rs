use etalon as wgpu;
/// Active GPU compute pipeline owner for Continuous Multi-Scale Foliage LOD (`FOLIAGE_LOD_WGSL`).
pub struct FoliageLodPipelineGpu {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl FoliageLodPipelineGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qualia_foliage_lod_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/foliage_lod.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qualia_foliage_lod_bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qualia_foliage_lod_pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qualia_foliage_lod_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("foliage_cull_and_generate"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// 64-byte Inherent Optical Properties (IOP) descriptor mirroring `DielectricVolumeIop` in `HYDROLOGIC_OPTICS_WGSL`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DielectricVolumeIopGpu {
    pub ior_spectrum: [f32; 3],
    pub _pad0: f32,
    pub absorption_coeff: [f32; 3],
    pub _pad1: f32,
    pub rayleigh_scattering: [f32; 3],
    pub _pad2: f32,
    pub mie_scattering: [f32; 4],
}

const _: [(); 64] = [(); std::mem::size_of::<DielectricVolumeIopGpu>()];

impl DielectricVolumeIopGpu {
    /// Evaluates Beer-Lambert transmittance $T(d) = \exp(-(a + b_r + b_m) d)$ with zero heap allocations.
    #[inline]
    pub fn beer_lambert_transmittance(&self, distance_m: f32) -> [f32; 3] {
        let d = distance_m.max(0.0);
        [
            (-((self.absorption_coeff[0] + self.rayleigh_scattering[0] + self.mie_scattering[0])
                * d))
                .exp(),
            (-((self.absorption_coeff[1] + self.rayleigh_scattering[1] + self.mie_scattering[1])
                * d))
                .exp(),
            (-((self.absorption_coeff[2] + self.rayleigh_scattering[2] + self.mie_scattering[2])
                * d))
                .exp(),
        ]
    }
}
