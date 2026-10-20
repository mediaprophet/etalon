//! This renders to the screen with compute shaders. Note that due to limitations in Firefox,
//! the wait will cause FPS to be capped at 10 when running on webgpu on Firefox. It is
//! therefore not recommended to use this code, at least until
//! <https://bugzilla.mozilla.org/show_bug.cgi?id=1870699> (and possibly further work) is resolved.

use web_time::Instant;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug)]
#[repr(C)]
#[repr(align(16))]
struct GlobalParams {
    time: f32,
    frame: f32,
    _padding: [u8; 8],
}

pub struct Example {
    pipeline: etalon::ComputePipeline,
    texture_view: etalon::TextureView,
    global_params: etalon::Buffer,
    bg: etalon::BindGroup,
    bgl: etalon::BindGroupLayout,
    blitter: etalon::util::TextureBlitter,
    frame_count: u32,
    start_time: Option<Instant>,
}
impl crate::framework::Example for Example {
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        let sm = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));
        let bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: None,
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
                    ty: etalon::BindingType::StorageTexture {
                        access: etalon::StorageTextureAccess::WriteOnly,
                        format: etalon::TextureFormat::Rgba8Unorm,
                        view_dimension: etalon::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let ppl = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: None,
            layout: Some(&ppl),
            module: &sm,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
        let blitter = etalon::util::TextureBlitter::new(device, config.view_formats[0]);
        let global_params = device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: size_of::<GlobalParams>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (texture_view, bg) =
            create_tv_and_bg(device, &bgl, &global_params, config.width, config.height);
        Self {
            pipeline,
            texture_view,
            global_params,
            bg,
            bgl,
            blitter,
            frame_count: 0,
            start_time: None,
        }
    }

    fn required_limits() -> etalon::Limits {
        etalon::Limits {
            max_storage_textures_per_shader_stage: 1,
            ..Default::default()
        }
    }

    fn resize(
        &mut self,
        config: &etalon::SurfaceConfiguration,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        let (texture_view, bg) = create_tv_and_bg(
            device,
            &self.bgl,
            &self.global_params,
            config.width,
            config.height,
        );
        self.bg = bg;
        self.texture_view = texture_view;
    }

    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let now = Instant::now();
        let start = *self.start_time.get_or_insert(now);
        let time_since_start = (now - start).as_secs_f32();
        queue.write_buffer(
            &self.global_params,
            0,
            bytemuck::bytes_of(&GlobalParams {
                time: time_since_start,
                frame: self.frame_count as f32,
                _padding: [0; 8],
            }),
        );
        let mut encoder = device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg, &[]);
            const SHADER_WORKGROUP_DIM: u32 = 16;
            pass.dispatch_workgroups(
                self.texture_view
                    .texture()
                    .width()
                    .div_ceil(SHADER_WORKGROUP_DIM),
                self.texture_view
                    .texture()
                    .height()
                    .div_ceil(SHADER_WORKGROUP_DIM),
                1,
            );
        }
        self.blitter
            .copy(device, &mut encoder, &self.texture_view, view);

        queue.submit([encoder.finish()]);
        device.poll(etalon::PollType::wait_indefinitely()).unwrap();

        self.frame_count += 1;
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}
}

fn create_tv_and_bg(
    device: &etalon::Device,
    bgl: &etalon::BindGroupLayout,
    global_params: &etalon::Buffer,
    width: u32,
    height: u32,
) -> (etalon::TextureView, etalon::BindGroup) {
    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Rgba8Unorm,
        usage: etalon::TextureUsages::STORAGE_BINDING | etalon::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let bg = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &[
            etalon::BindGroupEntry {
                binding: 0,
                resource: etalon::BindingResource::Buffer(global_params.as_entire_buffer_binding()),
            },
            etalon::BindGroupEntry {
                binding: 1,
                resource: etalon::BindingResource::TextureView(&view),
            },
        ],
    });
    (view, bg)
}

pub fn main() {
    crate::framework::run::<Example>("render-with-compute");
}
