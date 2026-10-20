pub struct Example {
    pipeline: etalon::RenderPipeline,
}
impl crate::framework::Example for Example {
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        let shader = device.create_shader_module(etalon::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_mesh_pipeline(&etalon::MeshPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            task: Some(etalon::TaskState {
                module: &shader,
                entry_point: Some("ts_main"),
                compilation_options: Default::default(),
            }),
            mesh: etalon::MeshState {
                module: &shader,
                entry_point: Some("ms_main"),
                compilation_options: Default::default(),
            },
            fragment: Some(etalon::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState {
                cull_mode: Some(etalon::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self { pipeline }
    }
    fn render(&mut self, view: &etalon::TextureView, device: &etalon::Device, queue: &etalon::Queue) {
        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&etalon::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(etalon::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: etalon::Operations {
                        load: etalon::LoadOp::Clear(etalon::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: etalon::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.push_debug_group("Prepare data for draw.");
            rpass.set_pipeline(&self.pipeline);
            rpass.pop_debug_group();
            rpass.insert_debug_marker("Draw!");
            rpass.draw_mesh_tasks(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
    fn required_downlevel_capabilities() -> etalon::DownlevelCapabilities {
        Default::default()
    }
    fn required_features() -> etalon::Features {
        etalon::Features::EXPERIMENTAL_MESH_SHADER | etalon::Features::PASSTHROUGH_SHADERS
    }
    fn required_limits() -> etalon::Limits {
        etalon::Limits::defaults().using_recommended_minimum_mesh_shader_values()
    }
    fn resize(
        &mut self,
        _config: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        // empty
    }
    fn update(&mut self, _event: winit::event::WindowEvent) {
        // empty
    }
}

pub fn main() {
    crate::framework::run::<Example>("mesh_shader");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "mesh_shader",
    image_path: "/examples/features/src/mesh_shader/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default()
        .features(etalon::Features::EXPERIMENTAL_MESH_SHADER | etalon::Features::PASSTHROUGH_SHADERS)
        .instance_flags(etalon::InstanceFlags::advanced_debugging())
        .limits(etalon::Limits::defaults().using_recommended_minimum_mesh_shader_values())
        .skip(etalon_test::FailureCase {
            backends: None,
            // Skip Mesa because LLVMPIPE has what is believed to be a driver bug
            vendor: Some(0x10005),
            adapter: None,
            driver: None,
            reasons: vec![],
            behavior: etalon_test::FailureBehavior::Ignore,
        }),
    comparisons: &[etalon_test::ComparisonType::Mean(0.005)],
    _phantom: std::marker::PhantomData::<Example>,
};
