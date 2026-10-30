// Flocking boids example with gpu compute update pass
// adapted from https://github.com/austinEng/webgpu-samples/blob/master/src/examples/computeBoids.ts

use etalon::util::DeviceExt;
use nanorand::{Rng, WyRand};

// number of boid particles to simulate

const NUM_PARTICLES: u32 = 1500;

// number of single-particle calculations (invocations) in each gpu work group

const PARTICLES_PER_GROUP: u32 = 64;

/// Example struct holds references to etalon resources and frame persistent data
struct Example {
    particle_bind_groups: Vec<etalon::BindGroup>,
    particle_buffers: Vec<etalon::Buffer>,
    vertices_buffer: etalon::Buffer,
    compute_pipeline: etalon::ComputePipeline,
    render_pipeline: etalon::RenderPipeline,
    work_group_count: u32,
    frame_num: usize,
}

impl crate::framework::Example for Example {
    fn required_limits() -> etalon::Limits {
        etalon::Limits::downlevel_defaults()
    }

    fn required_downlevel_capabilities() -> etalon::DownlevelCapabilities {
        etalon::DownlevelCapabilities {
            flags: etalon::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    /// constructs initial instance of Example struct
    fn init(
        config: &etalon::SurfaceConfiguration,
        _adapter: &etalon::Adapter,
        device: &etalon::Device,
        _queue: &etalon::Queue,
    ) -> Self {
        let compute_shader = device.create_shader_module(etalon::include_wgsl!("compute.wgsl"));
        let draw_shader = device.create_shader_module(etalon::include_wgsl!("draw.wgsl"));

        // buffer for simulation parameters uniform

        let sim_param_data = [
            0.04f32, // deltaT
            0.1,     // rule1Distance
            0.025,   // rule2Distance
            0.025,   // rule3Distance
            0.02,    // rule1Scale
            0.05,    // rule2Scale
            0.005,   // rule3Scale
        ]
        .to_vec();
        let sim_param_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Simulation Parameter Buffer"),
            contents: bytemuck::cast_slice(&sim_param_data),
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
        });

        // create compute bind layout group and compute pipeline layout

        let compute_bind_group_layout =
            device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                entries: &[
                    etalon::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: etalon::ShaderStages::COMPUTE,
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: etalon::BufferSize::new(
                                (sim_param_data.len() * size_of::<f32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    etalon::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: etalon::ShaderStages::COMPUTE,
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: etalon::BufferSize::new((NUM_PARTICLES * 16) as _),
                        },
                        count: None,
                    },
                    etalon::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: etalon::ShaderStages::COMPUTE,
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: etalon::BufferSize::new((NUM_PARTICLES * 16) as _),
                        },
                        count: None,
                    },
                ],
                label: None,
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });

        // create render pipeline with empty bind group layout

        let render_pipeline_layout =
            device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&etalon::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: etalon::VertexState {
                module: &draw_shader,
                entry_point: Some("main_vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(etalon::VertexBufferLayout {
                        array_stride: 4 * 4,
                        step_mode: etalon::VertexStepMode::Instance,
                        attributes: &etalon::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                    }),
                    Some(etalon::VertexBufferLayout {
                        array_stride: 2 * 4,
                        step_mode: etalon::VertexStepMode::Vertex,
                        attributes: &etalon::vertex_attr_array![2 => Float32x2],
                    }),
                ],
            },
            fragment: Some(etalon::FragmentState {
                module: &draw_shader,
                entry_point: Some("main_fs"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: etalon::PrimitiveState::default(),
            depth_stencil: None,
            multisample: etalon::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // create compute pipeline

        let compute_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // buffer for the three 2d triangle vertices of each instance

        let vertex_buffer_data = [-0.01f32, -0.02, 0.01, -0.02, 0.00, 0.02];
        let vertices_buffer = device.create_buffer_init(&etalon::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: etalon::BufferUsages::VERTEX | etalon::BufferUsages::COPY_DST,
        });

        // buffer for all particles data of type [(posx,posy,velx,vely),...]

        let mut initial_particle_data = vec![0.0f32; (4 * NUM_PARTICLES) as usize];
        let mut rng = WyRand::new_seed(42);
        let mut unif = || rng.generate::<f32>() * 2f32 - 1f32; // Generate a num (-1, 1)
        for particle_instance_chunk in initial_particle_data.chunks_mut(4) {
            particle_instance_chunk[0] = unif(); // posx
            particle_instance_chunk[1] = unif(); // posy
            particle_instance_chunk[2] = unif() * 0.1; // velx
            particle_instance_chunk[3] = unif() * 0.1; // vely
        }

        // creates two buffers of particle data each of size NUM_PARTICLES
        // the two buffers alternate as dst and src for each frame

        let mut particle_buffers = Vec::<etalon::Buffer>::new();
        let mut particle_bind_groups = Vec::<etalon::BindGroup>::new();
        for i in 0..2 {
            particle_buffers.push(
                device.create_buffer_init(&etalon::util::BufferInitDescriptor {
                    label: Some(&format!("Particle Buffer {i}")),
                    contents: bytemuck::cast_slice(&initial_particle_data),
                    usage: etalon::BufferUsages::VERTEX
                        | etalon::BufferUsages::STORAGE
                        | etalon::BufferUsages::COPY_DST,
                }),
            );
        }

        // create two bind groups, one for each buffer as the src
        // where the alternate buffer is used as the dst

        for i in 0..2 {
            particle_bind_groups.push(device.create_bind_group(&etalon::BindGroupDescriptor {
                layout: &compute_bind_group_layout,
                entries: &[
                    etalon::BindGroupEntry {
                        binding: 0,
                        resource: sim_param_buffer.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[i].as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },
                ],
                label: None,
            }));
        }

        // calculates number of work groups from PARTICLES_PER_GROUP constant
        let work_group_count = NUM_PARTICLES.div_ceil(PARTICLES_PER_GROUP);

        // returns Example struct and No encoder commands

        Example {
            particle_bind_groups,
            particle_buffers,
            vertices_buffer,
            compute_pipeline,
            render_pipeline,
            work_group_count,
            frame_num: 0,
        }
    }

    /// update is called for any WindowEvent not handled by the framework
    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    /// resize is called on WindowEvent::Resized events
    fn resize(
        &mut self,
        _sc_desc: &etalon::SurfaceConfiguration,
        _device: &etalon::Device,
        _queue: &etalon::Queue,
    ) {
        //empty
    }

    /// render is called each frame, dispatching compute groups proportional
    ///   a TriangleList draw call for all NUM_PARTICLES at 3 vertices each
    fn render(
        &mut self,
        view: &etalon::TextureView,
        device: &etalon::Device,
        queue: &etalon::Queue,
    ) {
        // create render pass descriptor and its color attachments
        let color_attachments = [Some(etalon::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: etalon::Operations {
                load: etalon::LoadOp::Clear(etalon::Color::BLACK),
                store: etalon::StoreOp::Store,
            },
        })];
        let render_pass_descriptor = etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        // get command encoder
        let mut command_encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        command_encoder.push_debug_group("compute boid movement");
        {
            // compute pass
            let mut cpass = command_encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.particle_bind_groups[self.frame_num % 2], &[]);
            cpass.dispatch_workgroups(self.work_group_count, 1, 1);
        }
        command_encoder.pop_debug_group();

        command_encoder.push_debug_group("render boids");
        {
            // render pass
            let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
            rpass.set_pipeline(&self.render_pipeline);
            // render dst particles
            rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
            // the three instance-local vertices
            rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
            rpass.draw(0..3, 0..NUM_PARTICLES);
        }
        command_encoder.pop_debug_group();

        // update frame count
        self.frame_num += 1;

        // done
        queue.submit(Some(command_encoder.finish()));
    }
}

/// run example
pub fn main() {
    crate::framework::run::<Example>("boids");
}

#[cfg(test)]
#[etalon_test::apply(etalon_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "boids",
    // Generated on 1080ti on Vk/Windows
    image_path: "/examples/features/src/boids/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: etalon::Features::default(),
    base_test_parameters: etalon_test::TestParameters::default()
        .downlevel_flags(etalon::DownlevelFlags::COMPUTE_SHADERS)
        .limits(etalon::Limits::downlevel_defaults()),
    comparisons: &[etalon_test::ComparisonType::Mean(0.005)],
    _phantom: std::marker::PhantomData::<Example>,
};
