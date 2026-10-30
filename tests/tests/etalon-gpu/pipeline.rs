use etalon_test::{
    apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE,
        RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX,
        NO_TARGETLESS_RENDER,
        DEPTH_BIAS_CLAMP,
    ]);
}

const INVALID_SHADER_DESC: etalon::ShaderModuleDescriptor = etalon::ShaderModuleDescriptor {
    label: Some("invalid shader"),
    source: etalon::ShaderSource::Wgsl(std::borrow::Cow::Borrowed("not valid wgsl")),
};

const TRIVIAL_COMPUTE_SHADER_DESC: etalon::ShaderModuleDescriptor =
    etalon::ShaderModuleDescriptor {
        label: Some("trivial compute shader"),
        source: etalon::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            "@compute @workgroup_size(1) fn main() {}",
        )),
    };

const TRIVIAL_VERTEX_SHADER_DESC: etalon::ShaderModuleDescriptor = etalon::ShaderModuleDescriptor {
    label: Some("trivial vertex shader"),
    source: etalon::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
        "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }",
    )),
};

const TRIVIAL_FRAGMENT_SHADER_DESC: etalon::ShaderModuleDescriptor =
    etalon::ShaderModuleDescriptor {
        label: Some("trivial fragment shader"),
        source: etalon::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(0); }",
        )),
    };

// Create an invalid shader and a compute pipeline that uses it
// with a default bindgroup layout, and then ask for that layout.
// Validation should fail, but etalon should not panic.
#[apply(gpu_test!)]
static COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(INVALID_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                                label: Some("compute pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: Some("doesn't exist"),
                                compilation_options: Default::default(),
                                cache: None,
                            });

                    // https://github.com/mediaprophet/etalon/issues/4167 this used to panic
                    pipeline.get_bind_group_layout(0);
                },
                Some("Shader 'invalid shader' parsing error"),
            );
        });

#[apply(gpu_test!)]
static COMPUTE_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .enable_noop(),
        )
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(TRIVIAL_COMPUTE_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                                label: Some("compute pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(u32::MAX);
                },
                Some("Bind group layout index 4294967295 is greater than the device's configured `max_bind_groups` limit"),
            );
        });

#[apply(gpu_test!)]
static RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_MODULE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let module = ctx.device.create_shader_module(INVALID_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_render_pipeline(&etalon::RenderPipelineDescriptor {
                                label: Some("render pipeline"),
                                layout: None,
                                vertex: etalon::VertexState {
                                    module: &module,
                                    entry_point: Some("doesn't exist"),
                                    compilation_options: Default::default(),
                                    buffers: &[],
                                },
                                primitive: Default::default(),
                                depth_stencil: None,
                                multisample: Default::default(),
                                fragment: None,
                                multiview_mask: None,
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(0);
                },
                Some("Shader 'invalid shader' parsing error"),
            );
        });

#[apply(gpu_test!)]
static RENDER_PIPELINE_DEFAULT_LAYOUT_BAD_BGL_INDEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .enable_noop(),
        )
        .run_sync(|ctx| {
            fail(
                &ctx.device,
                || {
                    let vs_module = ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC);
                    let fs_module = ctx
                        .device
                        .create_shader_module(TRIVIAL_FRAGMENT_SHADER_DESC);

                    let pipeline =
                        ctx.device
                            .create_render_pipeline(&etalon::RenderPipelineDescriptor {
                                label: Some("render pipeline"),
                                layout: None,
                                vertex: etalon::VertexState {
                                    module: &vs_module,
                                    entry_point: Some("main"),
                                    compilation_options: Default::default(),
                                    buffers: &[],
                                },
                                primitive: Default::default(),
                                depth_stencil: None,
                                multisample: Default::default(),
                                fragment: Some(etalon::FragmentState {
                                    module: &fs_module,
                                    entry_point: Some("main"),
                                    compilation_options: Default::default(),
                                    targets: &[Some(etalon::ColorTargetState {
                                        format: etalon::TextureFormat::Rgba8Unorm,
                                        blend: None,
                                        write_mask: etalon::ColorWrites::ALL,
                                    })],
                                }),
                                multiview_mask: None,
                                cache: None,
                            });

                    pipeline.get_bind_group_layout(u32::MAX);
                },
                Some("Bind group layout index 4294967295 is greater than the device's configured `max_bind_groups` limit"),
            );
        });

#[apply(gpu_test!)]
static NO_TARGETLESS_RENDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        fail(
            &ctx.device,
            || {
                // Testing multisampling is important, because some backends don't behave well if one
                // tries to compile code in an unsupported multisample count. Failing to validate here
                // has historically resulted in requesting the back end to compile code.
                for power_of_two in [1, 2, 4, 8, 16, 32, 64] {
                    let _ = ctx
                        .device
                        .create_render_pipeline(&etalon::RenderPipelineDescriptor {
                            label: None,
                            layout: None,
                            vertex: etalon::VertexState {
                                module: &ctx
                                    .device
                                    .create_shader_module(TRIVIAL_VERTEX_SHADER_DESC),
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                buffers: &[],
                            },
                            primitive: Default::default(),
                            depth_stencil: None,
                            multisample: etalon::MultisampleState {
                                count: power_of_two,
                                ..Default::default()
                            },
                            fragment: None,
                            multiview_mask: None,
                            cache: None,
                        });
                }
            },
            Some(concat!(
                "At least one color attachment or depth-stencil attachment was expected, ",
                "but no render target for the pipeline was specified."
            )),
        )
    });

// A non-zero depth bias clamp needs Vulkan's `depthBiasClamp` feature enabled on the device.
#[apply(gpu_test!)]
static DEPTH_BIAS_CLAMP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().downlevel_flags(etalon::DownlevelFlags::DEPTH_BIAS_CLAMP))
    .run_sync(|ctx| {
        let _ = ctx
            .device
            .create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: etalon::VertexState {
                    module: &ctx.device.create_shader_module(TRIVIAL_VERTEX_SHADER_DESC),
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: Some(etalon::DepthStencilState {
                    format: etalon::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(etalon::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: etalon::DepthBiasState {
                        constant: 1,
                        slope_scale: 1.0,
                        clamp: 0.01,
                    },
                }),
                multisample: Default::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            });
    });
