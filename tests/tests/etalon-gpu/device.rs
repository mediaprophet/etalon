use std::sync::atomic::AtomicBool;

use etalon::InstanceFlags;
use etalon_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        CROSS_DEVICE_BIND_GROUP_USAGE,
        DEVICE_DESTROY_THEN_MORE,
        DEVICE_DESTROY_THEN_LOST,
        DIFFERENT_BGL_ORDER_BW_SHADER_AND_API,
        DEVICE_DESTROY_THEN_BUFFER_CLEANUP,
        DEVICE_AND_QUEUE_HAVE_DIFFERENT_IDS,
        REQUEST_DEVICE_ERROR_MESSAGE_NATIVE,
    ]);

    #[cfg(not(wasm_test))]
    {
        vec.push(MULTIPLE_DEVICES);
    }
}

#[apply(gpu_test!)]
static CROSS_DEVICE_BIND_GROUP_USAGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .expect_fail(FailureCase::always())
            .enable_noop(),
    )
    .run_async(|ctx| async move {
        // Create a bind group using a layout from another device. This should be a validation
        // error but currently crashes.
        let (device2, _) =
            pollster::block_on(ctx.adapter.request_device(&Default::default())).unwrap();

        {
            let bind_group_layout =
                device2.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[],
                });

            let _bind_group = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[],
            });
        }

        ctx.async_poll(etalon::PollType::Poll).await.unwrap();
    });

#[apply(gpu_test!)]
static SINGLE_DEVICES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(InstanceFlags::STRICT_WEBGPU_COMPLIANCE)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        use pollster::FutureExt as _;
        ctx.adapter
            .request_device(&etalon::DeviceDescriptor {
                required_features: etalon::Features::empty(),
                required_limits: etalon::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            })
            .block_on()
            .expect("failed to create device");
        assert!(ctx
            .adapter
            .request_device(&etalon::DeviceDescriptor {
                required_features: etalon::Features::empty(),
                required_limits: etalon::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            })
            .block_on()
            .is_err());
    });

#[cfg(not(wasm_test))]
#[apply(gpu_test!)]
static MULTIPLE_DEVICES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        use pollster::FutureExt as _;
        ctx.adapter
            .request_device(&etalon::DeviceDescriptor {
                required_features: etalon::Features::empty(),
                required_limits: etalon::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            })
            .block_on()
            .expect("failed to create device");
        ctx.adapter
            .request_device(&etalon::DeviceDescriptor {
                required_features: etalon::Features::empty(),
                required_limits: etalon::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            })
            .block_on()
            .expect("failed to create device");
    });

#[apply(gpu_test!)]
static REQUEST_DEVICE_ERROR_MESSAGE_NATIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters({
        let default = TestParameters::default();

        // On CI, we have the vulkan SDK installed and this test initializes the Vulkan backend when all
        // normal tests do not, so we get these weird error messages. This only actually gets hooked up
        // on Windows, because that's the only platform where the actual runtime gets hooked up and there
        // are no drivers.
        if std::env::var("ETALON_CI").is_ok() && cfg!(windows) {
            default.expect_fail(
                FailureCase::always()
                    .validation_error("Registry lookup failed to get ICD manifest files.  Possibly missing Vulkan driver?")
                    .validation_error("vkCreateInstance: Found no drivers!")
            )
        } else {
            default
        }
    })
    .run_async(|_ctx| request_device_error_message());

/// Check that `RequestDeviceError`s produced have some diagnostic information.
async fn request_device_error_message() {
    // Not using initialize_test() because that doesn't let us catch the error
    // nor .await anything
    let (_instance, adapter, _surface_guard) =
        etalon_test::initialize_adapter(None, &TestParameters::default())
            .await
            .unwrap();

    let device_error = adapter
        .request_device(&etalon::DeviceDescriptor {
            // Force a failure by requesting absurd limits.
            required_features: etalon::Features::all(),
            required_limits: etalon::Limits {
                max_texture_dimension_1d: u32::MAX,
                max_texture_dimension_2d: u32::MAX,
                max_texture_dimension_3d: u32::MAX,
                max_bind_groups: u32::MAX,
                max_immediate_size: u32::MAX,
                ..Default::default()
            },
            ..Default::default()
        })
        .await
        .unwrap_err();

    let device_error = device_error.to_string();
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "wasm32", not(feature = "webgl")))] {
            // On WebGPU, so the error we get will be from the browser WebGPU API.
            // Per the WebGPU specification this should be a `TypeError` when features are not
            // available, <https://gpuweb.github.io/gpuweb/#dom-gpuadapter-requestdevice>,
            // and the stringification it goes through for Rust should put that in the message.
            let expected = "TypeError";
        } else {
            // This message appears whenever etalon-core is used as the implementation.
            let expected = "Unsupported features were requested:";
        }
    }
    assert!(device_error.contains(expected), "{device_error}");
}

// This is a test of device behavior after `device.destroy()`. Specifically, all operations
// should turn into no-ops, per spec.
#[apply(gpu_test!)]
static DEVICE_DESTROY_THEN_MORE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::CLEAR_TEXTURE)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        // Create some resources on the device that we will attempt to use *after* losing
        // the device.

        // Create some 512 x 512 2D textures.
        let texture_extent = etalon::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };
        let texture_for_view = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 2,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rg8Uint,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = texture_for_view.create_view(&etalon::TextureViewDescriptor::default());

        let texture_for_read = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 2,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rg8Uint,
            usage: etalon::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let texture_for_write = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 2,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rg8Uint,
            usage: etalon::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create some buffers.
        let buffer_source = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 256,
            usage: etalon::BufferUsages::MAP_WRITE | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_dest = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 256,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_for_map = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 256,
            usage: etalon::BufferUsages::MAP_WRITE | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_for_unmap = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 256,
            usage: etalon::BufferUsages::MAP_WRITE | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        // Create a shader module.
        let shader_module = ctx
            .device
            .create_shader_module(etalon::ShaderModuleDescriptor {
                label: None,
                source: etalon::ShaderSource::Wgsl(std::borrow::Cow::Borrowed("")),
            });

        // Create some command encoders.
        let mut encoder_for_clear = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        let mut encoder_for_compute_pass = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        let mut encoder_for_render_pass = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        let mut encoder_for_buffer_buffer_copy = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        let mut encoder_for_buffer_texture_copy = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        let mut encoder_for_texture_buffer_copy = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        let mut encoder_for_texture_texture_copy = ctx
            .device
            .create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        // Destroy the device. This will cause all further operations to become no-ops.
        ctx.device.destroy();

        // The following operations should fail if the device was not lost.
        // Since the device is lost, we check that they succeed.

        let _ = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: ctx.device.limits().max_buffer_size * 2,
            usage: etalon::BufferUsages::MAP_WRITE | etalon::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let _ = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: ctx.device.limits().max_texture_dimension_2d * 2,
                height: ctx.device.limits().max_texture_dimension_2d * 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 2,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rg8Uint,
            usage: etalon::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        encoder_for_clear.clear_texture(
            &texture_for_write,
            &etalon::ImageSubresourceRange {
                aspect: etalon::TextureAspect::StencilOnly, // texture has no stencil component
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            },
        );
        ctx.queue.submit([encoder_for_clear.finish()]);

        let query_set = ctx.device.create_query_set(&etalon::QuerySetDescriptor {
            label: None,
            ty: etalon::QueryType::Occlusion,
            count: u32::MAX, // can be at most 4096
        });

        let pass = encoder_for_compute_pass.begin_compute_pass(&etalon::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(etalon::ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: None,
            }),
        });
        drop(pass);
        ctx.queue.submit([encoder_for_compute_pass.finish()]);

        let pass = encoder_for_render_pass.begin_render_pass(&etalon::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(etalon::RenderPassColorAttachment {
                ops: etalon::Operations::default(),
                resolve_target: None,
                view: &target_view,
                depth_slice: Some(u32::MAX), // should not be provided for 2D views
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(pass);
        ctx.queue.submit([encoder_for_render_pass.finish()]);

        encoder_for_buffer_buffer_copy.copy_buffer_to_buffer(
            &buffer_source,
            0,
            &buffer_dest,
            0,
            u64::MAX, // out of bounds for both buffers
        );
        ctx.queue.submit([encoder_for_buffer_buffer_copy.finish()]);

        encoder_for_buffer_texture_copy.copy_buffer_to_texture(
            etalon::TexelCopyBufferInfo {
                buffer: &buffer_source,
                layout: etalon::TexelCopyBufferLayout {
                    offset: u64::MAX, // out of bounds for buffer
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
            },
            texture_for_write.as_image_copy(),
            texture_extent,
        );
        ctx.queue.submit([encoder_for_buffer_texture_copy.finish()]);

        encoder_for_texture_buffer_copy.copy_texture_to_buffer(
            texture_for_read.as_image_copy(),
            etalon::TexelCopyBufferInfo {
                buffer: &buffer_source,
                layout: etalon::TexelCopyBufferLayout {
                    offset: u64::MAX, // out of bounds for buffer
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
            },
            texture_extent,
        );
        ctx.queue.submit([encoder_for_texture_buffer_copy.finish()]);

        encoder_for_texture_texture_copy.copy_texture_to_texture(
            texture_for_read.as_image_copy(),
            texture_for_write.as_image_copy(),
            etalon::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: u32::MAX, // out of bounds for both textures
            },
        );
        ctx.queue
            .submit([encoder_for_texture_texture_copy.finish()]);

        let invalid_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[etalon::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: etalon::ShaderStages::all(),
                        ty: etalon::BindingType::Buffer {
                            ty: etalon::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(
                                ctx.device.limits().max_uniform_buffer_binding_size * 2,
                            ),
                        },
                        count: None,
                    }],
                });

        let _ = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &invalid_bind_group_layout,
            entries: &[etalon::BindGroupEntry {
                binding: 0,
                resource: etalon::BindingResource::Buffer(buffer_source.as_entire_buffer_binding()),
            }],
        });

        let invalid_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&invalid_bind_group_layout)],
                    immediate_size: 0,
                });

        let _ = ctx
            .device
            .create_shader_module(etalon::ShaderModuleDescriptor {
                label: None,
                source: etalon::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    "not a valid shader source",
                )),
            });

        let _ = ctx
            .device
            .create_render_pipeline(&etalon::RenderPipelineDescriptor {
                label: None,
                layout: Some(&invalid_pipeline_layout),
                vertex: etalon::VertexState {
                    module: &shader_module,
                    entry_point: Some(""),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: etalon::PrimitiveState::default(),
                depth_stencil: None,
                multisample: etalon::MultisampleState::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            });

        let _ = ctx
            .device
            .create_compute_pipeline(&etalon::ComputePipelineDescriptor {
                label: None,
                layout: Some(&invalid_pipeline_layout),
                module: &shader_module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });

        buffer_for_map
            .slice(..)
            .map_async(etalon::MapMode::Write, |_| ());

        buffer_for_unmap.unmap();
    });

#[apply(gpu_test!)]
static DEVICE_DESTROY_THEN_LOST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        // This test checks that when device.destroy is called, the provided
        // DeviceLostClosure is called with reason DeviceLostReason::Destroyed.
        static WAS_CALLED: AtomicBool = AtomicBool::new(false);

        // Set a LoseDeviceCallback on the device.
        let callback = Box::new(|reason, _m| {
            WAS_CALLED.store(true, std::sync::atomic::Ordering::SeqCst);
            assert!(
                matches!(reason, etalon::DeviceLostReason::Destroyed),
                "Device lost info reason should match DeviceLostReason::Destroyed."
            );
        });
        ctx.device.set_device_lost_callback(callback);

        // Destroy the device.
        ctx.device.destroy();

        // Make sure the device queues are empty, which ensures that the closure
        // has been called.
        assert!(ctx
            .async_poll(etalon::PollType::wait_indefinitely())
            .await
            .unwrap()
            .is_queue_empty());

        assert!(
            WAS_CALLED.load(std::sync::atomic::Ordering::SeqCst),
            "Device lost callback should have been called."
        );
    });

#[apply(gpu_test!)]
static DIFFERENT_BGL_ORDER_BW_SHADER_AND_API: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        // This test addresses a bug found in multiple backends where `etalon_core` and `etalon_hal`
        // backends made different assumptions about the element order of vectors of bind group
        // layout entries and bind group resource bindings.
        //
        // Said bug was exposed originally by:
        //
        // 1. Shader-declared bindings having a different order than resource bindings provided to
        //    `Device::create_bind_group`.
        // 2. Having more of one type of resource in the bind group than another.
        //
        // …such that internals would accidentally attempt to use an out-of-bounds index (of one
        // resource type) in the wrong list of a different resource type. Let's reproduce that
        // here.

        let trivial_shaders_with_some_reversed_bindings = concat!(
            "@group(0) @binding(3) var myTexture2: texture_2d<f32>;\n",
            "@group(0) @binding(2) var myTexture1: texture_2d<f32>;\n",
            "@group(0) @binding(1) var mySampler: sampler;\n",
            "\n",
            "@fragment\n",
            "fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4f {\n",
            "  return textureSample(myTexture1, mySampler, pos.xy) \n",
            "    + textureSample(myTexture2, mySampler, pos.xy);\n",
            "}\n",
            "\n",
            "@vertex\n",
            "fn vs_main() -> @builtin(position) vec4<f32> {\n",
            "  return vec4<f32>(0.0, 0.0, 0.0, 1.0);\n",
            "}\n",
        );

        let trivial_shaders_with_some_reversed_bindings =
            ctx.device
                .create_shader_module(etalon::ShaderModuleDescriptor {
                    label: None,
                    source: etalon::ShaderSource::Wgsl(
                        trivial_shaders_with_some_reversed_bindings.into(),
                    ),
                });

        let my_texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 1024,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rgba8Unorm,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT | etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let my_texture_view = my_texture.create_view(&etalon::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            aspect: etalon::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            swizzle: etalon::TextureComponentSwizzle::default(),
        });

        let my_sampler = ctx
            .device
            .create_sampler(&etalon::SamplerDescriptor::default());

        let render_pipeline = ctx
            .device
            .create_render_pipeline(&etalon::RenderPipelineDescriptor {
                fragment: Some(etalon::FragmentState {
                    module: &trivial_shaders_with_some_reversed_bindings,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(etalon::ColorTargetState {
                        format: etalon::TextureFormat::Bgra8Unorm,
                        blend: None,
                        write_mask: etalon::ColorWrites::ALL,
                    })],
                }),
                layout: None,

                // Other fields below aren't interesting for this text.
                label: None,
                vertex: etalon::VertexState {
                    module: &trivial_shaders_with_some_reversed_bindings,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: etalon::PrimitiveState::default(),
                depth_stencil: None,
                multisample: etalon::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // fail(&ctx.device, || {
        // }, "");
        let _ = ctx.device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: etalon::BindingResource::Sampler(&my_sampler),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: etalon::BindingResource::TextureView(&my_texture_view),
                },
                etalon::BindGroupEntry {
                    binding: 3,
                    resource: etalon::BindingResource::TextureView(&my_texture_view),
                },
            ],
        });
    });

#[apply(gpu_test!)]
static DEVICE_DESTROY_THEN_BUFFER_CLEANUP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        // When a device is destroyed, its resources should be released,
        // without causing a deadlock.

        // Create a buffer to be left around until the device is destroyed.
        let _buffer = ctx.device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 256,
            usage: etalon::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create a texture to be left around until the device is destroyed.
        let texture_extent = etalon::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };
        let _texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 2,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Rg8Uint,
            usage: etalon::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Destroy the device.
        ctx.device.destroy();

        // Poll the device, which should try to clean up its resources.
        ctx.instance.poll_all(true);
    });

#[apply(gpu_test!)]
static DEVICE_AND_QUEUE_HAVE_DIFFERENT_IDS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let TestingContext {
            adapter,
            device_features,
            device_limits,
            device,
            queue,
            ..
        } = ctx;

        drop(device);

        let (device2, queue2) =
            etalon_test::initialize_device(&adapter, device_features, device_limits).await;

        drop(queue);
        drop(device2);
        drop(queue2); // this would previously panic since we would try to use the Device ID to drop the Queue
    });
