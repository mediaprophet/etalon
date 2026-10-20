//! Tests for BGRA8UNORM_STORAGE feature

use std::borrow::Cow;

use etalon_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(BGRA8_UNORM_STORAGE);
}

const SHADER_SRC: &str = "
@group(0) @binding(0) var tex: texture_storage_2d<bgra8unorm, write>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>) {
    var texel = vec4f(0.0, 0.0, 1.0, 1.0);
    textureStore(tex, wgid.xy, texel);
}
";

#[apply(gpu_test!)]
static BGRA8_UNORM_STORAGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(etalon::Limits {
                max_storage_textures_per_shader_stage: 1,
                ..Default::default()
            })
            .features(etalon::Features::BGRA8UNORM_STORAGE),
    )
    .run_async(|ctx| async move {
        let device = &ctx.device;
        let texture = ctx.device.create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format: etalon::TextureFormat::Bgra8Unorm,
            usage: etalon::TextureUsages::STORAGE_BINDING | etalon::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&etalon::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            aspect: etalon::TextureAspect::All,
            base_mip_level: 0,
            base_array_layer: 0,
            mip_level_count: Some(1),
            array_layer_count: Some(1),
            swizzle: etalon::TextureComponentSwizzle::default(),
        });

        let readback_buffer = device.create_buffer(&etalon::BufferDescriptor {
            label: None,
            size: 256 * 256 * 4,
            usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
            label: None,
            entries: &[etalon::BindGroupLayoutEntry {
                binding: 0,
                visibility: etalon::ShaderStages::COMPUTE,
                ty: etalon::BindingType::StorageTexture {
                    access: etalon::StorageTextureAccess::WriteOnly,
                    format: etalon::TextureFormat::Bgra8Unorm,
                    view_dimension: etalon::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let bg = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[etalon::BindGroupEntry {
                binding: 0,
                resource: etalon::BindingResource::TextureView(&view),
            }],
        });

        let pl = device.create_pipeline_layout(&etalon::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let module = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: None,
            source: etalon::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SRC)),
        });

        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            module: &module,
            cache: None,
        });

        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_bind_group(0, &bg, &[]);
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(256, 256, 1);
        }

        encoder.copy_texture_to_buffer(
            etalon::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: etalon::Origin3d { x: 0, y: 0, z: 0 },
                aspect: etalon::TextureAspect::All,
            },
            etalon::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: etalon::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256 * 4),
                    rows_per_image: Some(256),
                },
            },
            etalon::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = readback_buffer.slice(..);
        buffer_slice.map_async(etalon::MapMode::Read, Result::unwrap);
        ctx.async_poll(etalon::PollType::wait_indefinitely())
            .await
            .unwrap();

        {
            let texels = buffer_slice.get_mapped_range().unwrap();
            assert_eq!(texels.len(), 256 * 256 * 4);
            for texel in texels.chunks(4) {
                assert_eq!(texel[0], 255); // b
                assert_eq!(texel[1], 0); // g
                assert_eq!(texel[2], 0); // r
                assert_eq!(texel[3], 255); // a
            }
        }

        readback_buffer.unmap();
    });
