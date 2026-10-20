//! Tests for FLOAT32_FILTERABLE feature.

use etalon_test::{apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        FLOAT32_FILTERABLE_WITHOUT_FEATURE,
        FLOAT32_FILTERABLE_WITH_FEATURE,
    ]);
}

fn create_texture_binding(device: &etalon::Device, format: etalon::TextureFormat, filterable: bool) {
    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format,
        usage: etalon::TextureUsages::TEXTURE_BINDING | etalon::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let view = texture.create_view(&etalon::TextureViewDescriptor::default());

    let bgl = device.create_bind_group_layout(&etalon::BindGroupLayoutDescriptor {
        label: None,
        entries: &[etalon::BindGroupLayoutEntry {
            binding: 0,
            visibility: etalon::ShaderStages::FRAGMENT,
            ty: etalon::BindingType::Texture {
                sample_type: etalon::TextureSampleType::Float { filterable },
                multisampled: false,
                view_dimension: etalon::TextureViewDimension::D2,
            },
            count: None,
        }],
    });

    let _bg = device.create_bind_group(&etalon::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[etalon::BindGroupEntry {
            binding: 0,
            resource: etalon::BindingResource::TextureView(&view),
        }],
    });
}

#[apply(gpu_test!)]
static FLOAT32_FILTERABLE_WITHOUT_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let device = &ctx.device;
        // Unorm textures are always filterable
        create_texture_binding(device, etalon::TextureFormat::R8Unorm, true);
        create_texture_binding(device, etalon::TextureFormat::R8Unorm, false);
        // As are float16 textures
        create_texture_binding(device, etalon::TextureFormat::R16Float, true);
        create_texture_binding(device, etalon::TextureFormat::R16Float, false);
        // Float 32 textures can be used as non-filterable only
        create_texture_binding(device, etalon::TextureFormat::R32Float, false);
        // This is supposed to fail, since we have not activated the feature
        fail(
            &ctx.device,
            || {
                create_texture_binding(device, etalon::TextureFormat::R32Float, true);
            },
            Some(concat!(
                "texture binding 0 expects sample type float { filterable: true }, ",
                "but was given a view with format r32float ",
                "(sample type float { filterable: false })"
            )),
        );
    });

#[apply(gpu_test!)]
static FLOAT32_FILTERABLE_WITH_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(etalon::Features::FLOAT32_FILTERABLE)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        let device = &ctx.device;
        // With the feature enabled, it does work!
        create_texture_binding(device, etalon::TextureFormat::R32Float, true);
        create_texture_binding(device, etalon::TextureFormat::Rg32Float, true);
        create_texture_binding(device, etalon::TextureFormat::Rgba32Float, true);
    });
