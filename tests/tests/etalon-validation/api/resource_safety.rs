use etalon_test::{fail, valid};

fn plane(device: &etalon::Device, format: etalon::TextureFormat) -> etalon::TextureView {
    device
        .create_texture(&etalon::TextureDescriptor {
            label: None,
            size: etalon::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: etalon::TextureDimension::D2,
            format,
            usage: etalon::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

#[test]
fn external_texture_rejects_foreign_planes() {
    let descriptor = etalon::DeviceDescriptor {
        required_features: etalon::Features::EXTERNAL_TEXTURE,
        ..Default::default()
    };
    let (device, _queue) = etalon::Device::noop(&descriptor);
    let (foreign, _foreign_queue) = etalon::Device::noop(&descriptor);
    let rgba = plane(&device, etalon::TextureFormat::Rgba8Unorm);
    let foreign_rgba = plane(&foreign, etalon::TextureFormat::Rgba8Unorm);
    let y = plane(&device, etalon::TextureFormat::R8Unorm);
    let uv = plane(&device, etalon::TextureFormat::Rg8Unorm);
    let foreign_uv = plane(&foreign, etalon::TextureFormat::Rg8Unorm);
    let descriptor = etalon::ExternalTextureDescriptor {
        label: None,
        width: 2,
        height: 2,
        format: etalon::ExternalTextureFormat::Rgba,
        yuv_conversion_matrix: [0.0; 16],
        gamut_conversion_matrix: [0.0; 9],
        src_transfer_function: Default::default(),
        dst_transfer_function: Default::default(),
        sample_transform: [0.0; 6],
        load_transform: [0.0; 6],
    };
    valid(&device, || {
        device.create_external_texture(&descriptor, &[&rgba])
    });
    fail(
        &device,
        || device.create_external_texture(&descriptor, &[&foreign_rgba]),
        Some("doesn't match"),
    );
    let descriptor = etalon::ExternalTextureDescriptor {
        format: etalon::ExternalTextureFormat::Nv12,
        ..descriptor
    };
    valid(&device, || {
        device.create_external_texture(&descriptor, &[&y, &uv])
    });
    fail(
        &device,
        || device.create_external_texture(&descriptor, &[&y, &foreign_uv]),
        Some("doesn't match"),
    );
}

#[test]
fn clear_texture_requires_all_depth_stencil_aspects() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor {
        required_features: etalon::Features::CLEAR_TEXTURE,
        ..Default::default()
    });
    let texture = device.create_texture(&etalon::TextureDescriptor {
        label: None,
        size: etalon::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: etalon::TextureDimension::D2,
        format: etalon::TextureFormat::Depth24PlusStencil8,
        usage: etalon::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let clear = |aspect| {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_texture(
            &texture,
            &etalon::ImageSubresourceRange {
                aspect,
                ..Default::default()
            },
        );
        encoder.finish()
    };
    valid(&device, || clear(etalon::TextureAspect::All));
    for aspect in [
        etalon::TextureAspect::DepthOnly,
        etalon::TextureAspect::StencilOnly,
    ] {
        fail(
            &device,
            || clear(aspect),
            Some("all aspects must be cleared"),
        );
    }
}

#[test]
fn lost_device_precedes_mapping_argument_errors() {
    use etalon_core::{
        device::DeviceError,
        resource::{BufferAccessError, BufferMapOperation},
    };
    use std::sync::mpsc;

    let (device, _queue) = etalon::Device::noop(&Default::default());
    let core_device = device.as_core().unwrap();
    let buffer = core_device.create_buffer(&etalon_core::resource::BufferDescriptor {
        label: None,
        size: 16,
        usage: etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let map = |mode, offset, size| {
        let (sender, receiver) = mpsc::channel();
        let submission = buffer.map_async(
            offset,
            Some(size),
            BufferMapOperation {
                mode,
                callback: Some(Box::new(move |result| sender.send(result).unwrap())),
            },
        );
        assert!(submission.is_none());
        receiver.try_recv().expect("mapping error callback")
    };
    let live_error = fail(&device, || map(etalon::MapMode::Read, 1, 4), Some("offset"));
    assert!(matches!(
        live_error,
        Err(BufferAccessError::UnalignedOffset { .. })
    ));
    device.destroy();
    for (mode, offset, size) in [
        (etalon::MapMode::Read, 1, 4),
        (etalon::MapMode::Read, 24, 4),
        (etalon::MapMode::Write, 0, 16),
    ] {
        assert!(matches!(
            map(mode, offset, size),
            Err(BufferAccessError::Device(DeviceError::Lost))
        ));
    }
}
