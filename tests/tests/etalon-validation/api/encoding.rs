//! Tests of [`etalon::CommandEncoder`] and related.

#[test]
fn as_hal() {
    // Sanity-test that the raw encoding API isn't completely broken.

    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    unsafe {
        encoder.as_hal_mut::<etalon_hal::api::Noop, _, ()>(|_| ());
    }
    encoder.finish();
}

#[test]
#[should_panic = "Mixing the etalon encoding API with the raw encoding API is not permitted"]
fn mix_apis_etalon_then_hal() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 256,
        usage: etalon::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&buffer, 0, None);
    unsafe {
        encoder.as_hal_mut::<etalon_hal::api::Noop, _, ()>(|_| ());
    }
}

#[test]
#[should_panic = "Mixing the etalon encoding API with the raw encoding API is not permitted"]
fn mix_apis_hal_then_etalon() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 256,
        usage: etalon::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());
    unsafe {
        encoder.as_hal_mut::<etalon_hal::api::Noop, _, ()>(|_| ());
    }
    encoder.clear_buffer(&buffer, 0, None);
}

/// Test that the command encoder’s label is remembered and used in errors.
#[test]
#[should_panic = "In CommandEncoder::finish, label = 'my encoder'"]
fn encoding_error_contains_label_of_encoder() {
    let (device, queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: Some("my buffer"),
        size: 1024,
        usage: etalon::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
        label: Some("my encoder"),
    });
    // This is erroneous because it is copying to the same buffer.
    encoder.copy_buffer_to_buffer(&buffer, 0, &buffer, 0, 10);
    queue.submit([encoder.finish()]);
}
