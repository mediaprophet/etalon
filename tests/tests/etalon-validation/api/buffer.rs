//! Tests of [`etalon::Buffer`] and related.

/// Ensures that submitting a command buffer referencing an already destroyed buffer
/// results in an error.
#[test]
#[should_panic = "Buffer with '' label has been destroyed"]
fn destroyed_buffer() {
    let (device, queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: 1024,
        usage: etalon::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&etalon::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&buffer, 0, None);

    buffer.destroy();

    queue.submit([encoder.finish()]);
}
