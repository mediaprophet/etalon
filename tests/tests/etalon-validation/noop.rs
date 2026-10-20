//! Tests of [`etalon::Backend::Noop`].

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

#[test]
fn device_is_not_available_by_default() {
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::NOOP,
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });

    pollster::block_on(instance.request_adapter(&etalon::RequestAdapterOptions::default()))
        .expect_err("noop backend adapter present when it should not be");
}

#[test]
fn device_is_available_when_requested() {
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::NOOP,
        backend_options: etalon::BackendOptions {
            noop: etalon::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });

    pollster::block_on(instance.request_adapter(&etalon::RequestAdapterOptions::default()))
        .expect("noop backend adapter absent when it should be");
}

#[test]
fn device_and_buffers() {
    let (device, queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());

    // Demonstrate that creating and *writing* to a buffer succeeds.
    // This also involves creation of a staging buffer.
    let buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: Some("hello world"),
        size: 8,
        usage: etalon::BufferUsages::COPY_DST | etalon::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    assert_eq!(buffer.size(), 8);
    queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]);
    queue.write_buffer(&buffer, 4, &[5, 6, 7, 8]);

    // Demonstrate that we can read back data from the buffer.
    // This also involves copy_buffer_to_buffer().
    let done: Arc<AtomicBool> = Arc::default();
    let done2 = done.clone();
    etalon::util::DownloadBuffer::read_buffer(&device, &queue, &buffer.slice(..), move |result| {
        assert_eq!(*result.unwrap(), [1, 2, 3, 4, 5, 6, 7, 8],);
        done.store(true, Relaxed);
    });
    device.poll(etalon::PollType::wait_indefinitely()).unwrap();
    assert!(done2.load(Relaxed));
}
