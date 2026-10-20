fn noop_adapter() -> etalon::Adapter {
    let instance = etalon::Instance::new(etalon::InstanceDescriptor {
        backends: etalon::Backends::NOOP,
        backend_options: etalon::BackendOptions {
            noop: etalon::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..etalon::InstanceDescriptor::new_without_display_handle()
    });

    pollster::block_on(instance.request_adapter(&etalon::RequestAdapterOptions::default()))
        .expect("noop backend adapter absent when it should be")
}

#[test]
fn request_no_experimental_features() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&etalon::DeviceDescriptor {
        // Not experimental
        required_features: etalon::Features::FLOAT32_FILTERABLE,
        experimental_features: etalon::ExperimentalFeatures::disabled(),
        ..Default::default()
    }));

    assert!(dq.is_ok());
}

#[test]
fn request_experimental_features() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&etalon::DeviceDescriptor {
        // Experimental
        required_features: etalon::Features::EXPERIMENTAL_MESH_SHADER,
        experimental_features: unsafe { etalon::ExperimentalFeatures::enabled() },
        ..Default::default()
    }));

    assert!(dq.is_ok());
}

#[test]
fn request_experimental_features_when_not_enabled() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&etalon::DeviceDescriptor {
        // Experimental
        required_features: etalon::Features::EXPERIMENTAL_MESH_SHADER,
        experimental_features: etalon::ExperimentalFeatures::disabled(),
        ..Default::default()
    }));

    assert!(dq.is_err());
}

#[test]
fn request_multiple_experimental_features_when_not_enabled() {
    let adapter = noop_adapter();

    let dq = pollster::block_on(adapter.request_device(&etalon::DeviceDescriptor {
        // Experimental
        required_features: etalon::Features::EXPERIMENTAL_MESH_SHADER
            | etalon::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        experimental_features: etalon::ExperimentalFeatures::disabled(),
        ..Default::default()
    }));

    assert!(dq.is_err());
}
