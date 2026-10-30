//! Tests of [`etalon::util`].

use nanorand::Rng;

/// Generate (deterministic) random staging belt operations to exercise its logic.
fn staging_belt_random_test(use_recall_on_submit: bool, discard_free_chunks: bool) {
    let (device, queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
    let mut rng = nanorand::WyRand::new_seed(0xDEAD_BEEF);
    let buffer_size = 1024;
    let align = etalon::COPY_BUFFER_ALIGNMENT;
    let mut belt = etalon::util::StagingBelt::new(device.clone(), buffer_size / 2);
    let target_buffer = device.create_buffer(&etalon::BufferDescriptor {
        label: None,
        size: buffer_size,
        usage: etalon::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for _batch in 0..100 {
        let mut encoder =
            device.create_command_encoder(&etalon::CommandEncoderDescriptor::default());

        for _write in 0..5 {
            let offset: u64 = rng.generate_range(0..=(buffer_size - align) / align) * align;
            let size: u64 = rng.generate_range(1..=(buffer_size - offset) / align) * align;
            println!("offset {offset} size {size}");

            let mut slice = belt.write_buffer(
                &mut encoder,
                &target_buffer,
                offset,
                etalon::BufferSize::new(size).unwrap(),
            );
            // token amount of actual writing, just in case it makes a difference
            slice.slice(..1).copy_from_slice(&[1]);
        }

        if use_recall_on_submit {
            belt.finish_and_recall_on_submit(&encoder);
            queue.submit([encoder.finish()]);
            // No explicit recall() needed.
        } else {
            belt.finish();
            queue.submit([encoder.finish()]);
            belt.recall();
        }
        if discard_free_chunks {
            device.poll(etalon::PollType::wait_indefinitely()).unwrap();
            belt.discard_free_chunks();
            belt.discard_free_chunks();
        }
    }
}

#[test]
fn staging_belt_manual_recall() {
    staging_belt_random_test(false, false);
}

#[test]
fn staging_belt_finish_and_recall_on_submit() {
    staging_belt_random_test(true, false);
}

#[test]
fn staging_belt_reuses_after_discard_manual_recall() {
    staging_belt_random_test(false, true);
}

#[test]
fn staging_belt_reuses_after_discard_on_submit() {
    staging_belt_random_test(true, true);
}

#[test]
fn staging_belt_panics_with_invalid_buffer_usages() {
    #[track_caller]
    fn test_if_panics(usage: etalon::BufferUsages) {
        let result = std::panic::catch_unwind(|| {
            let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
            let _belt =
                etalon::util::StagingBelt::new_with_buffer_usages(device.clone(), 512, usage);
        });

        if let Err(panic) = result {
            // according to [1] the panic payload is either a `&str` or `String`
            // [1]: https://doc.rust-lang.org/std/macro.panic.html

            let message = if let Some(message) = panic.downcast_ref::<&str>() {
                *message
            } else if let Some(message) = panic.downcast_ref::<String>() {
                message.as_str()
            } else {
                // don't know what this panic is, but it's not ours
                std::panic::resume_unwind(panic);
            };

            let expected_message = format!("Only BufferUsages::COPY_SRC may be used when Features::MAPPABLE_PRIMARY_BUFFERS is not enabled. Specified buffer usages: {usage:?}");
            if expected_message == message {
                // panicked with the correct message
            } else {
                // This is not our panic (or the panic message was changed)
                std::panic::resume_unwind(panic);
            }
        } else {
            panic!("StagingBelt::new_with_buffer_usages should panic without MAPPABLE_PRIMARY_BUFFERS with usage={usage:?}");
        }
    }

    // This tests that `StagingBelt::new_with_buffer_usages` panics for any buffer usages that contain anything else than `COPY_SRC | MAP_WRITE`.
    //
    // First we iterate over all possible buffer usages except `COPY_SRC | MAP_WRITE` (anything invalid).
    for mut usage in etalon::BufferUsages::all()
        .difference(etalon::BufferUsages::COPY_SRC | etalon::BufferUsages::MAP_WRITE)
        .iter()
    {
        // check if the constructor panics with the selected buffer usage
        test_if_panics(usage);

        // add MAP_WRITE to the selected buffer usage and check that the constructor still panics
        usage.insert(etalon::BufferUsages::MAP_WRITE);
        test_if_panics(usage);
    }
}

#[test]
fn staging_belt_works_with_non_exclusive_buffer_usages() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor::default());
    let _belt = etalon::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        etalon::BufferUsages::COPY_SRC,
    );
    let _belt = etalon::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        etalon::BufferUsages::COPY_SRC | etalon::BufferUsages::MAP_WRITE,
    );
    let _belt = etalon::util::StagingBelt::new_with_buffer_usages(
        device.clone(),
        512,
        etalon::BufferUsages::MAP_WRITE,
    );
}

#[test]
fn staging_belt_works_with_exclusive_buffer_usages_with_mappable_primary_buffers() {
    let (device, _queue) = etalon::Device::noop(&etalon::DeviceDescriptor {
        required_features: etalon::Features::MAPPABLE_PRIMARY_BUFFERS,
        ..Default::default()
    });

    // This tests that `StagingBelt::new_with_buffer_usages` works for any buffer usages that contain anything else than `COPY_SRC | MAP_WRITE`.
    //
    // First we iterate over all possible buffer usages except `COPY_SRC | MAP_WRITE` (anything that would be invalid without mappable primary buffers).
    for usage in etalon::BufferUsages::all()
        .difference(etalon::BufferUsages::COPY_SRC | etalon::BufferUsages::MAP_WRITE)
        .iter()
    {
        // Check that the constructor doesn't panic without explicit `MAP_WRITE`
        let _belt = etalon::util::StagingBelt::new_with_buffer_usages(device.clone(), 512, usage);

        // Check that the constructor doesn't panic with explicitly `MAP_WRITE`
        let _belt = etalon::util::StagingBelt::new_with_buffer_usages(
            device.clone(),
            512,
            etalon::BufferUsages::MAP_WRITE,
        );
    }
}
