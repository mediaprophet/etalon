use super::*;
use etalon_sense_core::{buffer::SampleLayout, quantity::SampleEncoding};

fn layout() -> SampleLayout {
    SampleLayout::new(1, [2, 0, 0, 0], [2, 0, 0, 0], SampleEncoding::I16).unwrap()
}

#[test]
fn exact_rssi_payload_is_checked_against_batch_records() {
    let records = [
        BleAdvertisementSample {
            bluetooth_address: 0x1122_3344_5566,
            rssi_dbm: -51,
            os_timestamp_100ns: 1_337,
            os_address_type: None,
        },
        BleAdvertisementSample {
            bluetooth_address: 0x9988_7766_5544,
            rssi_dbm: -72,
            os_timestamp_100ns: 1_338,
            os_address_type: None,
        },
    ];
    let first = (-51i16).to_le_bytes();
    let second = (-72i16).to_le_bytes();
    let payload = [first[0], first[1], second[0], second[1]];
    let layout = layout();
    assert_eq!(
        validate_batch_binding(&payload, &layout, &layout, &records),
        Ok(())
    );
    assert_eq!(records[0].bluetooth_address, 0x1122_3344_5566);
    assert_eq!(records[0].os_timestamp_100ns, 1_337);

    let zero = 0i16.to_le_bytes();
    let wrong_payload = [zero[0], zero[1], second[0], second[1]];
    assert_eq!(
        validate_batch_binding(&wrong_payload, &layout, &layout, &records),
        Err(ObservationError::MetadataMismatch),
    );
}

#[test]
fn wrong_count_or_layout_rejects_metadata_pairing() {
    let records = [BleAdvertisementSample {
        bluetooth_address: 1,
        rssi_dbm: -50,
        os_timestamp_100ns: 8,
        os_address_type: None,
    }];
    let payload = (-50i16).to_le_bytes();
    let one = SampleLayout::new(1, [1, 0, 0, 0], [2, 0, 0, 0], SampleEncoding::I16).unwrap();
    let two = layout();
    assert_eq!(
        validate_batch_binding(&payload, &one, &two, &records),
        Err(ObservationError::MetadataMismatch),
    );
    assert_eq!(
        validate_batch_binding(&payload, &one, &one, &[]),
        Err(ObservationError::MetadataMismatch),
    );
}

#[test]
fn dropped_counter_metadata_is_an_interval_not_per_batch_loss() {
    let interval = BleDropCountInterval {
        cumulative_before_drain: 12,
        cumulative_after_drain: 15,
    };
    assert_eq!(interval.cumulative_before_drain(), 12);
    assert_eq!(interval.cumulative_after_drain(), 15);
}
