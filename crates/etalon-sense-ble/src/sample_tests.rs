use super::*;

fn sample(address: u64, os_type: Option<i32>, ticks: i64) -> BleAdvertisementSample {
    BleAdvertisementSample {
        bluetooth_address: address,
        rssi_dbm: -67,
        os_timestamp_100ns: ticks,
        os_address_type: os_type,
    }
}

#[test]
fn typed_observation_preserves_raw_address_type_and_signed_ticks() {
    let raw = sample(0x0123_4567_89ab, Some(BluetoothAddressType::Random.0), -31);
    let observed = raw.radio_identifier_observation().unwrap();
    assert_eq!(
        observed.identifier,
        RadioIdentifier::Bluetooth {
            address: RadioAddress48::from_octets([0x01, 0x23, 0x45, 0x67, 0x89, 0xab]),
            kind: BluetoothAddressKind::Random,
        }
    );
    assert_eq!(
        observed.observed_at,
        RadioObservationTime::WindowsUtc100ns(-31)
    );
    assert_eq!(raw.bluetooth_address, 0x0123_4567_89ab);
    assert_eq!(raw.rssi_dbm, -67);
    assert_eq!(raw.os_timestamp_100ns, -31);
    assert_eq!(raw.os_address_type, Some(BluetoothAddressType::Random.0));
}

#[test]
fn address_kind_comes_only_from_reported_os_type() {
    for address in [0, 0x0200_0000_0001, 0xc000_0000_0001, 0xffff_ffff_ffff] {
        for (reported, expected) in [
            (None, BluetoothAddressKind::Unknown),
            (
                Some(BluetoothAddressType::Public.0),
                BluetoothAddressKind::Public,
            ),
            (
                Some(BluetoothAddressType::Random.0),
                BluetoothAddressKind::Random,
            ),
            (
                Some(BluetoothAddressType::Unspecified.0),
                BluetoothAddressKind::Unknown,
            ),
            (Some(-1), BluetoothAddressKind::Unknown),
            (Some(i32::MAX), BluetoothAddressKind::Unknown),
        ] {
            let raw = sample(address, reported, 0);
            assert_eq!(raw.bluetooth_address_kind(), expected);
            assert_eq!(raw.os_address_type, reported);
        }
    }
}

#[test]
fn unsupported_os_property_falls_back_without_losing_other_fields() {
    let unavailable = raw_address_type::<()>(Err(()));
    assert_eq!(unavailable, None);
    let raw = sample(0x1122_3344_5566, unavailable, 123);
    let observed = raw.radio_identifier_observation().unwrap();
    assert_eq!(raw.bluetooth_address_kind(), BluetoothAddressKind::Unknown);
    assert_eq!(
        observed.identifier.address().as_u64(),
        raw.bluetooth_address
    );
    assert_eq!(
        observed.observed_at,
        RadioObservationTime::WindowsUtc100ns(123)
    );
    assert_eq!(
        raw_address_type::<()>(Ok(BluetoothAddressType(91))),
        Some(91)
    );
}

#[test]
fn invalid_high_bits_are_preserved_raw_and_rejected_by_typed_projection() {
    let raw = sample(
        0x0001_1122_3344_5566,
        Some(BluetoothAddressType::Public.0),
        9,
    );
    assert_eq!(
        raw.radio_identifier_observation(),
        Err(ContractError::InvalidIdentifier)
    );
    assert_eq!(raw.bluetooth_address, 0x0001_1122_3344_5566);
    assert_eq!(raw.os_timestamp_100ns, 9);
}

#[test]
fn default_sample_does_not_assert_public_address_type() {
    let raw = BleAdvertisementSample::default();
    assert_eq!(raw.os_address_type, None);
    assert_eq!(raw.bluetooth_address_kind(), BluetoothAddressKind::Unknown);
}

#[test]
fn queue_keeps_address_type_and_timestamp_paired_with_each_advertisement() {
    let first = sample(0x1122_3344_5566, Some(BluetoothAddressType::Public.0), 7);
    let second = sample(0x1122_3344_5566, Some(BluetoothAddressType::Random.0), 8);
    let mut queue = crate::queue::FixedQueue::new();
    assert!(queue.push(first));
    assert!(queue.push(second));
    let mut records = [BleAdvertisementSample::default(); 2];
    assert!(queue.pop_exact(&mut records));
    assert_eq!(records, [first, second]);
    assert_ne!(
        records[0].radio_identifier_observation(),
        records[1].radio_identifier_observation()
    );
    queue.clear();
    assert_eq!(queue.len(), 0);
}
