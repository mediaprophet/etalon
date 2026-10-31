use super::*;

#[test]
fn asymmetric_octets_round_trip_in_explicit_network_order() {
    let address = RadioAddress48::from_octets([0x01, 0x23, 0x45, 0x67, 0x89, 0xab]);
    assert_eq!(address.as_u64(), 0x0123_4567_89ab);
    assert_eq!(RadioAddress48::try_from_u64(address.as_u64()), Ok(address));
    assert_eq!(core::mem::size_of::<RadioAddress48>(), 6);
}

#[test]
fn high_bits_are_rejected_without_aliasing_a_valid_address() {
    for value in [1_u64 << 48, 1_u64 << 63, u64::MAX, 0x0001_0123_4567_89ab] {
        assert_eq!(RadioAddress48::try_from_u64(value), Err(ContractError::InvalidIdentifier));
    }
}

#[test]
fn raw_zero_and_broadcast_values_are_preserved() {
    for octets in [[0; 6], [0xff; 6]] {
        let address = RadioAddress48::from_octets(octets);
        assert_eq!(address.octets(), octets);
        assert_eq!(RadioAddress48::try_from_u64(address.as_u64()), Ok(address));
    }
}

#[test]
fn identical_bytes_do_not_conflate_identifier_namespaces() {
    let address = RadioAddress48::from_octets([2, 3, 4, 5, 6, 7]);
    let labels = [
        RadioIdentifier::Mac(address),
        RadioIdentifier::Bssid(address),
        RadioIdentifier::Bluetooth { address, kind: BluetoothAddressKind::Unknown },
        RadioIdentifier::Bluetooth { address, kind: BluetoothAddressKind::Public },
        RadioIdentifier::Bluetooth { address, kind: BluetoothAddressKind::Random },
    ];
    for (index, label) in labels.iter().enumerate() {
        assert_eq!(label.address(), address);
        for other in &labels[index + 1..] {
            assert_ne!(label, other);
        }
    }
}

#[test]
fn unknown_bluetooth_kind_stays_unknown_regardless_of_address_bits() {
    for first in [0x00, 0x02, 0x40, 0x80, 0xc0, 0xff] {
        let identifier = RadioIdentifier::Bluetooth {
            address: RadioAddress48::from_octets([first, 1, 2, 3, 4, 5]),
            kind: BluetoothAddressKind::Unknown,
        };
        assert!(matches!(identifier, RadioIdentifier::Bluetooth {
            kind: BluetoothAddressKind::Unknown, ..
        }));
    }
}

#[test]
fn observations_keep_unknown_and_raw_signed_time_distinct() {
    let identifier = RadioIdentifier::Bssid(RadioAddress48::from_octets([1; 6]));
    let unknown = RadioIdentifierObservation {
        identifier,
        observed_at: RadioObservationTime::Unknown,
    };
    for ticks in [i64::MIN, -1, 0, i64::MAX] {
        let reported = RadioIdentifierObservation {
            identifier,
            observed_at: RadioObservationTime::WindowsUtc100ns(ticks),
        };
        assert_ne!(unknown, reported);
        assert_eq!(reported.observed_at, RadioObservationTime::WindowsUtc100ns(ticks));
    }
}

#[test]
fn clock_context_and_restart_epoch_remain_attached() {
    let identifier = RadioIdentifier::Mac(RadioAddress48::from_octets([2; 6]));
    let stamp = ClockStamp::new(7, 9, 100, 4).unwrap();
    let observation = RadioIdentifierObservation {
        identifier,
        observed_at: RadioObservationTime::Clock(stamp),
    };
    assert_eq!(observation.observed_at, RadioObservationTime::Clock(stamp));
    assert_ne!(observation.observed_at, RadioObservationTime::Clock(
        ClockStamp::new(7, 10, 100, 4).unwrap()
    ));
    assert_ne!(observation.observed_at, RadioObservationTime::WindowsUtc100ns(100));
}
