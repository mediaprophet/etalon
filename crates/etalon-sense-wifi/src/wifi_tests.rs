use super::*;

#[test]
fn wifi_error_variants_preserve_required_output_capacity() {
    assert_eq!(WifiError::OutputTooSmall { required: 3 }, WifiError::OutputTooSmall { required: 3 });
    assert_ne!(WifiError::PolicyDenied, WifiError::TooManyResults);
}

#[test]
fn table_validation_rejects_truncated_header_and_entry_array() {
    assert_eq!(validate_table(7, 0, 0, 8, 16), Err(WifiError::InvalidList));
    assert_eq!(validate_table(23, 1, 1, 8, 16), Err(WifiError::InvalidList));
}

#[test]
fn table_validation_enforces_hard_and_caller_supplied_bounds() {
    assert_eq!(
        validate_table(8 + 257 * 16, 257, 257, 8, 16),
        Err(WifiError::TooManyResults),
    );
    assert_eq!(
        validate_table(8 + 3 * 16, 3, 2, 8, 16),
        Err(WifiError::OutputTooSmall { required: 3 }),
    );
    assert_eq!(
        validate_table(MAX_BSS_LIST_BYTES + 1, 0, 0, 8, 16),
        Err(WifiError::InvalidList),
    );
}

#[test]
fn information_element_spans_must_stay_inside_the_reported_allocation() {
    assert_eq!(validate_ie_span(128, 32, 32, 96), Ok(()));
    assert_eq!(validate_ie_span(128, 32, 31, 1), Err(WifiError::InvalidList));
    assert_eq!(validate_ie_span(128, 8, 127, 2), Err(WifiError::InvalidList));
    assert_eq!(validate_ie_span(usize::MAX, 8, usize::MAX, 1), Err(WifiError::InvalidList));
}

#[test]
fn information_element_offset_is_relative_to_each_entry() {
    let first_entry_offset = 8usize;
    let later_entry_offset = 8usize + 128;
    let entries_end = later_entry_offset + 128;

    let first_ie = entry_relative_ie_start(first_entry_offset, entries_end - first_entry_offset).unwrap();
    let later_ie = entry_relative_ie_start(later_entry_offset, entries_end - later_entry_offset).unwrap();
    assert_eq!(first_ie, entries_end);
    assert_eq!(later_ie, entries_end);
    assert_eq!(validate_ie_span(300, entries_end, first_ie, 1), Ok(()));
    assert_eq!(validate_ie_span(300, entries_end, later_ie, 1), Ok(()));
    assert_eq!(entry_relative_ie_start(usize::MAX, 1), Err(WifiError::InvalidList));

    // Interpreting an entry-relative offset as allocation-relative would point
    // inside the fixed entry table and must not be accepted as a valid IE span.
    assert_eq!(validate_ie_span(300, entries_end, 128, 1), Err(WifiError::InvalidList));
}

#[test]
fn observation_keeps_rssi_and_quality_as_distinct_fields() {
    let observation = BssObservation {
        bssid: [1, 2, 3, 4, 5, 6],
        rssi_dbm: -57,
        link_quality: 83,
        center_frequency_khz: 5_180_000,
    };
    assert_eq!(observation.rssi_dbm, -57);
    assert_eq!(observation.link_quality, 83);
    assert_ne!(observation.rssi_dbm, observation.link_quality as i32);
}

#[test]
fn summary_marks_os_timestamp_and_cache_age_unknown() {
    let summary = BssReadSummary {
        written: 0,
        timestamp: TimestampStatus::Unknown,
        cache_age_known: false,
    };
    assert_eq!(summary.timestamp, TimestampStatus::Unknown);
    assert!(!summary.cache_age_known);
}

#[test]
fn current_interface_mac_preserves_exact_bytes_and_unknown_time() {
    let interface = GUID::from_u128(0x12345678_1234_5678_90ab_cdef01234567);
    let mut address = [0u8; 32];
    address[..6].copy_from_slice(&[0x02, 0x11, 0x22, 0x33, 0x44, 0x55]);
    let observation = checked_current_interface_mac(interface, interface, 6, address).unwrap();
    assert_eq!(observation.identifier, RadioIdentifier::Mac(RadioAddress48::from_octets([0x02, 0x11, 0x22, 0x33, 0x44, 0x55])));
    assert_eq!(observation.observed_at, RadioObservationTime::Unknown);
}

#[test]
fn current_interface_mac_rejects_a_different_row_and_non_six_byte_address() {
    let requested = GUID::from_u128(0x12345678_1234_5678_90ab_cdef01234567);
    let other = GUID::from_u128(0x22345678_1234_5678_90ab_cdef01234567);
    let address = [0xabu8; 32];
    assert_eq!(
        checked_current_interface_mac(requested, other, 6, address),
        Err(WifiError::InterfaceMismatch),
    );
    assert_eq!(
        checked_current_interface_mac(requested, requested, 5, address),
        Err(WifiError::InvalidPhysicalAddressLength(5)),
    );
    assert_eq!(
        checked_current_interface_mac(requested, requested, 7, address),
        Err(WifiError::InvalidPhysicalAddressLength(7)),
    );
}
