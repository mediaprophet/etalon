use super::*;

// Independent version-1 wire fixture: two raw I/Q pairs, no USB strings.
fn fixture() -> [u8; HEADER_BYTES + 4] {
    let mut bytes = [0; HEADER_BYTES + 4];
    bytes[..8].copy_from_slice(b"ETRTLSD1");
    bytes[8..12].copy_from_slice(&[1, 0, 0, 4]);
    bytes[16..24].copy_from_slice(&7u64.to_le_bytes());
    bytes[24..28].copy_from_slice(&3u32.to_le_bytes());
    bytes[28..32].copy_from_slice(&2u32.to_le_bytes());
    bytes[32..36].copy_from_slice(&2u32.to_le_bytes());
    bytes[40..48].copy_from_slice(&9u64.to_le_bytes());
    bytes[48..52].copy_from_slice(&100_000_000u32.to_le_bytes());
    bytes[52..56].copy_from_slice(&2_400_000u32.to_le_bytes());
    bytes[56..60].copy_from_slice(&(-5i32).to_le_bytes());
    bytes[68..72].copy_from_slice(&1u32.to_le_bytes());
    bytes[72..76].copy_from_slice(&100_000_000u32.to_le_bytes());
    bytes[76..80].copy_from_slice(&2_400_001u32.to_le_bytes());
    bytes[80..84].copy_from_slice(&(-5i32).to_le_bytes());
    bytes[96..104].copy_from_slice(&4u64.to_le_bytes());
    bytes[HEADER_BYTES..].copy_from_slice(&[0, 255, 127, 128]);
    bytes
}

#[test]
fn historical_payload_preserves_requested_and_reported_values_without_rounding() {
    let bytes = fixture();
    let decoded = decode_capture_payload(&bytes, bytes.len() as u64).unwrap();
    assert_eq!(
        (decoded.target_id, decoded.generation, decoded.device_index),
        (7, 3, 2)
    );
    assert_eq!((decoded.pair_count, decoded.host_read_sequence), (2, 9));
    assert_eq!(decoded.requested.sample_rate_hz(), 2_400_000);
    assert_eq!(decoded.applied.sample_rate_hz, 2_400_001);
    assert_eq!(decoded.requested.frequency_correction_ppm(), -5);
    assert!(decoded.requested.digital_agc());
    assert_eq!(decoded.raw_iq, &[0, 255, 127, 128]);
    assert!(decoded.usb_strings.serial.is_empty());
    assert!(matches!(
        decode_capture_payload(&bytes, bytes.len() as u64 - 1),
        Err(PayloadError::BudgetExceeded)
    ));
}

#[test]
fn malformed_storage_cannot_create_a_different_canonical_interpretation() {
    for offset in [0, 8, 10, 12, 36, 92, 110, 896, 1023] {
        let mut bytes = fixture();
        bytes[offset] ^= 0x80;
        assert!(decode_capture_payload(&bytes, bytes.len() as u64).is_err());
    }
    let mut bytes = fixture();
    bytes[96] = 3;
    assert!(decode_capture_payload(&bytes, bytes.len() as u64).is_err());
    let mut bytes = fixture();
    bytes[64] = 1; // Automatic gain has a canonical absent manual value.
    assert!(decode_capture_payload(&bytes, bytes.len() as u64).is_err());
    let bytes = fixture();
    assert!(decode_capture_payload(&bytes[..bytes.len() - 1], bytes.len() as u64).is_err());
}

#[test]
fn usb_descriptors_are_byte_preserved_and_zero_padding_is_canonical() {
    let mut bytes = fixture();
    bytes[108..110].copy_from_slice(&2u16.to_le_bytes());
    bytes[640..642].copy_from_slice(&[0xff, b'1']);
    assert_eq!(
        decode_capture_payload(&bytes, bytes.len() as u64)
            .unwrap()
            .usb_strings
            .serial,
        &[0xff, b'1']
    );
    bytes[642] = b'2';
    assert!(decode_capture_payload(&bytes, bytes.len() as u64).is_err());
    bytes[642] = 0;
    bytes[640] = 0;
    assert!(decode_capture_payload(&bytes, bytes.len() as u64).is_err());
}
