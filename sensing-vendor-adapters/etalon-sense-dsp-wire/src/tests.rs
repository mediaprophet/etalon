use super::*;

fn identity() -> RequestIdentity { RequestIdentity::new(7, 2, 9).unwrap() }

#[test]
fn request_header_is_explicit_little_endian() {
    let mut request = [0u8; 68];
    encode_request(identity(), 1, &[0x00, 0x80, 0xff, 0x7f], &mut request).unwrap();
    let expected = [
        b'E', b'T', b'I', b'Q', b'P', b'W', b'0', b'1',
        1, 0, 64, 0, 1, 0, 0, 0,
        0x31, 0x52, 0x45, 0x57, 0x4f, 0x50, 0x51, 0x49,
        7, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 1, 0, 0, 0,
        9, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 8, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0,
        0x00, 0x80, 0xff, 0x7f,
    ];
    assert_eq!(request, expected);
    assert_eq!(decode_request(&request), Ok((identity(), 1, &request[64..])));
}

#[test]
fn reject_each_fixed_header_field_and_length_mismatch() {
    let mut request = [0u8; 68];
    encode_request(identity(), 1, &[0; 4], &mut request).unwrap();
    for offset in [0, 8, 10, 12, 16, 48, 52, 56, 60] {
        let mut malformed = request;
        malformed[offset] ^= 0x80;
        assert!(decode_request(&malformed).is_err());
    }
    assert!(decode_request(&request[..67]).is_err());
    assert!(decode_request(&[0; 63]).is_err());
    let mut oversized = [0u8; 69];
    oversized[..68].copy_from_slice(&request);
    assert!(decode_request(&oversized).is_err());
}

#[test]
fn response_requires_echo_and_valid_numeric_domain() {
    let mut response = [0u8; 72];
    response[64..].copy_from_slice(&2_147_483_648f64.to_le_bytes());
    finish_response(identity(), 1, &mut response).unwrap();
    assert_eq!(decode_response(identity(), 1, &response), Ok(&response[64..]));
    assert!(decode_response(RequestIdentity::new(7, 2, 10).unwrap(), 1, &response).is_err());
    assert!(decode_response(RequestIdentity::new(8, 2, 9).unwrap(), 1, &response).is_err());
    assert!(decode_response(RequestIdentity::new(7, 3, 9).unwrap(), 1, &response).is_err());
    for invalid in [f64::NAN, f64::INFINITY, -1.0, -0.0, 0.5, 2_147_483_649.0] {
        response[64..].copy_from_slice(&invalid.to_le_bytes());
        assert!(decode_response(identity(), 1, &response).is_err());
    }
}

#[test]
fn capacities_and_identities_fail_closed() {
    assert_eq!(request_bytes(0), Err(WireError::InvalidCount));
    assert_eq!(response_bytes(MAX_PAIRS + 1), Err(WireError::InvalidCount));
    assert_eq!(request_bytes(MAX_PAIRS), Ok(4_194_368));
    assert_eq!(response_bytes(MAX_PAIRS), Ok(8_388_672));
    assert!(RequestIdentity::new(0, 2, 9).is_err());
    assert!(RequestIdentity::new(7, 0, 9).is_err());
    assert!(RequestIdentity::new(7, 2, 0).is_err());
    assert!(encode_request(identity(), 1, &[0; 3], &mut [0; 68]).is_err());
}

#[test]
fn loader_uri_has_one_canonical_generation_and_handle_export() {
    const PREFIX: &str = "file:///libetalon_skel.so?etalon_iq_power_skel_handle_invoke&_modver=1.0&etalon_generation=";
    assert_eq!(generation_from_uri(concat!("file:///libetalon_skel.so?",
        "etalon_iq_power_skel_handle_invoke&_modver=1.0&etalon_generation=4294967295&_dom=cdsp").as_bytes()),
        Ok(u32::MAX));
    let mut bytes = [0u8; 256];
    for invalid in ["0", "01", "-1", "+1", "4294967296", "1&etalon_generation=2",
        "1&_modver=1.0", "1&_mod=2.0", "1&_dom=cdsp&_dom=adsp", "1#fragment", "1%00", "1\0"] {
        bytes[..PREFIX.len()].copy_from_slice(PREFIX.as_bytes());
        bytes[PREFIX.len()..PREFIX.len()+invalid.len()].copy_from_slice(invalid.as_bytes());
        assert!(generation_from_uri(&bytes[..PREFIX.len()+invalid.len()]).is_err());
    }
    assert!(generation_from_uri(b"file:///lib.so?etalon_iq_power_skel_invoke&_modver=1.0&etalon_generation=1").is_err());
    assert!(generation_from_uri(b"file:///lib.so?etalon_iq_power_skel_handle_invoke&etalon_generation=1").is_err());
}
