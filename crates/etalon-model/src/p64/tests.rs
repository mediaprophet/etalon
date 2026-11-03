use super::*;

fn minimal() -> [u8; 256] {
    let mut bytes = [0; 256];
    bytes[..4].copy_from_slice(&P64_MAGIC);
    bytes[4..6].copy_from_slice(&P64_VERSION.to_le_bytes());
    bytes[6..8].copy_from_slice(&P64_FLAG_LITTLE_ENDIAN.to_le_bytes());
    for (offset, value) in [
        (12, 128u32),
        (16, 192),
        (20, 64),
        (24, 128),
        (28, 192),
        (32, 128),
        (40, 256),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    seal(&mut bytes);
    bytes
}

fn seal(bytes: &mut [u8; 256]) {
    let crc = integrity::crc32c(&bytes[..192]);
    bytes[192..196].copy_from_slice(&crc.to_le_bytes());
}

#[test]
fn crc_known_vector() {
    assert_eq!(integrity::crc32c(b"123456789"), 0xe306_9283);
    assert_eq!(integrity::crc32c(&[]), 0);
}

#[test]
fn empty_manifest_preserves_borrow_and_record() {
    let bytes = minimal();
    let view = P64View::parse(&bytes, IntegrityMode::Full).unwrap();
    assert_eq!(view.tensor_count(), 0);
    assert_eq!(view.bytes().as_ptr(), bytes.as_ptr());
    assert_eq!(view.manifold_record(0).unwrap().bytes(), &[0; 64]);
    assert!(view.tensor(0).is_err());
    assert!(view.manifold_record(1).is_err());
    assert_eq!(view.tokenizer_bytes(), &[]);
    assert_eq!(view.schedule_bytes().unwrap(), &[]);
}

#[test]
fn truncation_never_panics() {
    let bytes = minimal();
    for length in 0..196 {
        assert!(P64View::parse(&bytes[..length], IntegrityMode::Full).is_err());
    }
}

#[test]
fn raw_padding_and_nonfinite_values_are_preserved() {
    let mut bytes = minimal();
    bytes[44] = 23;
    bytes[120..128].fill(17);
    bytes[128..132].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    bytes[168..192].fill(71);
    seal(&mut bytes);
    let view = P64View::parse(&bytes, IntegrityMode::Full).unwrap();
    assert_eq!(view.header().reserved[0], 23);
    assert_eq!(view.hparams().reserved, [0; 8]);
    assert_eq!(view.bytes()[120..128], [17; 8]);
    let record = view.manifold_record(0).unwrap();
    assert_eq!(record.values_le()[0].to_bits(), f32::NAN.to_bits());
    assert_eq!(record.bytes()[40..], [71; 24]);
}

#[test]
fn integrity_modes_do_not_silently_upgrade() {
    let mut bytes = minimal();
    bytes[44] ^= 1;
    assert!(P64View::parse(&bytes, IntegrityMode::Structure).is_ok());
    for mode in [IntegrityMode::Metadata, IntegrityMode::Full] {
        assert_eq!(
            P64View::parse(&bytes, mode).unwrap_err(),
            P64Error::MetadataChecksum
        );
    }
}

#[test]
fn schedule_is_checked_on_access_without_tightening_parse() {
    let mut bytes = minimal();
    bytes[6..8].copy_from_slice(&(P64_FLAG_LITTLE_ENDIAN | P64_FLAG_LAYER_SCHEDULE).to_le_bytes());
    bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    seal(&mut bytes);
    let view = P64View::parse(&bytes, IntegrityMode::Full).unwrap();
    assert!(view.schedule_bytes().is_err());
}

#[test]
fn legacy_layout_read_write_and_sizes() {
    assert_eq!(core::mem::size_of::<P64WeightHeader>(), 64);
    assert_eq!(core::mem::align_of::<P64WeightHeader>(), 64);
    assert_eq!(core::mem::size_of::<P64TensorEntry>(), 64);
    assert_eq!(core::mem::size_of::<P64HParams>(), 64);
    assert_eq!(core::mem::size_of::<P64LayerScheduleEntry>(), 64);
    let bytes = minimal();
    let header = P64WeightHeader::read_le(&bytes).unwrap();
    let mut output = [0; 64];
    header.write_le(&mut output);
    assert_eq!(output, bytes[..64]);
    assert!(P64WeightHeader::read_le(&bytes[..63]).is_err());
    assert!(P64HParams::read_le(&bytes[..63]).is_err());
}

#[test]
fn structure_still_rejects_invalid_header_and_count() {
    let bytes = minimal();
    for offset in [0, 4, 6, 40] {
        let mut altered = bytes;
        altered[offset] ^= 1;
        assert!(P64View::parse(&altered, IntegrityMode::Structure).is_err());
    }
    let mut altered = bytes;
    altered[64..68].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        P64View::parse(&altered, IntegrityMode::Structure).unwrap_err(),
        P64Error::ArithmeticOverflow(P64OverflowContext::ManifoldCount)
    );
}
