use super::*;

#[test]
fn decodes_byte_order_and_explicit_scale() {
    let mut out = [0.0; 2];
    assert_eq!(
        decode_i16_into(
            &[0x00, 0x80, 0xff, 0x7f],
            &mut out,
            ByteOrder::LittleEndian,
            1.0 / 32768.0
        ),
        Ok(2)
    );
    assert_eq!(out[0], -1.0);
    assert_eq!(out[1], 32767.0 / 32768.0);
    assert_eq!(
        decode_i16_into(
            &[0x80, 0x00, 0x7f, 0xff],
            &mut out,
            ByteOrder::BigEndian,
            1.0 / 32768.0
        ),
        Ok(2)
    );
    assert_eq!(out[0], -1.0);
}

#[test]
fn invalid_input_leaves_output_unchanged() {
    let mut out = [4.0];
    assert_eq!(
        decode_i16_into(&[0], &mut out, ByteOrder::LittleEndian, 1.0),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(
        decode_i16_into(&[1, 0], &mut out, ByteOrder::LittleEndian, f32::NAN),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(out, [4.0]);

    assert_eq!(
        decode_i16_into(&[0xff, 0x7f], &mut out, ByteOrder::LittleEndian, f32::MAX),
        Err(SignalError::NumericRange)
    );
    assert_eq!(out, [4.0]);
}

#[test]
fn unsigned_samples_use_the_caller_selected_zero_code() {
    let mut out = [0.0; 3];
    assert_eq!(
        decode_u8_centered_into(&[0, 128, 255], &mut out, 128, 1.0),
        Ok(3)
    );
    assert_eq!(out, [-128.0, 0.0, 127.0]);
}
