use super::*;

#[test]
fn i16_iq_pipeline_respects_order_and_scaling() {
    let mut scratch = [0.0; 2];
    let mut power = [0.0; 1];
    assert_eq!(i16_iq_power_into(&[3, 0, 4, 0], ByteOrder::LittleEndian, 1.0, &mut scratch, &mut power), Ok(1));
    assert_eq!(power, [25.0]);
}

#[test]
fn invalid_shape_leaves_both_buffers_untouched() {
    let mut scratch = [7.0; 2];
    let mut power = [8.0; 1];
    assert_eq!(i16_iq_power_into(&[3, 0], ByteOrder::LittleEndian, 1.0, &mut scratch, &mut power), Err(SignalError::LengthMismatch));
    assert_eq!(scratch, [7.0; 2]);
    assert_eq!(power, [8.0; 1]);
}

#[test]
fn numeric_overflow_leaves_scratch_and_power_untouched() {
    let mut scratch = [7.0; 2];
    let mut power = [8.0; 1];
    let bytes = [0xff, 0x7f, 0xff, 0x7f];
    assert_eq!(
        i16_iq_power_into(
            &bytes,
            ByteOrder::LittleEndian,
            f32::MAX,
            &mut scratch,
            &mut power,
        ),
        Err(SignalError::NumericRange)
    );
    assert_eq!(scratch, [7.0; 2]);
    assert_eq!(power, [8.0; 1]);
}
