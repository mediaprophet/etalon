use super::*;

#[test]
fn complex_power_preserves_tail() {
    let mut out = [0.0, 0.0, 99.0];
    assert_eq!(power_into(&[3.0, 4.0, -1.0, 2.0], &mut out), Ok(2));
    assert_eq!(out, [25.0, 5.0, 99.0]);
}

#[test]
fn invalid_input_does_not_change_output() {
    let mut out = [7.0];
    assert_eq!(power_into(&[1.0, f32::NAN], &mut out), Err(SignalError::NonFiniteInput));
    assert_eq!(out, [7.0]);
    assert_eq!(power_into(&[1.0], &mut out), Err(SignalError::LengthMismatch));
    assert_eq!(out, [7.0]);
}
