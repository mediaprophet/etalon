use super::*;

#[test]
fn two_bin_transform_has_explicit_scaling() {
    let samples = [3.0, 0.0, 1.0, 0.0];
    let coefficients = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    let mut out = [0.0; 4];
    assert_eq!(
        transform_into(&samples, &coefficients, &mut out, DftScale::Raw),
        Ok(2)
    );
    assert_eq!(out, [4.0, 0.0, 2.0, 0.0]);
    assert_eq!(
        transform_into(
            &samples,
            &coefficients,
            &mut out,
            DftScale::DivideBySampleCount
        ),
        Ok(2)
    );
    assert_eq!(out, [2.0, 0.0, 1.0, 0.0]);
}

#[test]
fn invalid_coefficients_do_not_modify_output() {
    let mut out = [9.0; 2];
    assert_eq!(
        transform_into(&[1.0, 0.0], &[f32::NAN, 0.0], &mut out, DftScale::Raw),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(out, [9.0; 2]);

    assert_eq!(
        transform_into(&[1.0, 0.0], &[1.01, 0.0], &mut out, DftScale::Raw),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(out, [9.0; 2]);
}

#[test]
fn shape_failures_do_not_partially_write_bins() {
    let mut out = [7.0; 4];
    assert_eq!(
        transform_into(&[1.0, 0.0, 2.0, 0.0], &[1.0, 0.0], &mut out, DftScale::Raw),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(out, [7.0; 4]);
}
