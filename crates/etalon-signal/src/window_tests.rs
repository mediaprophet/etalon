use super::*;

#[test]
fn applies_exact_caller_coefficients_without_normalization() {
    let samples = [2.0, -3.0, 4.0];
    let coefficients = [0.5, 2.0, -0.25];
    let mut out = [99.0; 4];
    assert_eq!(apply_window_into(&samples, &coefficients, &mut out), Ok(3));
    assert_eq!(out, [1.0, -6.0, -1.0, 99.0]);
}

#[test]
fn rectangular_window_is_a_caller_supplied_identity() {
    let samples = [0.25, -1.0, 2.0];
    let coefficients = [1.0, 1.0, 1.0];
    let mut out = [0.0; 3];
    apply_window_into(&samples, &coefficients, &mut out).unwrap();
    assert_eq!(out, samples);
}

#[test]
fn invalid_lengths_leave_output_untouched() {
    let samples = [1.0, 2.0];
    let mut out = [9.0; 2];
    assert_eq!(
        apply_window_into(&samples, &[1.0], &mut out),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(
        apply_window_into(&samples, &[1.0, 1.0], &mut out[..1]),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(out, [9.0; 2]);
}

#[test]
fn non_finite_inputs_leave_output_untouched() {
    let mut out = [7.0; 2];
    assert_eq!(
        apply_window_into(&[1.0, f32::NAN], &[1.0, 1.0], &mut out),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(
        apply_window_into(&[1.0, 2.0], &[1.0, f32::INFINITY], &mut out),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(out, [7.0; 2]);
}

#[test]
fn f32_result_overflow_is_reported_before_writing() {
    let mut out = [5.0; 2];
    assert_eq!(
        apply_window_into(&[f32::MAX, 1.0], &[2.0, 1.0], &mut out),
        Err(SignalError::NumericRange)
    );
    assert_eq!(out, [5.0; 2]);
}

#[test]
fn empty_samples_are_rejected() {
    assert_eq!(
        apply_window_into(&[], &[], &mut []),
        Err(SignalError::EmptyInput)
    );
}
