use super::power_at_coefficient;
use crate::SignalError;

#[test]
fn computes_known_quadrature_bin_power() {
    // Four samples of a unit sinusoid at omega = pi/2; 2*cos(omega) = 0.
    assert_eq!(power_at_coefficient(&[1.0, 0.0, -1.0, 0.0], 0.0), Ok(4.0));
}

#[test]
fn rejects_empty_nonfinite_and_invalid_coefficient_inputs() {
    assert_eq!(power_at_coefficient(&[], 0.0), Err(SignalError::EmptyInput));
    assert_eq!(
        power_at_coefficient(&[1.0, f32::NAN], 0.0),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(
        power_at_coefficient(&[1.0], f64::NAN),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(
        power_at_coefficient(&[1.0], 2.01),
        Err(SignalError::InvalidParameter)
    );
}

#[test]
fn explicit_frequency_coefficient_does_not_claim_sample_rate_mapping() {
    // This primitive accepts a recurrence coefficient only; no timestamp or
    // sampling-frequency claim is inferred from the samples.
    assert!(power_at_coefficient(&[0.0, 0.0, 0.0, 0.0], 1.0).is_ok());
}
