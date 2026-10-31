use super::summarize;
use crate::SignalError;

#[test]
fn summarizes_values_without_allocating_or_normalizing_them() {
    let result = summarize(&[-1.0, 0.0, 1.0]).unwrap();
    assert_eq!(result.sample_count(), 3);
    assert_eq!(result.mean(), 0.0);
    assert_eq!(result.mean_square(), 2.0 / 3.0);
    assert_eq!(result.variance(), 2.0 / 3.0);
    assert_eq!(result.minimum(), -1.0);
    assert_eq!(result.maximum(), 1.0);
}

#[test]
fn rejects_empty_and_nonfinite_input() {
    assert_eq!(summarize(&[]), Err(SignalError::EmptyInput));
    assert_eq!(
        summarize(&[0.0, f32::NAN]),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(
        summarize(&[f32::INFINITY]),
        Err(SignalError::NonFiniteInput)
    );
}

#[test]
fn finite_f32_extremes_keep_finite_wide_power() {
    let result = summarize(&[f32::MAX, -f32::MAX]).unwrap();
    assert_eq!(result.mean(), 0.0);
    assert!(result.mean_square().is_finite());
    assert!(result.variance().is_finite());
}
