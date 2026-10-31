use super::*;

#[test]
fn conjugates_reference_before_multiplication() {
    let result = correlate_iq(
        &[1.0, 0.0],
        &[0.0, 1.0],
        0,
        CorrelationScaling::RawSum,
    )
    .unwrap();
    assert_eq!(result.real(), 0.0);
    assert_eq!(result.imag(), 1.0);
    assert_eq!(result.overlap_samples(), 1);
}

#[test]
fn positive_lag_advances_observed_and_negative_lag_advances_reference() {
    let reference = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
    let observed = [10.0, 0.0, 20.0, 0.0, 30.0, 0.0];
    let positive = correlate_iq(
        &reference,
        &observed,
        1,
        CorrelationScaling::RawSum,
    )
    .unwrap();
    assert_eq!(positive.real(), 1.0 * 20.0 + 2.0 * 30.0);
    assert_eq!(positive.overlap_samples(), 2);

    let negative = correlate_iq(
        &reference,
        &observed,
        -1,
        CorrelationScaling::RawSum,
    )
    .unwrap();
    assert_eq!(negative.real(), 2.0 * 10.0 + 3.0 * 20.0);
    assert_eq!(negative.overlap_samples(), 2);
}

#[test]
fn scaling_is_explicit_raw_sum_or_overlap_mean() {
    let reference = [1.0, 0.0, 1.0, 0.0];
    let observed = [2.0, 0.0, 4.0, 0.0];
    let raw = correlate_iq(
        &reference,
        &observed,
        0,
        CorrelationScaling::RawSum,
    )
    .unwrap();
    let mean = correlate_iq(
        &reference,
        &observed,
        0,
        CorrelationScaling::MeanByOverlap,
    )
    .unwrap();
    assert_eq!(raw.real(), 6.0);
    assert_eq!(mean.real(), 3.0);
    assert_eq!(raw.imag(), mean.imag());
}

#[test]
fn rejects_empty_odd_and_out_of_range_inputs() {
    assert_eq!(
        correlate_iq(&[], &[1.0, 0.0], 0, CorrelationScaling::RawSum),
        Err(SignalError::EmptyInput)
    );
    assert_eq!(
        correlate_iq(&[1.0], &[1.0, 0.0], 0, CorrelationScaling::RawSum),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(
        correlate_iq(&[1.0, 0.0], &[1.0, 0.0], 1, CorrelationScaling::RawSum),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(
        correlate_iq(&[1.0, 0.0], &[1.0, 0.0], -1, CorrelationScaling::RawSum),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(
        correlate_iq(
            &[1.0, 0.0],
            &[1.0, 0.0],
            i64::MIN,
            CorrelationScaling::RawSum
        ),
        Err(SignalError::InvalidParameter)
    );
}

#[test]
fn rejects_non_finite_values_even_outside_overlap() {
    assert_eq!(
        correlate_iq(
            &[1.0, 0.0, f32::NAN, 0.0],
            &[2.0, 0.0],
            0,
            CorrelationScaling::RawSum
        ),
        Err(SignalError::NonFiniteInput)
    );
}

#[test]
fn accumulation_uses_f64_before_products_and_sum() {
    let max = f32::MAX;
    let result = correlate_iq(
        &[max, 0.0, max, 0.0],
        &[max, 0.0, max, 0.0],
        0,
        CorrelationScaling::RawSum,
    )
    .unwrap();
    assert!(result.real().is_finite());
    assert!(result.real() > f64::from(max));
    assert_eq!(result.imag(), 0.0);
}
