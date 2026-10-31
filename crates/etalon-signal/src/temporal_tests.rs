use super::*;

#[test]
fn moving_average_uses_partial_warmup_and_continues_across_blocks() {
    let mut filter = MovingAverage::new(3).unwrap();
    let mut history = [0.0; 3];
    let mut output = [0.0; 2];

    assert_eq!(filter.warmup_remaining(), 3);
    assert_eq!(filter.process(&[3.0, 6.0], &mut output, &mut history), Ok(2));
    assert_eq!(output, [3.0, 4.5]);
    assert_eq!(filter.warmup_remaining(), 1);

    assert_eq!(filter.process(&[9.0, 12.0], &mut output, &mut history), Ok(2));
    assert_eq!(output, [6.0, 9.0]);
    assert_eq!(filter.warmup_remaining(), 0);
}

#[test]
fn partially_filled_history_evicts_old_values_in_order() {
    let mut filter = MovingAverage::new(4).unwrap();
    let mut history = [0.0; 4];
    let mut first = [0.0; 2];
    filter.process(&[1.0, 2.0], &mut first, &mut history).unwrap();
    let mut next = [0.0; 4];
    filter
        .process(&[3.0, 4.0, 5.0, 6.0], &mut next, &mut history)
        .unwrap();
    assert_eq!(next, [2.0, 2.5, 3.5, 4.5]);
}

#[test]
fn explicit_gap_reset_starts_a_fresh_warmup() {
    let mut filter = MovingAverage::new(3).unwrap();
    let mut history = [0.0; 3];
    let mut output = [0.0; 2];
    filter.process(&[2.0, 4.0], &mut output, &mut history).unwrap();

    filter.reset_after_gap(&mut history).unwrap();
    assert_eq!(filter.warmup_remaining(), 3);
    assert_eq!(history, [0.0; 3]);
    let mut one = [0.0; 1];
    filter.process(&[10.0], &mut one, &mut history).unwrap();
    assert_eq!(one, [10.0]);
    assert_eq!(filter.warmup_remaining(), 2);
}

#[test]
fn moving_average_failures_leave_state_output_and_history_unchanged() {
    let mut filter = MovingAverage::new(2).unwrap();
    let mut history = [0.0; 2];
    let mut output = [7.0; 2];
    let before_warmup = filter.warmup_remaining();
    let before_history = history;
    let before_output = output;

    assert_eq!(
        filter.process(&[1.0], &mut output, &mut history),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(filter.warmup_remaining(), before_warmup);
    assert_eq!(history, before_history);
    assert_eq!(output, before_output);

    assert_eq!(
        filter.process(&[1.0, f32::NAN], &mut output, &mut history),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(filter.warmup_remaining(), before_warmup);
    assert_eq!(history, before_history);
    assert_eq!(output, before_output);
}

#[test]
fn reset_rejects_wrong_history_length_transactionally() {
    let mut filter = MovingAverage::new(2).unwrap();
    let mut history = [1.0, 3.0];
    let mut output = [0.0; 2];
    filter.process(&[1.0, 3.0], &mut output, &mut history).unwrap();
    let before_warmup = filter.warmup_remaining();
    let mut short_history = [8.0];

    assert_eq!(
        filter.reset(&mut short_history),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(filter.warmup_remaining(), before_warmup);
    assert_eq!(short_history, [8.0]);

    let mut next_output = [0.0];
    filter
        .process(&[5.0], &mut next_output, &mut history)
        .unwrap();
    assert_eq!(next_output, [4.0]);
}

#[test]
fn baseline_subtraction_is_pointwise_and_caller_buffered() {
    let input = [4.0, 2.0, -3.0];
    let baseline = [1.0, 2.0, -5.0];
    let mut output = [0.0; 3];
    assert_eq!(subtract_baseline(&input, &baseline, &mut output), Ok(3));
    assert_eq!(output, [3.0, 0.0, 2.0]);
}

#[test]
fn baseline_subtraction_failure_does_not_publish_partial_output() {
    let input = [1.0, f32::NAN];
    let baseline = [0.0, 0.0];
    let mut output = [9.0; 2];
    assert_eq!(
        subtract_baseline(&input, &baseline, &mut output),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(output, [9.0; 2]);

    assert_eq!(
        subtract_baseline(&[f32::MAX], &[-f32::MAX], &mut output[..1]),
        Err(SignalError::NumericRange)
    );
    assert_eq!(output, [9.0; 2]);
}

#[test]
fn baseline_subtraction_checks_shape_before_writing() {
    let mut output = [5.0; 2];
    assert_eq!(
        subtract_baseline(&[1.0, 2.0], &[0.0], &mut output),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(output, [5.0; 2]);
}
