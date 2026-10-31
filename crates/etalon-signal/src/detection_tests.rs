use super::*;

#[test]
fn hysteresis_holds_state_between_thresholds_across_chunks() {
    let mut h = Hysteresis::new(2.0, 4.0, ThresholdState::Below).unwrap();
    let mut out = [None; 4];
    assert_eq!(h.process_into(&[1.0, 4.0, 3.0, 2.0], &mut out), Ok(2));
    assert_eq!(
        out[0].unwrap(),
        ThresholdTransition {
            sample_index: 1,
            state: ThresholdState::Above
        }
    );
    assert_eq!(
        out[1].unwrap(),
        ThresholdTransition {
            sample_index: 3,
            state: ThresholdState::Below
        }
    );
    assert_eq!(h.state(), ThresholdState::Below);
}

#[test]
fn invalid_input_does_not_change_state_or_output() {
    let mut h = Hysteresis::new(0.0, 1.0, ThresholdState::Below).unwrap();
    let marker = Some(ThresholdTransition {
        sample_index: 42,
        state: ThresholdState::Above,
    });
    let mut out = [marker; 2];
    assert_eq!(
        h.process_into(&[2.0, f32::NAN], &mut out),
        Err(SignalError::NonFiniteInput)
    );
    assert_eq!(h.state(), ThresholdState::Below);
    assert_eq!(out, [marker; 2]);
}
