use super::*;

#[test]
fn unwraps_both_directions_across_the_principal_boundary() {
    let pi = core::f32::consts::PI;
    let mut state = PhaseState::uninitialized();
    let mut out = [0.0; 4];
    let input = [pi - 0.1, -pi + 0.1, -pi + 0.2, pi - 0.2];
    assert_eq!(
        unwrap_phase_into(&input, &[false; 4], &mut state, &mut out),
        Ok(4)
    );
    assert!(out[1] > out[0]);
    assert!(out[2] > out[1]);
    assert!(out[3] < out[2]);
    assert!((out[1] - (f64::from(pi) + 0.1)).abs() < 1e-6);
}

#[test]
fn state_carries_across_chunks_and_can_be_explicitly_initialized() {
    let pi = core::f32::consts::PI;
    let mut state = PhaseState::at(-pi + 0.1, f64::from(pi) + 0.1).unwrap();
    let mut out = [0.0; 2];
    unwrap_phase_into(&[pi - 0.1, -pi + 0.2], &[false; 2], &mut state, &mut out).unwrap();
    assert!(out[0] > f64::from(pi));
    assert!(out[1] > out[0]);
    assert_eq!(state.previous_unwrapped(), Some(out[1]));
}

#[test]
fn explicit_discontinuity_reanchors_the_phase() {
    let mut state = PhaseState::uninitialized();
    let mut out = [0.0; 4];
    let input = [0.2, 0.3, -2.5, -2.4];
    unwrap_phase_into(&input, &[false, false, true, false], &mut state, &mut out).unwrap();
    assert_eq!(out, [0.2, 0.3, -2.5, -2.4]);
}

#[test]
fn invalid_input_does_not_modify_state_or_output() {
    let mut state = PhaseState::at(0.5, 3.0).unwrap();
    let before = state;
    let mut out = [17.0; 2];
    assert_eq!(
        unwrap_phase_into(&[0.6, 4.0], &[false; 2], &mut state, &mut out),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(state, before);
    assert_eq!(out, [17.0; 2]);
}

#[test]
fn mismatched_lengths_leave_state_and_output_untouched() {
    let mut state = PhaseState::uninitialized();
    let mut out = [9.0; 2];
    assert_eq!(
        unwrap_phase_into(&[0.1, 0.2], &[false], &mut state, &mut out),
        Err(SignalError::LengthMismatch)
    );
    assert_eq!(state, PhaseState::uninitialized());
    assert_eq!(out, [9.0; 2]);
}

#[test]
fn empty_chunk_is_a_no_op_and_reset_clears_state() {
    let mut state = PhaseState::at(0.25, 4.0).unwrap();
    let mut output = [];
    assert_eq!(unwrap_phase_into(&[], &[], &mut state, &mut output), Ok(0));
    assert!(state.is_initialized());
    state.reset();
    assert!(!state.is_initialized());
}
