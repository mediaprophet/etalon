use super::*;

#[test]
fn prepared_mel_reuses_filterbank_and_preserves_output_tail() {
    let shape = MelShape { n_frames: 2, n_bins: 4, n_mel: 2 };
    let weights = [1., 1., 0., 0., 0., 0., 1., 1.];
    let prepared = PreparedMel::new(shape, &weights).unwrap();
    let mut out = [99.; 5];
    assert_eq!(prepared.apply_into(&[1.,2.,3.,4.,5.,6.,7.,8.], &mut out), Ok(4));
    assert_eq!(out, [3.,7.,11.,15.,99.]);
    prepared.apply_into(&[0.; 8], &mut out).unwrap();
    assert_eq!(out, [0.,0.,0.,0.,99.]);
}

#[test]
fn invalid_mel_shapes_fail_before_output_mutation() {
    let shape = MelShape { n_frames: 1, n_bins: 2, n_mel: 1 };
    let mut out = [99.];
    assert_eq!(mel_apply_into(&[1.], &[1.,1.], shape, &mut out), Err(SignalError::LengthMismatch));
    assert_eq!(out, [99.]);
    assert_eq!(mel_apply_into(&[1.,1.], &[1.], shape, &mut out), Err(SignalError::LengthMismatch));
    assert_eq!(out, [99.]);
    let huge = MelShape { n_frames: usize::MAX, n_bins: 2, n_mel: 1 };
    assert_eq!(mel_apply_into(&[], &[], huge, &mut out), Err(SignalError::NumericRange));
    assert_eq!(out, [99.]);
}

#[test]
fn zero_bins_preserves_scalar_oracle_zero_sum() {
    let shape = MelShape { n_frames: 1, n_bins: 0, n_mel: 2 };
    let mut out = [99.; 2];
    assert_eq!(mel_apply_into(&[], &[], shape, &mut out), Ok(2));
    assert_eq!(out, [0.; 2]);
}
