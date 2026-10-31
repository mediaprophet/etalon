use super::*;

#[test]
fn transform_four_real_samples_with_explicit_forward_twiddles() {
    let input = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let twiddles = [1.0, 0.0, 0.0, -1.0];
    let mut scratch = [0.0; 8];
    let mut output = [0.0; 8];
    assert_eq!(
        forward_into(&input, &twiddles, &mut scratch, &mut output),
        Ok(4)
    );
    assert_eq!(output, [10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0]);
}

#[test]
fn rejects_invalid_shape_without_touching_output() {
    let mut scratch = [0.0; 6];
    let mut output = [7.0; 6];
    assert_eq!(
        forward_into(&[0.0; 6], &[0.0; 3], &mut scratch, &mut output),
        Err(SignalError::InvalidParameter)
    );
    assert_eq!(output, [7.0; 6]);
}

#[test]
fn single_sample_requires_no_twiddles() {
    let mut scratch = [0.0; 2];
    let mut output = [0.0; 2];
    assert_eq!(
        forward_into(&[3.0, -2.0], &[], &mut scratch, &mut output),
        Ok(1)
    );
    assert_eq!(output, [3.0, -2.0]);
}

#[test]
fn transform_matches_reference_matrix_for_complex_input() {
    let input = [1.0, 2.0, -3.0, 1.0, 2.0, -1.0, 0.5, 4.0];
    let twiddles = [1.0, 0.0, 0.0, -1.0];
    let reference = [
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
    ];
    let mut scratch = [0.0; 8];
    let mut fft = [0.0; 8];
    let mut dft = [0.0; 8];
    forward_into(&input, &twiddles, &mut scratch, &mut fft).unwrap();
    crate::dft::transform_into(&input, &reference, &mut dft, crate::dft::DftScale::Raw).unwrap();
    for (actual, expected) in fft.iter().zip(dft) {
        assert!((f64::from(*actual) - expected).abs() < 1e-6);
    }
}
