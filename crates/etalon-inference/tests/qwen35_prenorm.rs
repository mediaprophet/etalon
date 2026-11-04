#![cfg(feature = "inference-model")]

#[test]
fn prenorm_denominator_overflow_preserves_output_bits() {
    use etalon_inference::ordinary::qwen35::{zero_centered_rms_norm, Error};
    use etalon_inference_kernel::recurrent::QwenDeltaError;
    let input = [(f32::MAX * 0.5).sqrt()];
    assert!((input[0] * input[0]).is_finite());
    let sentinel = 0x8000_0000;
    let mut output = [f32::from_bits(sentinel)];
    assert_eq!(
        zero_centered_rms_norm(&input, &[0.0], f32::MAX, &mut output),
        Err(Error::Numerical(QwenDeltaError::NonFinite))
    );
    assert_eq!(output[0].to_bits(), sentinel);
}

#[test]
fn folded_norm_keeps_small_direct_multiplier_and_matches_raw_convention() {
    use etalon_inference::ordinary::qwen35::{ordinary_rms_norm, zero_centered_rms_norm};
    let mut raw = [0.0];
    let mut folded = [0.0];
    zero_centered_rms_norm(&[2.0], &[0.0], 1e-6, &mut raw).unwrap();
    ordinary_rms_norm(&[2.0], &[1.0], false, 1e-6, &mut folded).unwrap();
    assert_eq!(raw, folded);
    ordinary_rms_norm(&[2.0], &[1e-10], false, 1e-6, &mut folded).unwrap();
    assert!(folded[0] > 0.0);
    assert!(folded[0] < 2e-10);
}
