//! Independent arithmetic and buffer contracts, not checkpoint qualification.
use etalon_inference_kernel::qwen::*;
#[path = "support/qwen_numeric_alloc.rs"]
mod allocation;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn rms_hand_vector_uses_prefolded_gamma_and_preserves_tail() {
    let tail = f32::from_bits(0x7fc12345);
    let mut out = [91.0, 92.0, tail, -0.0];
    rms_norm_into(&[3.0, 4.0], &[2.0, 0.5], 3.5, &mut out).unwrap();
    assert_eq!(&out[..2], &[1.5, 0.5]);
    assert_eq!(out[2].to_bits(), tail.to_bits());
    assert_eq!(out[3].to_bits(), (-0.0f32).to_bits());
    rms_norm_into(&[1.0, 1.0], &[1.0, 1.0], 3.0, &mut out).unwrap();
    assert_eq!(&out[..2], &[0.5, 0.5]);
}

#[test]
fn group_norms_have_independent_streams_and_exact_tails() {
    let input = [3.0, 4.0, 0.0, 5.0];
    let gamma = [2.0, 0.5, 3.0, 2.0];
    let mut out = [99.0; 6];
    group_rms_norm_into(&input, 2, 2, &gamma, 3.5, &mut out).unwrap();
    assert_eq!(out, [1.5, 0.5, 0.0, 2.5, 99.0, 99.0]);
    let mut inplace = [3.0, 4.0, 0.0, 5.0, -0.0, f32::from_bits(0x7fc00071)];
    group_rms_norm_in_place(&mut inplace, 2, 2, &gamma, 3.5).unwrap();
    assert_eq!(&inplace[..4], &out[..4]);
    assert_eq!(inplace[4].to_bits(), 0x80000000);
    assert_eq!(inplace[5].to_bits(), 0x7fc00071);
}

#[test]
fn group_inplace_and_into_retain_distinct_float_operation_order() {
    let input = [0.3, 4.0];
    let gamma = [0.1, 1.0];
    let mut out = [0.0; 2];
    let mut inplace = input;
    group_rms_norm_into(&input, 1, 2, &gamma, 1e-6, &mut out).unwrap();
    group_rms_norm_in_place(&mut inplace, 1, 2, &gamma, 1e-6).unwrap();
    // Frozen original witness: (x * inverse) * gamma vs x * (inverse * gamma).
    assert_eq!(out[0].to_bits(), 0x3c2d4ab7);
    assert_eq!(inplace[0].to_bits(), 0x3c2d4ab8);
    assert_eq!(out[1].to_bits(), inplace[1].to_bits());
}

#[test]
fn mixing_average_and_injection_factor_two_have_hand_computed_values() {
    let input = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0];
    let mut out = [93.0; 5];
    mix_streams_into(&input, &[0.5, 1.0, 2.0, 1.0, 0.5, 0.0], 2, 3, &mut out).unwrap();
    assert_eq!(out, [2.75, 2.75, 3.0, 93.0, 93.0]);
    let mut state = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 97.0];
    inject_stream_update(
        &mut state,
        &[2.0, -4.0, 8.0],
        &[0.5, 0.25, 0.0, 1.0, -0.5, 0.125],
        2,
        3,
    )
    .unwrap();
    assert_eq!(state, [3.0, 0.0, 3.0, 9.0, 11.0, 13.0, 97.0]);
}

#[test]
fn all_short_group_buffers_and_overflow_refuse_before_mutation_or_epsilon() {
    let input = [1.0; 6];
    for short in 0..6 {
        for which in 0..3 {
            let mut out = [91.0; 8];
            let result = match which {
                0 => group_rms_norm_into(&input[..short], 2, 3, &input, f32::NAN, &mut out),
                1 => group_rms_norm_into(&input, 2, 3, &input[..short], f32::NAN, &mut out),
                _ => group_rms_norm_into(&input, 2, 3, &input, f32::NAN, &mut out[..short]),
            };
            assert_eq!(result, Err(QwenNumericError::BufferTooSmall));
            assert_eq!(out, [91.0; 8]);
        }
        let mut state = [93.0; 8];
        assert_eq!(
            group_rms_norm_in_place(&mut state[..short], 2, 3, &input, f32::NAN),
            Err(QwenNumericError::BufferTooSmall)
        );
        assert_eq!(state, [93.0; 8]);
        assert_eq!(
            group_rms_norm_in_place(&mut state, 2, 3, &input[..short], f32::NAN),
            Err(QwenNumericError::BufferTooSmall)
        );
        assert_eq!(state, [93.0; 8]);
    }
    let mut out = [97.0; 3];
    assert_eq!(
        group_rms_norm_into(&[], usize::MAX, 2, &[], f32::NAN, &mut out),
        Err(QwenNumericError::BufferTooSmall)
    );
    assert_eq!(
        group_rms_norm_in_place(&mut out, usize::MAX, 2, &[], f32::NAN),
        Err(QwenNumericError::BufferTooSmall)
    );
    assert_eq!(
        mix_streams_into(&[], &[], usize::MAX, 2, &mut out),
        Err(QwenNumericError::BufferTooSmall)
    );
    assert_eq!(
        inject_stream_update(&mut out, &[], &[], usize::MAX, 2),
        Err(QwenNumericError::BufferTooSmall)
    );
    assert_eq!(out, [97.0; 3]);
    for epsilon in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            rms_norm_into(&[1.0], &[], epsilon, &mut out),
            Err(QwenNumericError::BufferTooSmall)
        );
        assert_eq!(
            rms_norm_into(&[1.0], &[1.0], epsilon, &mut out),
            Err(QwenNumericError::InvalidEpsilon)
        );
        assert_eq!(
            group_rms_norm_into(&[1.0], 1, 1, &[1.0], epsilon, &mut out),
            Err(QwenNumericError::InvalidEpsilon)
        );
        assert_eq!(
            group_rms_norm_in_place(&mut out, 1, 1, &[1.0], epsilon),
            Err(QwenNumericError::InvalidEpsilon)
        );
        assert_eq!(out, [97.0; 3]);
    }
}

#[test]
fn empty_geometry_and_residual_refusal_semantics_are_preserved() {
    let mut out = [71.0; 4];
    rms_norm_into(&[], &[], 1.0, &mut out).unwrap();
    group_rms_norm_into(&[], 0, 3, &[], 1.0, &mut out).unwrap();
    group_rms_norm_in_place(&mut out, 3, 0, &[], 1.0).unwrap();
    assert_eq!(out, [71.0; 4]);
    mix_streams_into(&[], &[], 0, 3, &mut out).unwrap();
    assert_eq!(out, [0.0, 0.0, 0.0, 71.0]);
    assert_eq!(
        inject_stream_update(&mut out, &[], &[], 0, 3),
        Err(QwenNumericError::BufferTooSmall)
    );
    inject_stream_update(&mut out, &[1.0; 3], &[], 0, 3).unwrap();
    assert_eq!(out, [0.0, 0.0, 0.0, 71.0]);
    add_into(&[1.0, 4.0], &[3.0, -2.0, 99.0], &mut out).unwrap();
    assert_eq!(out, [4.0, 2.0, 0.0, 71.0]);
    let before = bits(&out);
    assert_eq!(
        add_assign(&mut out, &[1.0]),
        Err(QwenNumericError::BufferTooSmall)
    );
    assert_eq!(
        add_into(&[1.0, 2.0], &[1.0], &mut out),
        Err(QwenNumericError::BufferTooSmall)
    );
    assert_eq!(bits(&out), before);
    add_assign(&mut out[..2], &[0.5, 0.25, 100.0]).unwrap();
    assert_eq!(out, [4.5, 2.25, 0.0, 71.0]);
}

#[test]
fn mix_and_inject_short_slices_preserve_all_caller_state() {
    let input = [1.0; 6];
    for short in 0..6 {
        let mut out = [81.0; 8];
        assert_eq!(
            mix_streams_into(&input[..short], &input, 2, 3, &mut out),
            Err(QwenNumericError::BufferTooSmall)
        );
        assert_eq!(
            mix_streams_into(&input, &input[..short], 2, 3, &mut out),
            Err(QwenNumericError::BufferTooSmall)
        );
        assert_eq!(
            inject_stream_update(&mut out[..short], &input, &input, 2, 3),
            Err(QwenNumericError::BufferTooSmall)
        );
        assert_eq!(
            inject_stream_update(&mut out, &input, &input[..short], 2, 3),
            Err(QwenNumericError::BufferTooSmall)
        );
        if short < 3 {
            assert_eq!(
                mix_streams_into(&input, &input, 2, 3, &mut out[..short]),
                Err(QwenNumericError::BufferTooSmall)
            );
            assert_eq!(
                inject_stream_update(&mut out, &input[..short], &input, 2, 3),
                Err(QwenNumericError::BufferTooSmall)
            );
        }
        assert_eq!(out, [81.0; 8]);
    }
    assert_eq!(
        QwenNumericError::BufferTooSmall.to_string(),
        "Qwen numeric caller buffer is too small"
    );
    assert_eq!(
        QwenNumericError::InvalidEpsilon.to_string(),
        "Qwen RMSNorm epsilon must be finite and positive"
    );
}

#[test]
fn special_values_are_not_silently_sanitized() {
    let mut out = [99.0; 3];
    rms_norm_into(&[f32::INFINITY], &[1.0], 1.0, &mut out).unwrap();
    assert!(out[0].is_nan());
    rms_norm_into(&[-0.0], &[1.0], 1.0, &mut out).unwrap();
    assert_eq!(out[0].to_bits(), 0x80000000);
    mix_streams_into(&[f32::NAN], &[0.0], 1, 1, &mut out).unwrap();
    assert!(out[0].is_nan());
    assert_eq!(&out[1..], &[99.0, 99.0]);
}

#[test]
fn public_hot_successes_and_refusals_allocate_zero_with_positive_control() {
    let positive = allocation::AllocGuard::begin("positive", true);
    let bytes = vec![7u8; 89];
    std::hint::black_box(&bytes);
    drop(bytes);
    assert!(positive.check().is_err());
    let mut out = [1.0; 8];
    let input = [1.0; 8];
    let measured = allocation::AllocGuard::begin("all numeric operations", true);
    for _ in 0..64 {
        rms_norm_into(&input, &input, 1.0, &mut out).unwrap();
        add_assign(&mut out, &input).unwrap();
        add_into(&input, &input, &mut out).unwrap();
        group_rms_norm_into(&input, 2, 4, &input, 1.0, &mut out).unwrap();
        group_rms_norm_in_place(&mut out, 2, 4, &input, 1.0).unwrap();
        mix_streams_into(&input, &input, 2, 4, &mut out).unwrap();
        inject_stream_update(&mut out, &input, &input, 2, 4).unwrap();
        assert_eq!(
            rms_norm_into(&input, &input, 0.0, &mut out),
            Err(QwenNumericError::InvalidEpsilon)
        );
        assert_eq!(
            group_rms_norm_in_place(&mut out, usize::MAX, 2, &input, 1.0),
            Err(QwenNumericError::BufferTooSmall)
        );
    }
    assert!(measured.check().is_ok());
}
