use etalon_inference_kernel::recurrent::gated_deltanet::step_gated_deltanet;
use etalon_inference_kernel::recurrent::*;

#[test]
fn new_decay_order_is_explicit_and_legacy_is_unchanged() {
    let mut legacy = [4.0, 8.0, 2.0, 6.0];
    let mut qwen = legacy;
    let mut old_out = [0.0; 2];
    let mut out = [0.0; 2];
    step_gated_deltanet(
        &mut legacy,
        &[1., 0.],
        &[1., 0.],
        &[3., 5.],
        &[0.5],
        &[0.25],
        2,
        2,
        &mut old_out,
    )
    .unwrap();
    step_qwen_delta(
        &mut qwen,
        &[1., 0.],
        &[1., 0.],
        &[3., 5.],
        0.5,
        0.25,
        2,
        2,
        &mut out,
    )
    .unwrap();
    assert_eq!(legacy, [1.75, 3.25, 1., 3.]);
    assert_eq!(old_out, [1.75, 3.25]);
    assert_eq!(qwen, [2.25, 4.25, 1., 3.]);
    assert_eq!(out, [2.25, 4.25]);
}

#[test]
fn recurrence_rejects_all_admission_failures_atomically() {
    for (decay, beta, q) in [
        (f32::NAN, 0.5, 1.),
        (0.5, f32::INFINITY, 1.),
        (-0.1, 0.5, 1.),
        (0.5, 1.1, 1.),
        (0.5, 0.5, f32::NAN),
    ] {
        let mut state = [1., 2.];
        let mut out = [99., 99.];
        assert!(step_qwen_delta(
            &mut state,
            &[q],
            &[1.],
            &[1., 2.],
            decay,
            beta,
            1,
            2,
            &mut out
        )
        .is_err());
        assert_eq!(state, [1., 2.]);
        assert_eq!(out, [99., 99.]);
    }
    let mut state = [1., 2.];
    let mut out = [99., 99.];
    assert_eq!(
        step_qwen_delta(&mut state, &[], &[], &[], 0.5, 0.5, usize::MAX, 2, &mut out),
        Err(QwenDeltaError::SizeOverflow)
    );
    assert_eq!(state, [1., 2.]);
    assert_eq!(out, [99., 99.]);
    assert_eq!(
        step_qwen_delta(
            &mut state,
            &[1.],
            &[1.],
            &[1., 2.],
            0.5,
            0.5,
            0,
            2,
            &mut out
        ),
        Err(QwenDeltaError::DimensionMismatch)
    );
    // The second column overflows only after the first would have succeeded.
    state = [1., f32::MAX];
    assert_eq!(
        step_qwen_delta(&mut state, &[1.], &[2.], &[1., 2.], 1., 1., 1, 2, &mut out),
        Err(QwenDeltaError::NonFinite)
    );
    assert_eq!(state, [1., f32::MAX]);
    assert_eq!(out, [99., 99.]);
}

#[test]
fn stable_extreme_gates_and_silu_are_finite() {
    let mut decay = [9.; 2];
    let mut beta = [9.; 2];
    qwen_gates(
        &[100., -100.],
        &[f32::MAX, -f32::MAX],
        &[0., 0.],
        &[0., 0.],
        &mut decay,
        &mut beta,
    )
    .unwrap();
    assert!(decay
        .iter()
        .all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
    assert_eq!(beta, [1., 0.]);
    let mut out = [9.; 2];
    silu_into(&[f32::MAX, -f32::MAX], &mut out).unwrap();
    assert_eq!(out[0], f32::MAX);
    assert_eq!(out[1], 0.);
    let before = decay;
    assert_eq!(
        qwen_gates(
            &[0., 1.],
            &[0., 0.],
            &[0., f32::MAX],
            &[0., 0.],
            &mut decay,
            &mut beta
        ),
        Err(QwenDeltaError::NonFinite)
    );
    assert_eq!(decay, before);
    assert_eq!(beta, [1., 0.]);
}

fn layout() -> QwenHeadLayout {
    QwenHeadLayout {
        key_heads: 1,
        value_heads: 2,
        key_dim: 1,
        value_dim: 1,
    }
}

#[test]
fn projection_helpers_reject_without_partial_writes() {
    let mut qkv = [99.; 4];
    let mut z = [99.; 2];
    let mut b = [99.; 2];
    let mut a = [99.; 2];
    assert!(unpack_qwen_projections(
        layout(),
        QwenProjectionPacking::GlobalQkvzBa,
        &[1., 2., 3., 4., 5., f32::NAN],
        &[1.; 4],
        &mut qkv,
        &mut z,
        &mut b,
        &mut a
    )
    .is_err());
    assert_eq!(qkv, [99.; 4]);
    assert_eq!(z, [99.; 2]);
    assert_eq!(b, [99.; 2]);
    assert_eq!(a, [99.; 2]);
    let mut q = [99.; 2];
    let mut k = [99.; 2];
    assert_eq!(
        normalize_qwen_qk(layout(), &[1., f32::MAX, 0., 0.], &mut q, &mut k),
        Err(QwenDeltaError::NonFinite)
    );
    assert_eq!(q, [99.; 2]);
    assert_eq!(k, [99.; 2]);
    let invalid = QwenHeadLayout {
        key_heads: 3,
        ..layout()
    };
    assert!(normalize_qwen_qk(invalid, &[], &mut q, &mut k).is_err());
    let overflow = QwenHeadLayout {
        key_heads: usize::MAX,
        value_heads: usize::MAX,
        key_dim: 2,
        value_dim: 1,
    };
    assert_eq!(
        normalize_qwen_qk(overflow, &[], &mut q, &mut k),
        Err(QwenDeltaError::SizeOverflow)
    );
    let mut out = [99.; 2];
    for eps in [0., -1., f32::NAN, f32::INFINITY] {
        assert_eq!(
            gated_rms_norm(&[1., 2.], &[1., 2.], &[1., 1.], eps, &mut out),
            Err(QwenDeltaError::InvalidEpsilon)
        );
        assert_eq!(out, [99.; 2]);
    }
    assert_eq!(
        gated_rms_norm(&[1., 2.], &[1., f32::MAX], &[1., f32::MAX], 1e-6, &mut out),
        Err(QwenDeltaError::NonFinite)
    );
    assert_eq!(out, [99.; 2]);
    assert_eq!(
        silu_into(&[1., f32::NAN], &mut out),
        Err(QwenDeltaError::NonFinite)
    );
    assert_eq!(out, [99.; 2]);
}
