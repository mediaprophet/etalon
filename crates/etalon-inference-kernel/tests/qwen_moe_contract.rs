//! Independent arithmetic expectations; no full-model or GPU qualification.
use etalon_inference_kernel::moe::{nvfp4::*, routing::*};
use etalon_inference_kernel::recurrent::gated_deltanet::*;
#[path = "support/qwen_moe_alloc.rs"]
mod allocation;

#[test]
fn routing_ties_selected_normalization_and_output_tails() {
    let mut ids = [999; 5];
    let mut weights = [-9.0; 5];
    assert_eq!(
        route_topk(&[2.0, 5.0, 5.0, -1.0], 3, false, &mut ids, &mut weights),
        Ok(3)
    );
    assert_eq!(ids, [1, 2, 0, 999, 999]);
    for (actual, expected) in weights[..3]
        .iter()
        .zip([5.0 / 12.0, 5.0 / 12.0, 2.0 / 12.0])
    {
        assert!((actual - expected).abs() < 1e-7);
    }
    route_topk(&[7.0, 7.0, 7.0, 7.0], 2, true, &mut ids, &mut weights).unwrap();
    assert_eq!(&ids[..2], &[0, 1]);
    assert_eq!(&weights[..2], &[0.5, 0.5]);
    assert_eq!(&weights[3..], &[-9.0; 2]);
    route_topk(&[-2.0, -1.0], 2, false, &mut ids, &mut weights).unwrap();
    assert_eq!(&ids[..2], &[1, 0]);
    assert_eq!(&weights[..2], &[0.0, 0.0]);
}

#[test]
fn routing_refusal_order_and_preserved_legacy_edges() {
    let mut ids = [91; 2];
    let mut weights = [92.0; 2];
    assert_eq!(
        route_topk(&[], 17, true, &mut ids, &mut weights),
        Err(MoeError::TopKExceedsCapacity)
    );
    assert_eq!(
        route_topk(&[], 3, true, &mut ids, &mut weights),
        Err(MoeError::BufferTooSmall)
    );
    assert_eq!(
        route_topk(&[], 2, true, &mut ids, &mut weights),
        Err(MoeError::InsufficientExperts)
    );
    assert_eq!(ids, [91; 2]);
    assert_eq!(weights, [92.0; 2]);
    assert_eq!(route_topk(&[], 0, true, &mut ids, &mut weights), Ok(0));
    assert!(std::panic::catch_unwind(|| route_topk(&[1.0], 0, false, &mut [], &mut [])).is_err());
    // Existing u16 index projection is preserved, not endorsed as safe admission.
    let mut logits = vec![0.0; 65_537];
    logits[65_536] = 1.0;
    route_topk(&logits, 1, false, &mut ids, &mut weights).unwrap();
    assert_eq!(ids[0], 0);
    route_topk(&[f32::NAN, 3.0], 1, false, &mut ids, &mut weights).unwrap();
    assert_eq!(ids[0], 1);
    route_topk(&[f32::INFINITY], 1, true, &mut ids, &mut weights).unwrap();
    assert!(weights[0].is_nan());
}

#[test]
fn expert_combination_exposes_existing_partial_failure_and_zero_weight_skip() {
    let mut out = [99.0; 3];
    combine_expert_outputs(&[&[2.0, 4.0], &[-2.0, 8.0]], &[0.25, 0.75], 2, &mut out).unwrap();
    assert_eq!(out, [-1.0, 7.0, 99.0]);
    assert_eq!(
        combine_expert_outputs(&[&[]], &[], 4, &mut out),
        Err(MoeError::BufferTooSmall)
    );
    assert_eq!(
        combine_expert_outputs(&[&[]], &[], 2, &mut out),
        Err(MoeError::DimensionMismatch)
    );
    assert_eq!(out, [-1.0, 7.0, 99.0]);
    assert_eq!(
        combine_expert_outputs(&[&[2.0, 4.0], &[]], &[0.5, 1.0], 2, &mut out),
        Err(MoeError::DimensionMismatch)
    );
    assert_eq!(out, [1.0, 2.0, 99.0]);
    combine_expert_outputs(&[&[]], &[0.0], 2, &mut out).unwrap();
    assert_eq!(out, [0.0, 0.0, 99.0]);
}

#[test]
fn fp8_special_values_and_packed_nibble_order_are_explicit() {
    for (byte, expected) in [
        (0x00, 0.0f32),
        (0x80, -0.0),
        (0x01, 1.0 / 512.0),
        (0x07, 7.0 / 512.0),
        (0x08, 1.0 / 64.0),
        (0x38, 1.0),
        (0xb8, -1.0),
        (0x7e, 448.0),
        (0xfe, -448.0),
        (0x7f, 0.0),
        (0xff, 0.0),
    ] {
        assert_eq!(fp8_e4m3_to_f32(byte).to_bits(), expected.to_bits());
    }
    let mut out = [99.0; 16];
    dequantize_nvfp4_block(
        &[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe],
        0x38,
        2.0,
        &mut out,
    );
    let expected: [f32; 16] = [
        0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
    ];
    assert_eq!(out.map(f32::to_bits), expected.map(f32::to_bits));
}

#[test]
fn nvfp4_two_rows_and_refusal_order_preserve_tails() {
    let mut packed = [0x22; 16];
    packed[8..].fill(0xaa);
    let input = [1.0; 17];
    let mut out = [99.0; 3];
    nvfp4_gemv_zero_heap(&input, &packed, &[0x38, 0x40], 0.5, 2, 16, &mut out).unwrap();
    assert_eq!(out, [8.0, -16.0, 99.0]);
    for (cols, rows, expected) in [
        (15, 9, Nvfp4Error::InvalidBlockAlignment),
        (16, 9, Nvfp4Error::BufferTooSmall),
        (16, 2, Nvfp4Error::DimensionMismatch),
    ] {
        assert_eq!(
            nvfp4_gemv_zero_heap(&input, &[], &[], 1.0, rows, cols, &mut out),
            Err(expected)
        );
        assert_eq!(out, [8.0, -16.0, 99.0]);
    }
    let mut row = [77.0; 18];
    assert_eq!(
        dequantize_nvfp4_row(&[0; 7], &[], 1.0, &mut row),
        Err(Nvfp4Error::InvalidBlockAlignment)
    );
    assert_eq!(
        dequantize_nvfp4_row(&[0; 8], &[], 1.0, &mut row),
        Err(Nvfp4Error::BufferTooSmall)
    );
    assert_eq!(row, [77.0; 18]);
    dequantize_nvfp4_row(&[0x22; 8], &[0x38], 3.0, &mut row).unwrap();
    assert_eq!(&row[..16], &[3.0; 16]);
    assert_eq!(&row[16..], &[77.0; 2]);
    nvfp4_gemv_zero_heap(&[], &[], &[], 1.0, 2, 0, &mut out).unwrap();
    assert_eq!(out, [0.0, 0.0, 99.0]);
}

#[test]
fn causal_convolution_history_and_unused_capacity_are_not_interleaved() {
    let mut state = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 91.0];
    let mut out = [92.0; 3];
    let weights = [1.0, 2.0, 4.0, 8.0, -1.0, 0.5, 0.0, 2.0];
    step_causal_conv1d(&mut state, &weights, &[4.0, 40.0], &mut out).unwrap();
    assert_eq!(state, [2.0, 3.0, 4.0, 20.0, 30.0, 40.0, 91.0]);
    assert_eq!(out, [49.0, 80.0, 92.0]);
    let before = state;
    assert_eq!(
        step_causal_conv1d(&mut state, &weights[..7], &[5.0, 50.0], &mut out),
        Err(QwenStateError::BufferTooSmall)
    );
    assert_eq!(state, before);
    assert_eq!(out, [49.0, 80.0, 92.0]);
    step_causal_conv1d(&mut state, &[], &[], &mut out).unwrap();
    assert_eq!(state, before);
}

#[test]
fn delta_rule_uses_old_prediction_then_decay_and_first_gate_only() {
    let mut state = [1.0, 2.0, 3.0, 4.0, 91.0];
    let mut out = [92.0; 3];
    step_gated_deltanet(
        &mut state,
        &[2.0, -1.0],
        &[1.0, 2.0],
        &[9.0, 14.0],
        &[0.5, f32::NAN],
        &[0.25, f32::NAN],
        2,
        2,
        &mut out,
    )
    .unwrap();
    assert_eq!(state, [1.0, 2.0, 2.5, 4.0, 91.0]);
    assert_eq!(out, [-0.5, 0.0, 92.0]);
    step_gated_deltanet(
        &mut state,
        &[1.0, 1.0],
        &[0.0, 1.0],
        &[1.0, 2.0],
        &[],
        &[],
        2,
        2,
        &mut out,
    )
    .unwrap();
    assert_eq!(state, [1.0, 2.0, 1.0, 2.0, 91.0]);
    assert_eq!(out, [2.0, 4.0, 92.0]);
    let before = state;
    assert_eq!(
        step_gated_deltanet(&mut state, &[], &[1.0], &[], &[], &[], 2, 2, &mut out),
        Err(QwenStateError::BufferTooSmall)
    );
    assert_eq!(state, before);
    assert_eq!(out, [2.0, 4.0, 92.0]);
    step_gated_deltanet(&mut state, &[], &[], &[1.0, 2.0], &[], &[], 0, 2, &mut out).unwrap();
    assert_eq!(state, before);
    assert_eq!(out, [0.0, 0.0, 92.0]);
}

#[test]
fn success_and_refusal_paths_allocate_nothing_after_admission() {
    let mut ids = [0; 2];
    let mut weights = [0.0; 2];
    let mut out = [0.0; 18];
    let mut state = [0.0; 6];
    allocation::assert_zero_alloc("three primitive families", || {
        route_topk(&[1.0, 2.0], 2, true, &mut ids, &mut weights).unwrap();
        assert!(route_topk(&[], 3, true, &mut ids, &mut weights).is_err());
        combine_expert_outputs(&[&[1.0, 2.0]], &[0.5], 2, &mut out).unwrap();
        assert!(combine_expert_outputs(&[&[]], &[1.0], 2, &mut out).is_err());
        dequantize_nvfp4_row(&[0x22; 8], &[0x38], 1.0, &mut out).unwrap();
        assert!(dequantize_nvfp4_row(&[0; 7], &[], 1.0, &mut out).is_err());
        nvfp4_gemv_zero_heap(&[1.0; 16], &[0x22; 8], &[0x38], 1.0, 1, 16, &mut out).unwrap();
        assert!(nvfp4_gemv_zero_heap(&[], &[], &[], 1.0, 1, 16, &mut out).is_err());
        step_causal_conv1d(&mut state, &[1.0; 8], &[1.0; 2], &mut out).unwrap();
        assert!(step_causal_conv1d(&mut state, &[], &[1.0; 2], &mut out).is_err());
        step_gated_deltanet(
            &mut state,
            &[1.0; 2],
            &[1.0; 2],
            &[1.0; 2],
            &[],
            &[],
            2,
            2,
            &mut out,
        )
        .unwrap();
        assert!(step_gated_deltanet(&mut state, &[], &[], &[], &[], &[], 2, 2, &mut out).is_err());
    });
}
