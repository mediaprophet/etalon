
#[test]
fn metal_mega_pass_platform_guard() {
    if cfg!(not(target_os = "macos")) {
        assert!(!metal_mega_pass_available());
        assert_eq!(metal_lane_capability(), MetalCapabilityState::Unavailable);
        assert_eq!(metal_lane_capability_state(), "unavailable");
        assert!(try_metal_mega_pass(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            &[],
            &[],
            &[],
            None,
            None,
            0,
            0
        )
        .is_none());
    }
}

#[test]
fn execute_metal_mega_pass_two_layer_end_to_end() {
    clear_metal_kv_cache();
    let mut eye4 = [0.0f32; 16];
    for i in 0..4 {
        eye4[i * 4 + i] = 1.0;
    }
    let eye_b: &[u8] = bytemuck::cast_slice(&eye4);
    let kv_w = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let kv_b: &[u8] = bytemuck::cast_slice(&kv_w);
    let norm_w = [1.0f32; 4];
    let mk_lw = || MetalPassLayerWeights {
        attn_norm: &norm_w,
        q_raw: eye_b,
        k_raw: kv_b,
        v_raw: kv_b,
        o_raw: eye_b,
        ffn_norm: &norm_w,
        gate_raw: eye_b,
        up_raw: eye_b,
        down_raw: eye_b,
    };
    let mk_dims = || MetalPassLayerDims {
        q_in: 4,
        q_out: 4,
        kv_in: 4,
        kv_out: 2,
        o_in: 4,
        o_out: 4,
        gate_in: 4,
        gate_out: 4,
        up_in: 4,
        up_out: 4,
        down_in: 4,
        down_out: 4,
    };
    let (layers, dims) = ([mk_lw(), mk_lw()], [mk_dims(), mk_dims()]);
    let mut lm_head = [0.0f32; 24];
    lm_head[4 * 4] = 10.0;
    let lm_b: &[u8] = bytemuck::cast_slice(&lm_head);
    let hidden = [2.0f32, 0.5, 0.1, 0.1];
    for pos in 0..2 {
        let tok = execute_metal_mega_pass(
            4,
            2,
            1,
            2,
            2,
            pos,
            8,
            16,
            2,
            10_000.0,
            1.0,
            1e-5,
            &hidden,
            &layers,
            &dims,
            Some(&norm_w),
            Some(lm_b),
            4,
            6,
        );
        assert_eq!(tok, Some(4));
    }
}

#[test]
fn project_weight_bytes_q8_0_and_q4_k() {
    let mut q8 = [0u8; Q8_0_BLOCK_BYTES];
    let one = half::f16::from_f32(1.0).to_le_bytes();
    q8[0] = one[0];
    q8[1] = one[1];
    for i in 0..32 {
        q8[2 + i] = 2u8;
    }
    let x = [1.0f32; 32];
    let mut out = [0.0f32; 1];
    assert!(project_weight_bytes(&q8, 32, 1, &x, &mut out));
    assert!((out[0] - 64.0).abs() < 1e-3);

    let mut q4k = [0u8; Q4_K_BLOCK_BYTES];
    q4k[0] = one[0];
    q4k[1] = one[1];
    for i in 4..Q4_K_BLOCK_BYTES {
        q4k[i] = 0x33;
    }
    assert!(project_weight_bytes(&q4k, 32, 1, &x, &mut out));
    assert!((out[0] - 96.0).abs() < 1e-3);
}

#[test]
fn metal_prefetch_rejects_empty_and_overflow_inputs() {
    assert!(!metal_double_buffered_prefetch(&[], 0));
    assert!(!metal_double_buffered_prefetch(&[1, 2, 3, 4], u64::MAX));
    assert!(!metal_double_buffered_prefetch(
        &[1, 2, 3, 4],
        METAL_PREFETCH_STAGING_CAPACITY
    ));
}
use super::*;
