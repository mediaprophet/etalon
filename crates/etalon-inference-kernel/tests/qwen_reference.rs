//! Float64 oracle comparison; bounded fixture tolerance fixed before execution.
use etalon_inference_kernel::recurrent::gated_deltanet::step_causal_conv1d;
use etalon_inference_kernel::recurrent::{qwen_delta::*, qwen_projection::*};
#[path = "support/qwen_numeric_alloc.rs"]
mod allocation;
#[path = "support/qwen_reference_fixture.rs"]
#[allow(dead_code, clippy::excessive_precision)]
mod fixture;
use fixture::*;

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
            "lane {i}: {a} != {e}"
        );
    }
}
fn layout() -> QwenHeadLayout {
    QwenHeadLayout {
        key_heads: 2,
        value_heads: 4,
        key_dim: 2,
        value_dim: 3,
    }
}

#[test]
fn independent_two_token_rectangular_recurrence_and_intermediates() {
    let mut state = INITIAL_STATE;
    let mut history = INITIAL_HISTORY;
    let tokens = [
        (
            &T0_RAW,
            &T0_Z,
            &T0_A,
            &T0_B,
            &T0_CONV,
            &T0_Q,
            &T0_K,
            &T0_V,
            &T0_DECAY,
            &T0_BETA,
            &T0_STATE,
            &T0_HISTORY,
            &T0_Y,
            &T0_GATED,
        ),
        (
            &T1_RAW,
            &T1_Z,
            &T1_A,
            &T1_B,
            &T1_CONV,
            &T1_Q,
            &T1_K,
            &T1_V,
            &T1_DECAY,
            &T1_BETA,
            &T1_STATE,
            &T1_HISTORY,
            &T1_Y,
            &T1_GATED,
        ),
    ];
    for (raw, z, a, b, ec, eq, ek, ev, ed, eb, es, eh, ey, eg) in tokens {
        let mut conv = [0.0; 20];
        let mut q = [0.0; 8];
        let mut k = [0.0; 8];
        let mut decay = [0.0; 4];
        let mut beta = [0.0; 4];
        let mut y = [0.0; 12];
        let mut gated = [0.0; 12];
        let guard = allocation::AllocGuard::begin("qwen independent pipeline", true);
        step_causal_conv1d(&mut history, &TAPS, raw, &mut conv).unwrap();
        let preactivation = conv;
        silu_into(&preactivation, &mut conv).unwrap();
        normalize_qwen_qk(layout(), &conv, &mut q, &mut k).unwrap();
        qwen_gates(a, b, &ALOG, &DT, &mut decay, &mut beta).unwrap();
        for h in 0..4 {
            step_qwen_delta(
                &mut state[h * 6..h * 6 + 6],
                &q[h * 2..h * 2 + 2],
                &k[h * 2..h * 2 + 2],
                &conv[8 + h * 3..11 + h * 3],
                decay[h],
                beta[h],
                2,
                3,
                &mut y[h * 3..h * 3 + 3],
            )
            .unwrap();
        }
        for h in 0..4 {
            gated_rms_norm(
                &y[h * 3..h * 3 + 3],
                &z[h * 3..h * 3 + 3],
                &GAMMA,
                1e-6,
                &mut gated[h * 3..h * 3 + 3],
            )
            .unwrap();
        }
        assert!(guard.check().is_ok());
        close(&conv, ec);
        close(&q, eq);
        close(&k, ek);
        close(&conv[8..], ev);
        close(&decay, ed);
        close(&beta, eb);
        close(&state, es);
        close(&history, eh);
        close(&y, ey);
        close(&gated, eg);
    }
}

#[test]
fn explicit_global_and_grouped_fusion_are_distinct_permutations() {
    let global: [f32; 32] = core::array::from_fn(|i| i as f32 + 1.0);
    let ba = [101., 102., 103., 104., 201., 202., 203., 204.];
    let mut grouped = [0.0; 32];
    let mut grouped_ba = [0.0; 8];
    for h in 0..2 {
        let base = h * 16;
        grouped[base..base + 2].copy_from_slice(&global[h * 2..h * 2 + 2]);
        grouped[base + 2..base + 4].copy_from_slice(&global[4 + h * 2..6 + h * 2]);
        grouped[base + 4..base + 10].copy_from_slice(&global[8 + h * 6..14 + h * 6]);
        grouped[base + 10..base + 16].copy_from_slice(&global[20 + h * 6..26 + h * 6]);
        grouped_ba[h * 4..h * 4 + 2].copy_from_slice(&ba[h * 2..h * 2 + 2]);
        grouped_ba[h * 4 + 2..h * 4 + 4].copy_from_slice(&ba[4 + h * 2..6 + h * 2]);
    }
    assert_ne!(grouped, global);
    for (packing, x, gates) in [
        (QwenProjectionPacking::GlobalQkvzBa, &global, &ba),
        (
            QwenProjectionPacking::GroupedByKeyHead,
            &grouped,
            &grouped_ba,
        ),
    ] {
        let mut qkv = [0.; 20];
        let mut z = [0.; 12];
        let mut b = [0.; 4];
        let mut a = [0.; 4];
        let guard = allocation::AllocGuard::begin("unpack", true);
        unpack_qwen_projections(
            layout(),
            packing,
            x,
            gates,
            &mut qkv,
            &mut z,
            &mut b,
            &mut a,
        )
        .unwrap();
        assert!(guard.check().is_ok());
        assert_eq!(qkv, global[..20]);
        assert_eq!(z, global[20..]);
        assert_eq!(b, ba[..4]);
        assert_eq!(a, ba[4..]);
    }
}

#[test]
fn delta_invalid_inputs_preserve_state_and_output_bits() {
    for case in 0..10 {
        let mut state = INITIAL_STATE[..6].to_vec();
        let mut out = [f32::from_bits(0x7fc01234), -0.0, 97.];
        let before: Vec<_> = state.iter().map(|x| x.to_bits()).collect();
        let out_before = out.map(f32::to_bits);
        let mut q = [0.2, 0.3];
        let mut k = [0.4, 0.5];
        let mut v = [0.6, 0.7, 0.8];
        let mut decay = 0.7;
        let mut beta = 0.4;
        match case {
            0 => q[1] = f32::NAN,
            1 => k[0] = f32::INFINITY,
            2 => v[2] = f32::NAN,
            3 => decay = -0.1,
            4 => beta = 1.1,
            5 => decay = f32::NAN,
            6 => beta = f32::INFINITY,
            _ => {}
        }
        let result = step_qwen_delta(
            &mut state,
            &q,
            &k,
            &v,
            decay,
            beta,
            if case == 7 {
                3
            } else if case == 8 {
                0
            } else if case == 9 {
                usize::MAX
            } else {
                2
            },
            3,
            &mut out,
        );
        assert!(result.is_err(), "case {case}");
        assert_eq!(
            state.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            before
        );
        assert_eq!(out.map(f32::to_bits), out_before);
    }
}

#[test]
fn invalid_projection_buffers_and_nonfinite_inputs_preserve_every_output() {
    for case in 0..4 {
        let mut x = [0.25; 32];
        let ba = [0.1; 8];
        let mut qkv = [91.; 20];
        let mut z = [92.; 12];
        let mut b = [93.; 4];
        let mut a = [94.; 4];
        if case == 0 {
            x[31] = f32::NAN;
        }
        let result = unpack_qwen_projections(
            if case == 3 {
                QwenHeadLayout {
                    value_heads: 3,
                    ..layout()
                }
            } else {
                layout()
            },
            QwenProjectionPacking::GlobalQkvzBa,
            if case == 1 { &x[..31] } else { &x },
            &ba,
            if case == 2 { &mut qkv[..19] } else { &mut qkv },
            &mut z,
            &mut b,
            &mut a,
        );
        assert!(result.is_err());
        assert_eq!(qkv, [91.; 20]);
        assert_eq!(z, [92.; 12]);
        assert_eq!(b, [93.; 4]);
        assert_eq!(a, [94.; 4]);
    }
}

#[test]
fn invalid_gates_and_norm_preserve_all_output_bits() {
    for case in 0..6 {
        let mut a = [0.1; 4];
        let b = [0.2; 4];
        let mut alog = [0.3; 4];
        let dt = [0.4; 4];
        let mut decay = [-0.0; 4];
        let mut beta = [97.; 4];
        if case == 0 {
            a[3] = f32::NAN;
        }
        if case == 1 {
            alog[0] = f32::INFINITY;
        }
        if case < 3 {
            let r = qwen_gates(
                &a,
                if case == 2 { &b[..3] } else { &b },
                &alog,
                &dt,
                &mut decay,
                &mut beta,
            );
            assert!(r.is_err());
            assert_eq!(decay.map(f32::to_bits), [0x80000000; 4]);
            assert_eq!(beta, [97.; 4]);
        } else {
            let mut input = [0.1; 3];
            let z = [0.2; 3];
            let weight = [1.; 3];
            let mut out = [-0.0; 3];
            if case == 3 {
                input[2] = f32::NAN;
            }
            let r = gated_rms_norm(
                &input,
                &z,
                if case == 4 { &weight[..2] } else { &weight },
                if case == 5 { -1.0 } else { 1e-6 },
                &mut out,
            );
            assert!(r.is_err());
            assert_eq!(out.map(f32::to_bits), [0x80000000; 3]);
        }
    }
}
