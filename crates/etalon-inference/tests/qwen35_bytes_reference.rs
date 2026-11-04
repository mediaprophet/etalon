#![cfg(feature = "inference-model")]
//! Independent scalar-oracle expectations; no production-generated expected values.
use etalon_inference::ordinary::qwen35::{
    execute_normalized,
    weights::{CpuByteHost, SliceBytes},
    Host, WorkspaceLayout,
};
use etalon_model::inference::{qwen_roles::*, ModelHyperparams, TensorInfo, ARCH_QWEN35_MOE};
#[path = "../../etalon-inference-kernel/tests/support/qwen_numeric_alloc.rs"]
mod allocation;
#[path = "../../etalon-inference-kernel/tests/support/qwen_reference_fixture.rs"]
#[allow(dead_code, clippy::excessive_precision)]
mod fixture;
use fixture::*;

fn close(a: &[f32], e: &[f32]) {
    assert_eq!(a.len(), e.len());
    for (i, (&a, &e)) in a.iter().zip(e).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
            "lane {i}: {a} != {e}"
        );
    }
}
fn tensor(values: &mut Vec<u8>, data: &[f32], cols: usize, rows: usize) -> TensorInfo {
    let info = TensorInfo {
        dims: [cols as u64, rows as u64, 0, 0],
        n_dims: if rows == 0 { 1 } else { 2 },
        ggml_type: 0,
        byte_offset: values.len() as u64,
    };
    for x in data {
        values.extend_from_slice(&x.to_le_bytes());
    }
    info
}
fn projection(values: &mut Vec<u8>, data: &[f32], cols: usize, rows: usize) -> Projection {
    Projection {
        weight: tensor(values, data, cols, rows),
        row_scale: None,
    }
}
fn setup(packing: Qwen35Packing) -> (Vec<u8>, Qwen35Roles) {
    let g = Qwen35Geometry::from_hyperparams(ModelHyperparams {
        architecture: ARCH_QWEN35_MOE,
        n_embd: 3,
        ssm_group_count: 2,
        ssm_time_step_rank: 4,
        ssm_state_size: 2,
        ssm_inner_size: 12,
        ssm_conv_kernel: 4,
        ..Default::default()
    })
    .unwrap();
    let mut values = Vec::new();
    let projections = if packing == Qwen35Packing::Separate {
        Qwen35Projections::Separate {
            qkv: projection(&mut values, &PROJ_WEIGHTS[..60], 3, 20),
            z: projection(&mut values, &PROJ_WEIGHTS[60..96], 3, 12),
            b: projection(&mut values, &PROJ_WEIGHTS[96..108], 3, 4),
            a: projection(&mut values, &PROJ_WEIGHTS[108..], 3, 4),
        }
    } else {
        let mut rows: Vec<usize> = (0..32).collect();
        let mut ba: Vec<usize> = (32..40).collect();
        if packing == Qwen35Packing::FusedKeyHeadGrouped {
            rows.clear();
            ba.clear();
            for h in 0..2 {
                rows.extend(h * 2..h * 2 + 2);
                rows.extend(4 + h * 2..6 + h * 2);
                rows.extend(8 + h * 6..14 + h * 6);
                rows.extend(20 + h * 6..26 + h * 6);
                ba.extend(32 + h * 2..34 + h * 2);
                ba.extend(36 + h * 2..38 + h * 2);
            }
        }
        let matrix = |rows: Vec<usize>| {
            rows.into_iter()
                .flat_map(|r| PROJ_WEIGHTS[r * 3..r * 3 + 3].iter().copied())
                .collect::<Vec<_>>()
        };
        Qwen35Projections::Fused {
            qkvz: projection(&mut values, &matrix(rows), 3, 32),
            ba: projection(&mut values, &matrix(ba), 3, 8),
            packing,
        }
    };
    let mut roles = Qwen35Roles {
        geometry: g,
        projections,
        conv: tensor(&mut values, &TAPS, 80, 0),
        a_log: tensor(&mut values, &ALOG, 4, 0),
        dt_bias: tensor(&mut values, &DT, 4, 0),
        norm: tensor(&mut values, &GAMMA, 3, 0),
        out: projection(&mut values, &OUT_WEIGHTS, 12, 3),
    };
    roles.conv.dims = [4, 1, 20, 0];
    roles.conv.n_dims = 3;
    (values, roles)
}

fn fp8_output(bytes: &mut Vec<u8>, roles: &mut Qwen35Roles) {
    roles.out.weight.byte_offset = bytes.len() as u64;
    roles.out.weight.ggml_type = etalon_model::ggml::GGML_TYPE_FP8_E4M3;
    for code in [0x38, 0x40, 0xb8] {
        bytes.extend_from_slice(&[code; 12]);
    }
    roles.out.row_scale = Some(tensor(bytes, &[0.5, 0.25, 2.], 3, 0));
}
fn live_bits(h: &[f32], history: &[f32], state: &[f32]) -> Vec<u32> {
    h.iter()
        .chain(history)
        .chain(state)
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn borrowed_f32_bytes_all_profiles_match_two_token_independent_oracle_zero_heap() {
    for packing in [
        Qwen35Packing::Separate,
        Qwen35Packing::FusedGlobal,
        Qwen35Packing::FusedKeyHeadGrouped,
    ] {
        let (bytes, roles) = setup(packing);
        let host = CpuByteHost {
            source: &SliceBytes { bytes: &bytes },
        };
        let mut history = INITIAL_HISTORY;
        let mut state = INITIAL_STATE;
        let mut workspace = vec![0.; WorkspaceLayout::checked(roles.geometry).unwrap().floats];
        for (normalized, eo, eh, es) in [
            (&T0_HIDDEN, &T0_PROJECTED_OUTPUT, &T0_HISTORY, &T0_STATE),
            (&T1_HIDDEN, &T1_PROJECTED_OUTPUT, &T1_HISTORY, &T1_STATE),
        ] {
            let mut hidden = [1.25, -0.75, 0.5];
            let expected = core::array::from_fn::<_, 3, _>(|i| hidden[i] + eo[i]);
            let guard = allocation::AllocGuard::begin("borrowed f32 byte sequence", true);
            execute_normalized(
                &host,
                &roles,
                normalized,
                &mut hidden,
                &mut history,
                &mut state,
                1e-6,
                &mut workspace,
            )
            .unwrap();
            assert!(guard.check().is_ok());
            close(&hidden, &expected);
            close(&history, eh);
            close(&state, es);
        }
    }
}

#[test]
fn real_fp8_byte_output_decodes_and_scales_weights_before_dot() {
    let (mut bytes, mut roles) = setup(Qwen35Packing::Separate);
    fp8_output(&mut bytes, &mut roles);
    let host = CpuByteHost {
        source: &SliceBytes { bytes: &bytes },
    };
    let mut history = INITIAL_HISTORY;
    let mut state = INITIAL_STATE;
    let mut workspace = vec![0.; WorkspaceLayout::checked(roles.geometry).unwrap().floats];
    for (normalized, gated, eh, es) in [
        (&T0_HIDDEN, &T0_GATED, &T0_HISTORY, &T0_STATE),
        (&T1_HIDDEN, &T1_GATED, &T1_HISTORY, &T1_STATE),
    ] {
        let mut hidden = [1.25, -0.75, 0.5];
        let sum = gated.iter().map(|v| *v as f64).sum::<f64>();
        let expected = [
            (1.25 + sum * 0.5) as f32,
            (-0.75 + sum * 0.5) as f32,
            (0.5 - sum * 2.) as f32,
        ];
        let guard = allocation::AllocGuard::begin("borrowed fp8 byte sequence", true);
        execute_normalized(
            &host,
            &roles,
            normalized,
            &mut hidden,
            &mut history,
            &mut state,
            1e-6,
            &mut workspace,
        )
        .unwrap();
        assert!(guard.check().is_ok());
        close(&hidden, &expected);
        close(&history, eh);
        close(&state, es);
    }
    // This overflow witness distinguishes weight scaling before DOT from output scaling.
    let mut witness = vec![0x40u8, 0x40];
    let scale = tensor(&mut witness, &[0.25], 1, 0);
    let p = Projection {
        weight: TensorInfo {
            dims: [2, 1, 0, 0],
            n_dims: 2,
            ggml_type: etalon_model::ggml::GGML_TYPE_FP8_E4M3,
            byte_offset: 0,
        },
        row_scale: Some(scale),
    };
    let host = CpuByteHost {
        source: &SliceBytes { bytes: &witness },
    };
    let mut out = [0.];
    host.gemm(&p, &[f32::MAX * 0.25; 2], &mut out).unwrap();
    assert!(out[0].is_finite());
    close(&out, &[f32::MAX * 0.25]);
}

#[test]
fn actual_half_bfloat_and_fp8_codes_have_independent_known_values() {
    for (kind, codes) in [
        (1u32, [0x3c00u16, 0xc000, 0x3800, 0x4200]),
        (30u32, [0x3f80u16, 0xc000, 0x3f00, 0x4040]),
    ] {
        let bytes: Vec<u8> = codes.into_iter().flat_map(u16::to_le_bytes).collect();
        let p = Projection {
            weight: TensorInfo {
                dims: [2, 2, 0, 0],
                n_dims: 2,
                ggml_type: kind,
                byte_offset: 0,
            },
            row_scale: None,
        };
        let host = CpuByteHost {
            source: &SliceBytes { bytes: &bytes },
        };
        let mut out = [0.; 2];
        host.gemm(&p, &[2., 4.], &mut out).unwrap();
        assert_eq!(out, [-6., 13.]);
        let mut decoded = [0.; 4];
        host.read(
            &TensorInfo {
                dims: [4, 0, 0, 0],
                n_dims: 1,
                ..p.weight
            },
            &mut decoded,
        )
        .unwrap();
        assert_eq!(decoded, [1., -2., 0.5, 3.]);
    }
    let mut bytes = vec![0, 1, 0x38, 0x40, 0x7e, 0xfe];
    let scale = tensor(&mut bytes, &[1.; 6], 6, 0);
    let p = Projection {
        weight: TensorInfo {
            dims: [1, 6, 0, 0],
            n_dims: 2,
            ggml_type: etalon_model::ggml::GGML_TYPE_FP8_E4M3,
            byte_offset: 0,
        },
        row_scale: Some(scale),
    };
    let mut out = [0.; 6];
    CpuByteHost {
        source: &SliceBytes { bytes: &bytes },
    }
    .gemm(&p, &[1.], &mut out)
    .unwrap();
    assert_eq!(out, [0., 1. / 512., 1., 2., 448., -448.]);
}

#[test]
fn bad_borrowed_ranges_nan_codes_and_scales_preserve_live_buffers() {
    for mode in 0..9 {
        let (mut bytes, mut roles) = setup(Qwen35Packing::Separate);
        fp8_output(&mut bytes, &mut roles);
        let out = roles.out.weight.byte_offset as usize;
        let scale = roles.out.row_scale.unwrap().byte_offset as usize;
        match mode {
            0 => bytes.truncate(out + 35),
            1 => roles.out.weight.byte_offset = u64::MAX,
            2 => bytes[out + 35] = 0x7f,
            3 => bytes[scale..scale + 4].copy_from_slice(&f32::NAN.to_le_bytes()),
            4 => bytes[scale..scale + 4].copy_from_slice(&0f32.to_le_bytes()),
            5 => bytes[scale..scale + 4].copy_from_slice(&(-1f32).to_le_bytes()),
            6 => roles.out.row_scale = None,
            7 => roles.out.row_scale.as_mut().unwrap().dims[0] = 2,
            8 => {
                let at = roles.conv.byte_offset as usize;
                bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
            _ => unreachable!(),
        }
        let host = CpuByteHost {
            source: &SliceBytes { bytes: &bytes },
        };
        let mut hidden = [1.25, -0.75, 0.5];
        let mut history = INITIAL_HISTORY;
        let mut state = INITIAL_STATE;
        let before = live_bits(&hidden, &history, &state);
        let mut workspace = vec![0.; WorkspaceLayout::checked(roles.geometry).unwrap().floats];
        assert!(
            execute_normalized(
                &host,
                &roles,
                &T0_HIDDEN,
                &mut hidden,
                &mut history,
                &mut state,
                1e-6,
                &mut workspace
            )
            .is_err(),
            "mode {mode}"
        );
        assert_eq!(live_bits(&hidden, &history, &state), before, "mode {mode}");
    }
}
