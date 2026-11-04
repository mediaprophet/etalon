#![cfg(feature = "inference-model")]
//! Independent scalar-oracle expectations; no production-generated expected values.
use etalon_inference::ordinary::qwen35::{
    execute_normalized, fixture::F32FixtureHost, Error, Host, WorkspaceLayout,
};
use etalon_model::inference::{qwen_roles::*, ModelHyperparams, TensorInfo, ARCH_QWEN35_MOE};
#[path = "../../etalon-inference-kernel/tests/support/qwen_numeric_alloc.rs"]
mod allocation;
#[path = "../../etalon-inference-kernel/tests/support/qwen_reference_fixture.rs"]
#[allow(dead_code, clippy::excessive_precision)]
mod fixture;
use fixture::*;
use std::cell::Cell;

fn close(a: &[f32], e: &[f32]) {
    assert_eq!(a.len(), e.len());
    for (i, (&a, &e)) in a.iter().zip(e).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
            "lane {i}: {a} != {e}"
        );
    }
}
fn tensor(values: &mut Vec<f32>, data: &[f32], cols: usize, rows: usize) -> TensorInfo {
    let info = TensorInfo {
        dims: [cols as u64, rows as u64, 0, 0],
        n_dims: if rows == 0 { 1 } else { 2 },
        ggml_type: 0,
        byte_offset: (values.len() * 4) as u64,
    };
    values.extend_from_slice(data);
    info
}
fn projection(values: &mut Vec<f32>, data: &[f32], cols: usize, rows: usize) -> Projection {
    Projection {
        weight: tensor(values, data, cols, rows),
        row_scale: None,
    }
}
fn setup(packing: Qwen35Packing) -> (Vec<f32>, Qwen35Roles) {
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
struct Provider<'a> {
    values: &'a [f32],
    out: TensorInfo,
    alog: TensorInfo,
    mode: u8,
    calls: Cell<usize>,
    scale_seen: Cell<bool>,
}
impl Host for Provider<'_> {
    fn admits(&self, p: &Projection) -> bool {
        if p.weight == self.out && self.mode == 5 {
            p.row_scale.is_some()
        } else {
            F32FixtureHost {
                values: self.values,
            }
            .admits(p)
        }
    }
    fn read(&self, t: &TensorInfo, out: &mut [f32]) -> Result<(), Error> {
        F32FixtureHost {
            values: self.values,
        }
        .read(t, out)?;
        if self.mode == 3 && *t == self.alog {
            out[3] = -100.;
        }
        Ok(())
    }
    fn gemm(&self, p: &Projection, input: &[f32], out: &mut [f32]) -> Result<(), Error> {
        self.calls.set(self.calls.get() + 1);
        if p.weight == self.out {
            match self.mode {
                1 => {
                    out[0] = 23.;
                    return Err(Error::Provider);
                }
                2 => {
                    out[0] = 23.;
                    return Ok(());
                }
                4 => {
                    out.fill(f32::MAX);
                    return Ok(());
                }
                5 => {
                    let scale = p.row_scale.as_ref().ok_or(Error::Provider)?;
                    assert_eq!(scale.dims[0], 3);
                    self.scale_seen.set(true);
                    let mut scales = [0.; 3];
                    F32FixtureHost {
                        values: self.values,
                    }
                    .read(scale, &mut scales)?;
                    // Exact E4M3 codes 0x38=1,0x40=2,0xb8=-1, with distinct row scales.
                    for (r, (&code, &scale)) in
                        [0x38u8, 0x40, 0xb8].iter().zip(scales.iter()).enumerate()
                    {
                        let sign = if code & 128 == 0 { 1. } else { -1. };
                        let weight = sign
                            * (1. + (code & 7) as f32 / 8.)
                            * 2f32.powi(((code >> 3) & 15) as i32 - 7);
                        out[r] = input.iter().map(|v| v * weight).sum::<f32>() * scale;
                    }
                    return Ok(());
                }
                _ => {}
            }
        }
        F32FixtureHost {
            values: self.values,
        }
        .gemm(p, input, out)
    }
}
fn provider<'a>(values: &'a [f32], roles: &Qwen35Roles, mode: u8) -> Provider<'a> {
    Provider {
        values,
        out: roles.out.weight,
        alog: roles.a_log,
        mode,
        calls: Cell::new(0),
        scale_seen: Cell::new(false),
    }
}
fn live_bits(hidden: &[f32], history: &[f32], state: &[f32]) -> Vec<u32> {
    hidden
        .iter()
        .chain(history)
        .chain(state)
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn separate_global_and_grouped_two_token_full_sequence_matches_float64_oracle() {
    for packing in [
        Qwen35Packing::Separate,
        Qwen35Packing::FusedGlobal,
        Qwen35Packing::FusedKeyHeadGrouped,
    ] {
        let (values, roles) = setup(packing);
        let host = provider(&values, &roles, 0);
        let mut workspace = vec![0.; WorkspaceLayout::checked(roles.geometry).unwrap().floats];
        let mut history = INITIAL_HISTORY;
        let mut state = INITIAL_STATE;
        for (normalized, expected_out, expected_history, expected_state) in [
            (&T0_HIDDEN, &T0_PROJECTED_OUTPUT, &T0_HISTORY, &T0_STATE),
            (&T1_HIDDEN, &T1_PROJECTED_OUTPUT, &T1_HISTORY, &T1_STATE),
        ] {
            // Deliberately distinct residual proves the API consumes prenormalized input.
            let mut hidden = [1.25, -0.75, 0.5];
            let expected = core::array::from_fn::<_, 3, _>(|i| hidden[i] + expected_out[i]);
            let guard = allocation::AllocGuard::begin("prepared learned projection sequence", true);
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
            close(&history, expected_history);
            close(&state, expected_state);
        }
    }
}

#[test]
fn late_provider_partial_success_late_head_and_residual_overflow_are_atomic() {
    for mode in 1..=4 {
        let (values, roles) = setup(Qwen35Packing::Separate);
        let host = provider(&values, &roles, mode);
        let mut hidden = if mode == 4 {
            [f32::MAX; 3]
        } else {
            [1.25, -0.75, 0.5]
        };
        let mut history = INITIAL_HISTORY;
        let mut state = INITIAL_STATE;
        if mode == 3 {
            for i in 18..24 {
                state[i] = if i < 21 { f32::MAX } else { -f32::MAX };
            }
        }
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
        assert_eq!(host.calls.get(), if mode == 3 { 4 } else { 5 });
    }
}

#[test]
fn fp8_output_row_scales_reach_provider_and_change_numerics() {
    let (mut values, mut roles) = setup(Qwen35Packing::Separate);
    roles.out.weight.ggml_type = etalon_model::ggml::GGML_TYPE_FP8_E4M3;
    roles.out.row_scale = Some(tensor(&mut values, &[0.5, 0.25, 2.], 3, 0));
    let host = provider(&values, &roles, 5);
    let mut history = INITIAL_HISTORY;
    let mut state = INITIAL_STATE;
    let mut hidden = [1.25, -0.75, 0.5];
    let sum = T0_GATED.iter().map(|v| *v as f64).sum::<f64>();
    let expected = [
        (1.25 + 0.5 * sum) as f32,
        (-0.75 + 0.5 * sum) as f32,
        (0.5 - 2. * sum) as f32,
    ];
    let mut workspace = vec![0.; WorkspaceLayout::checked(roles.geometry).unwrap().floats];
    let guard = allocation::AllocGuard::begin("prepared scaled provider", true);
    execute_normalized(
        &host,
        &roles,
        &T0_HIDDEN,
        &mut hidden,
        &mut history,
        &mut state,
        1e-6,
        &mut workspace,
    )
    .unwrap();
    assert!(guard.check().is_ok());
    assert!(host.scale_seen.get());
    close(&hidden, &expected);
    close(&state, &T0_STATE);
    close(&history, &T0_HISTORY);
}
