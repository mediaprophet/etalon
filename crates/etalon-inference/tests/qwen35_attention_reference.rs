#![cfg(feature = "inference-model")]
use etalon_inference::ordinary::qwen35::{
    fixture::F32FixtureHost,
    full_attention::{execute_normalized, TextRope, WorkspaceLayout},
    ordinary_rms_norm, zero_centered_rms_norm, Error, Host,
};
use etalon_model::inference::{qwen_attention::*, qwen_roles::Projection, TensorInfo};
#[path = "../../etalon-inference-kernel/tests/support/qwen_numeric_alloc.rs"]
mod allocation;
#[path = "support/qwen_attention_fixture.rs"]
#[allow(dead_code, clippy::excessive_precision)]
mod fixture;
use fixture::*;
use std::cell::Cell;

fn close(a: &[f32], expected: &[f32]) {
    assert_eq!(a.len(), expected.len());
    for (i, (&a, &e)) in a.iter().zip(expected).enumerate() {
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
fn setup(folded: bool) -> (Vec<f32>, Qwen35AttentionRoles) {
    let mut data = Vec::new();
    let qn = Q_NORM.map(|x| if folded { 1.0 + x } else { x });
    let kn = K_NORM.map(|x| if folded { 1.0 + x } else { x });
    let roles = Qwen35AttentionRoles {
        geometry: Qwen35AttentionGeometry {
            hidden: 3,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 6,
            rotary_dim: 4,
        },
        q_gate: projection(&mut data, &Q_WEIGHT, 3, 24),
        k: projection(&mut data, &K_WEIGHT, 3, 6),
        v: projection(&mut data, &V_WEIGHT, 3, 6),
        out: projection(&mut data, &OUT_WEIGHT, 12, 3),
        q_norm: tensor(&mut data, &qn, 6, 0),
        k_norm: tensor(&mut data, &kn, 6, 0),
        norm: if folded {
            QkNorm::FoldedMultiplier
        } else {
            QkNorm::RawZeroCentered
        },
    };
    (data, roles)
}
const FREQ: [f32; 2] = [1.0, 0.01];

#[test]
fn three_token_gqa_partial_rope_and_raw_folded_norm_match_pinned_oracle() {
    let mut raw_results = [[0.0; 3]; 3];
    for folded in [false, true] {
        let (data, roles) = setup(folded);
        let host = F32FixtureHost { values: &data };
        let layout = WorkspaceLayout::checked(roles.geometry, 3).unwrap();
        let mut scratch = vec![0.0; layout.floats];
        let (mut keys, mut values, mut cursor) = ([0.0; 18], [0.0; 18], 0);
        for (position, input, normalized, expected, key, value) in [
            (0, HIDDEN_0, NORMALIZED_0, RESIDUAL_0, KEY_0, VALUE_0),
            (1, HIDDEN_1, NORMALIZED_1, RESIDUAL_1, KEY_1, VALUE_1),
            (2, HIDDEN_2, NORMALIZED_2, RESIDUAL_2, KEY_2, VALUE_2),
        ] {
            let mut actual_norm = [0.0; 3];
            zero_centered_rms_norm(&input, &INPUT_NORM, 1e-6, &mut actual_norm).unwrap();
            close(&actual_norm, &normalized);
            let mut hidden = input;
            let guard = allocation::AllocGuard::begin("full-attention", true);
            let result = execute_normalized(
                &host,
                &roles,
                &actual_norm,
                &mut hidden,
                &mut keys,
                &mut values,
                &mut cursor,
                TextRope {
                    position,
                    inv_freq: &FREQ,
                },
                1e-6,
                &mut scratch,
            );
            let allocated = guard.check();
            result.unwrap();
            allocated.unwrap();
            close(&hidden, &expected);
            close(&keys[position * 6..position * 6 + 6], &key);
            close(&values[position * 6..position * 6 + 6], &value);
            let (projected, queries, gated, probs): (&[f32], &[f32], &[f32], &[f32]) =
                match position {
                    0 => (&PROJECTED_0, &QUERIES_0, &GATED_0, &PROBS_0),
                    1 => (&PROJECTED_1, &QUERIES_1, &GATED_1, &PROBS_1),
                    _ => (&PROJECTED_2, &QUERIES_2, &GATED_2, &PROBS_2),
                };
            close(&scratch[..24], projected);
            close(&scratch[24..36], queries);
            close(&scratch[36..42], &key);
            close(&scratch[42..48], &value);
            close(&scratch[60..72], gated);
            close(&scratch[72..75], &expected);
            close(&scratch[75..75 + position + 1], &probs[position + 1..]);
            assert_eq!(cursor, position + 1);
            if folded {
                assert_eq!(hidden, raw_results[position]);
            } else {
                raw_results[position] = hidden;
            }
        }
        assert!(KEY_1.iter().zip(WRONG_ADJACENT_KEY_1).any(|(a,b)| (a-b).abs()>0.1));
        assert!(keys[6..12].iter().zip(WRONG_ADJACENT_KEY_1).any(|(a,b)| (a-b).abs()>0.1));
        keys.fill(0.0);
        values.fill(0.0);
        cursor = 0;
        let mut hidden = HIDDEN_0;
        execute_normalized(
            &host,
            &roles,
            &NORMALIZED_0,
            &mut hidden,
            &mut keys,
            &mut values,
            &mut cursor,
            TextRope {
                position: 0,
                inv_freq: &FREQ,
            },
            1e-6,
            &mut scratch,
        )
        .unwrap();
        close(&hidden, &RESIDUAL_0);
        assert_eq!(cursor, 1);
    }
}

struct Failing<'a> {
    inner: F32FixtureHost<'a>,
    calls: Cell<usize>,
    fail: usize,
    nan: bool,
}
impl Host for Failing<'_> {
    fn admits(&self, p: &Projection) -> bool {
        self.inner.admits(p)
    }
    fn read(&self, t: &TensorInfo, o: &mut [f32]) -> Result<(), Error> {
        if self.fail == 0 {
            o[0] = 1.0;
            return Ok(());
        }
        self.inner.read(t, o)
    }
    fn gemm(&self, p: &Projection, x: &[f32], o: &mut [f32]) -> Result<(), Error> {
        let n = self.calls.get() + 1;
        self.calls.set(n);
        if n == self.fail {
            o[0] = if self.nan { f32::NAN } else { 123.0 };
            if self.nan {
                return Ok(());
            }
            return Err(Error::Provider);
        }
        self.inner.gemm(p, x, o)
    }
}

#[test]
fn malformed_position_provider_partial_and_nonfinite_failures_preserve_live_state() {
    let (data, roles) = setup(false);
    let layout = WorkspaceLayout::checked(roles.geometry, 3).unwrap();
    for (fail, nan, position) in [(0, false, 1), (1, false, 1), (4, false, 1), (4, true, 1), (99, false, 0)] {
        let host = Failing {
            inner: F32FixtureHost { values: &data },
            calls: Cell::new(0),
            fail,
            nan,
        };
        let mut scratch = vec![0.0; layout.floats];
        let (mut keys, mut values, mut cursor) = ([0.25; 18], [-0.5; 18], 1);
        let mut hidden = HIDDEN_1;
        assert!(execute_normalized(
            &host,
            &roles,
            &NORMALIZED_1,
            &mut hidden,
            &mut keys,
            &mut values,
            &mut cursor,
            TextRope {
                position,
                inv_freq: &FREQ
            },
            1e-6,
            &mut scratch
        )
        .is_err());
        assert_eq!(hidden, HIDDEN_1);
        assert_eq!(keys, [0.25; 18]);
        assert_eq!(values, [-0.5; 18]);
        assert_eq!(cursor, 1);
    }
}

#[test]
fn explicit_input_norm_fold_is_identical() {
    let mut raw = [0.0; 3];
    let mut folded = [0.0; 3];
    zero_centered_rms_norm(&HIDDEN_1, &INPUT_NORM, 1e-6, &mut raw).unwrap();
    ordinary_rms_norm(
        &HIDDEN_1,
        &INPUT_NORM.map(|x| 1.0 + x),
        false,
        1e-6,
        &mut folded,
    )
    .unwrap();
    assert_eq!(raw, folded);
    close(&raw, &NORMALIZED_1);
}

#[test]
fn malformed_buffers_and_nonfinite_inputs_preserve_live_state() {
    let (data, roles) = setup(false);
    let host = F32FixtureHost { values: &data };
    let layout = WorkspaceLayout::checked(roles.geometry, 3).unwrap();
    assert!(WorkspaceLayout::checked(roles.geometry, usize::MAX).is_err());
    let mut oversized = roles.geometry;
    oversized.query_heads = usize::MAX;
    assert!(WorkspaceLayout::checked(oversized, 3).is_err());
    for case in 0..7 {
        let mut scratch = vec![0.0; layout.floats];
        let (mut keys, mut values, mut cursor) = ([0.25; 18], [-0.5; 18], 1);
        if case == 5 { cursor = 3; }
        if case == 6 { cursor = usize::MAX; }
        let original_cursor = cursor;
        let mut hidden = HIDDEN_1;
        let mut normalized = NORMALIZED_1;
        if case == 4 {
            normalized[1] = f32::NAN;
        }
        let ks = if case == 0 {
            &mut keys[..17]
        } else {
            &mut keys[..]
        };
        let vs = if case == 1 {
            &mut values[..17]
        } else {
            &mut values[..]
        };
        let freq = if case == 2 { &FREQ[..1] } else { &FREQ[..] };
        let ws = if case == 3 {
            &mut scratch[..layout.floats - 1]
        } else {
            &mut scratch[..]
        };
        assert!(execute_normalized(
            &host,
            &roles,
            &normalized,
            &mut hidden,
            ks,
            vs,
            &mut cursor,
            TextRope {
                position: original_cursor,
                inv_freq: freq
            },
            1e-6,
            ws
        )
        .is_err());
        assert_eq!(hidden, HIDDEN_1);
        assert_eq!(keys, [0.25; 18]);
        assert_eq!(values, [-0.5; 18]);
        assert_eq!(cursor, original_cursor);
    }
}
