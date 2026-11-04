#![cfg(feature = "inference-model")]
use etalon_inference::ordinary::qwen35::{
    fixture::F32FixtureHost, model_execute::*, model_plan::*, state::*, Error, Host,
};
use etalon_model::inference::{
    moe_routing::MoeRoutingSettings, qwen_attention::*, qwen_roles::*, *,
};
#[path = "../../etalon-inference-kernel/tests/support/qwen_numeric_alloc.rs"]
mod allocation;
#[path = "support/qwen_model_fixture.rs"]
#[allow(clippy::excessive_precision)]
mod fixture;
use fixture::*;
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-6 + 2e-5 * b.abs(),
            "lane {i}: {a} != {b}"
        );
    }
}
fn setup() -> (TensorIndex, Vec<f32>) {
    let mut idx = TensorIndex {
        entries: vec![],
        tensor_data_start: 0,
        token_embd: None,
        output_weight: None,
        output_norm: None,
        ple_ngram_embedding: None,
        moe_routing: MoeRoutingSettings {
            expert_count: Some(2),
            experts_per_token: Some(2),
        },
        ple_config: None,
        hyperparams: ModelHyperparams {
            architecture: ARCH_QWEN35_MOE,
            n_embd: 3,
            n_layer: 2,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 6,
            ssm_group_count: 1,
            ssm_time_step_rank: 2,
            ssm_state_size: 2,
            ssm_inner_size: 4,
            ssm_conv_kernel: 2,
            ..Default::default()
        },
        max_tensor_bytes: 0,
        max_layer_tensor_bytes: 0,
    };
    let mut data = vec![];
    for &(name, dims, values) in WEIGHTS {
        let mut shape = [0; 4];
        for (i, &d) in dims.iter().enumerate() {
            shape[i] = d as u64;
        }
        let t = TensorInfo {
            dims: shape,
            n_dims: dims.len() as u32,
            ggml_type: 0,
            byte_offset: (data.len() * 4) as u64,
        };
        idx.entries
            .push((etalon_model::gguf::fnv1a64(name.as_bytes()), t));
        data.extend_from_slice(values);
        if name == "model.norm.weight" {
            idx.output_norm = Some(t);
        }
        if name == "lm_head.weight" {
            idx.output_weight = Some(t);
        }
    }
    (idx, data)
}
const FREQ: [f32; 2] = [1.0, 0.01];
fn profile() -> Profile<'static> {
    Profile {
        packing: Qwen35Packing::Separate,
        ordinary_norm: QkNorm::RawZeroCentered,
        qk_norm: QkNorm::RawZeroCentered,
        epsilon: 1e-6,
        rotary_inv_freq: &FREQ,
        max_context: 2,
        output: OutputContract::Named { vocab: 4 },
    }
}
fn kinds() -> [LayerKind; 2] {
    [
        LayerKind::Linear,
        LayerKind::FullAttention(Qwen35AttentionGeometry {
            hidden: 3,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 6,
            rotary_dim: 4,
        }),
    ]
}
struct Check {
    token: usize,
    count: usize,
}
impl Observer for Check {
    fn observe(&mut self, l: usize, s: Stage, x: &[f32]) {
        let expected: &[f32] = match (self.token, l, s) {
            (0, 0, Stage::Attention) => &T0_ATTENTION0,
            (0, 0, Stage::Ffn) => &T0_FFN0,
            (0, 1, Stage::Attention) => &T0_ATTENTION1,
            (0, 1, Stage::Ffn) => &T0_FFN1,
            (1, 0, Stage::Attention) => &T1_ATTENTION0,
            (1, 0, Stage::Ffn) => &T1_FFN0,
            (1, 1, Stage::Attention) => &T1_ATTENTION1,
            (1, 1, Stage::Ffn) => &T1_FFN1,
            (0, _, Stage::FinalNorm) => &T0_FINAL,
            (1, _, Stage::FinalNorm) => &T1_FINAL,
            (0, _, Stage::Logits) => &T0_LOGITS,
            (1, _, Stage::Logits) => &T1_LOGITS,
            _ => panic!("unexpected stage"),
        };
        close(x, expected);
        self.count += 1;
    }
}
#[test]
fn independent_mixed_two_token_stages_caches_logits_reset_and_zero_heap() {
    let (idx, data) = setup();
    let kinds = kinds();
    let p = profile();
    let r = measure(&idx, &kinds, p, usize::MAX).unwrap();
    assert_eq!(
        measure(&idx, &kinds, p, r.bytes - 1),
        Err(PlanError::Budget)
    );
    assert_eq!(measure(&idx, &kinds, p, r.bytes).unwrap(), r);
    let mut slots = [None; 2];
    let plan = prepare(&idx, &kinds, p, r.bytes, &mut slots).unwrap();
    let (mut history, mut recurrent, mut kv, mut scratch) = (
        vec![0.0; r.history],
        vec![0.0; r.recurrent],
        vec![0.0; r.kv],
        vec![0.0; r.scratch],
    );
    let mut state =
        ModelState::bind(&plan, &mut history, &mut recurrent, &mut kv, &mut scratch).unwrap();
    let host = F32FixtureHost { values: &data };
    for _replay in 0..2 {
        state.reset();
        for token in 0..2 {
            let mut hidden = if token == 0 { T0_INPUT } else { T1_INPUT };
            let mut logits = [99.0; 4];
            let mut obs = Check { token, count: 0 };
            let guard = allocation::AllocGuard::begin("mixed-model", true);
            let result = step(
                &host,
                &plan,
                &mut state,
                token,
                &mut hidden,
                &mut logits,
                &mut obs,
            );
            let alloc = guard.check();
            result.unwrap();
            alloc.unwrap();
            assert_eq!(obs.count, 6);
            close(&hidden, if token == 0 { &T0_FINAL } else { &T1_FINAL });
            close(&logits, if token == 0 { &T0_LOGITS } else { &T1_LOGITS });
            let (h, s, k) = state.cache();
            close(h, if token == 0 { &T0_HISTORY } else { &T1_HISTORY });
            close(s, if token == 0 { &T0_STATE } else { &T1_STATE });
            close(k, if token == 0 { &T0_KV } else { &T1_KV });
            assert_eq!(state.position(), Ok(token + 1));
        }
    }
}
struct LateFail<'a> {
    inner: F32FixtureHost<'a>,
    offset: u64,
}
impl Host for LateFail<'_> {
    fn admits(&self, p: &Projection) -> bool {
        self.inner.admits(p)
    }
    fn read(&self, t: &TensorInfo, o: &mut [f32]) -> Result<(), Error> {
        self.inner.read(t, o)
    }
    fn gemm(&self, p: &Projection, x: &[f32], o: &mut [f32]) -> Result<(), Error> {
        if p.weight.byte_offset == self.offset {
            o[0] = 123.;
            return Err(Error::Provider);
        }
        self.inner.gemm(p, x, o)
    }
}
#[test]
fn late_output_failure_withholds_publication_and_requires_reset() {
    let (idx, data) = setup();
    let kinds = kinds();
    let mut slots = [None; 2];
    let plan = prepare(&idx, &kinds, profile(), usize::MAX, &mut slots).unwrap();
    let r = plan.requirements();
    let (mut h, mut s, mut k, mut w) = (
        vec![0.; r.history],
        vec![0.; r.recurrent],
        vec![0.; r.kv],
        vec![0.; r.scratch],
    );
    let mut state = ModelState::bind(&plan, &mut h, &mut s, &mut k, &mut w).unwrap();
    let host = LateFail {
        inner: F32FixtureHost { values: &data },
        offset: plan.output().weight.byte_offset,
    };
    let mut hidden = T0_INPUT;
    let mut logits = [77.; 4];
    assert!(step(
        &host,
        &plan,
        &mut state,
        0,
        &mut hidden,
        &mut logits,
        &mut NoopObserver
    )
    .is_err());
    assert_eq!(hidden, T0_INPUT);
    assert_eq!(logits, [77.; 4]);
    assert_eq!(state.position(), Err(SequenceError::Invalidated));
    assert!(step(
        &host.inner,
        &plan,
        &mut state,
        0,
        &mut hidden,
        &mut logits,
        &mut NoopObserver
    )
    .is_err());
    state.reset();
    step(
        &host.inner,
        &plan,
        &mut state,
        0,
        &mut hidden,
        &mut logits,
        &mut NoopObserver,
    )
    .unwrap();
    close(&logits, &T0_LOGITS);
}

#[test]
fn missing_routing_and_mismatched_explicit_profile_fail_closed() {
    let (mut idx, _) = setup();
    let kinds = kinds();
    let mut bad = profile();
    bad.rotary_inv_freq = &FREQ[..1];
    assert!(measure(&idx, &kinds, bad, usize::MAX).is_err());
    let mut bad = profile();
    bad.output = OutputContract::Named { vocab: 5 };
    assert!(measure(&idx, &kinds, bad, usize::MAX).is_err());
    idx.moe_routing = Default::default();
    assert_eq!(
        measure(&idx, &kinds, profile(), usize::MAX),
        Err(PlanError::MissingRouting)
    );
}
