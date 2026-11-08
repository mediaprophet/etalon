#![cfg(feature = "cpu-inference")]
use etalon_inference::ordinary::{expert_source::ExpertSource, LayerEvent, LayerProvider};
use etalon_model::inference::{LayerTensors, ModelHyperparams, TensorInfo};

struct Host {
    state: Vec<f32>,
    bytes: Option<Vec<u8>>,
    ffn_calls: usize,
}
impl LayerProvider for Host {
    type Index = (ModelHyperparams, LayerTensors, MoeRoutingSettings);
    fn moe_routing(index: &Self::Index) -> MoeRoutingSettings {
        index.2
    }
    fn hyperparams(index: &Self::Index) -> ModelHyperparams {
        index.0
    }
    fn data_start(_: &Self::Index) -> u64 {
        64
    }
    fn tensors(index: &Self::Index, _: u32) -> LayerTensors {
        index.1
    }
    fn model_bytes(&self) -> Option<&[u8]> {
        self.bytes.as_ref().map(|bytes| bytes.as_slice())
    }
    fn state(&self) -> &[f32] {
        &self.state
    }
    fn state_mut(&mut self) -> &mut [f32] {
        &mut self.state
    }
    fn gemm(
        &self,
        _: &Self::Index,
        _: &TensorInfo,
        input: &[f32],
        out: &mut [f32],
        _: usize,
        n_out: usize,
    ) -> bool {
        if self.bytes.is_none() {
            return false;
        }
        out[..n_out].copy_from_slice(&input[..n_out]);
        true
    }
    fn norm(&self, _: &TensorInfo, _: &mut [f32]) -> usize {
        0
    }
    fn raw(&self, _: &TensorInfo) -> Option<&[u8]> {
        None
    }
    fn event(&self, _: LayerEvent) {}
    fn moe(
        &mut self,
        _: &Self::Index,
        _: usize,
        _: &LayerTensors,
        _: &mut [f32],
        _: &[f32],
    ) -> bool {
        false
    }
    fn record_ffn_input(&self, _: &[f32]) {}
    fn attention(
        &mut self,
        _: &Self::Index,
        _: u32,
        _: u32,
        _: &[f32],
        _: usize,
        _: &LayerTensors,
        _: &mut [f32],
        _: &mut [f32],
    ) -> Option<usize> {
        None
    }
    fn ffn_dispatch(
        &mut self,
        _: &Self::Index,
        _: &mut [f32],
        _: usize,
        _: &LayerTensors,
        _: &mut [f32],
        _: &mut [f32],
    ) -> bool {
        self.ffn_calls += 1;
        true
    }
    fn profile_attention(&self, _: u64) {}
    fn profile_ffn(&self, _: u64) {}
    fn layer_diagnostic(&self, _: u32, _: bool, _: &LayerTensors, _: &[f32], _: &[f32]) {}
    fn expert_source(&self) -> Option<&dyn ExpertSource> {
        None
    }
}

use etalon_inference::ordinary::moe::execute_moe;
use etalon_model::inference::moe_routing::MoeRoutingSettings;

fn fixture() -> (Host, LayerTensors) {
    let mut bytes = vec![0xA5; 64];
    for tensor in 0..4 {
        for expert in 0..512 {
            let value = match tensor {
                0 => expert as f32 / 64.0,
                1 | 2 => 1.0,
                _ => (expert + 1) as f32,
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let info = |slot: u64, router| TensorInfo {
        dims: if router {
            [1, 512, 0, 0]
        } else {
            [1, 1, 512, 0]
        },
        n_dims: if router { 2 } else { 3 },
        ggml_type: 0,
        byte_offset: slot * 512 * 4,
    };
    let tensors = LayerTensors {
        moe_router: Some(info(0, true)),
        moe_gate_exps: Some(info(1, false)),
        moe_up_exps: Some(info(2, false)),
        moe_down_exps: Some(info(3, false)),
        ..LayerTensors::default()
    };
    (
        Host {
            state: vec![],
            bytes: Some(bytes),
            ffn_calls: 0,
        },
        tensors,
    )
}

fn expected(k: usize) -> f64 {
    let mut weighted = 0.0;
    let mut mass = 0.0;
    for expert in (512 - k..512).rev() {
        let probability = ((expert as f64 - 511.0) / 64.0).exp();
        weighted += probability * (expert + 1) as f64 / (1.0 + (-1.0f64).exp());
        mass += probability;
    }
    weighted / mass
}

#[test]
fn ordinary_execution_consumes_provider_topk_and_nonzero_data_start() {
    let (host, tensors) = fixture();
    for explicit in [Some(6), Some(8), Some(10), None] {
        let settings = MoeRoutingSettings {
            expert_count: Some(512),
            experts_per_token: explicit,
        };
        let index = (ModelHyperparams::default(), tensors, settings);
        let mut out = [0.0];
        assert!(execute_moe(&host, &index, 1, &tensors, &mut out, &[1.0]));
        let want = expected(explicit.unwrap_or(8) as usize);
        assert!(
            (out[0] as f64 - want).abs() < 1e-4,
            "{explicit:?}: {} vs {want}",
            out[0]
        );
    }
}

#[test]
fn ordinary_refuses_bad_settings_and_short_buffers() {
    let (host, tensors) = fixture();
    for settings in [
        MoeRoutingSettings {
            expert_count: Some(256),
            experts_per_token: Some(8),
        },
        MoeRoutingSettings {
            expert_count: Some(512),
            experts_per_token: Some(0),
        },
        MoeRoutingSettings {
            expert_count: Some(512),
            experts_per_token: Some(17),
        },
    ] {
        let index = (ModelHyperparams::default(), tensors, settings);
        let mut out = [42.0];
        assert!(!execute_moe(&host, &index, 1, &tensors, &mut out, &[1.0]));
        assert_eq!(out, [42.0]);
    }
    let index = (
        ModelHyperparams::default(),
        tensors,
        MoeRoutingSettings::default(),
    );
    assert!(!execute_moe(&host, &index, 1, &tensors, &mut [], &[1.0]));
    assert!(!execute_moe(&host, &index, 1, &tensors, &mut [42.0], &[]));
}
