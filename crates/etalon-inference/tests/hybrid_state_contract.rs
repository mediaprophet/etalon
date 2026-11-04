#![cfg(feature = "cpu-inference")]
use etalon_inference::ordinary::{
    expert_source::ExpertSource, HybridEngine, LayerEvent, LayerProvider,
};
use etalon_model::inference::{LayerTensors, ModelHyperparams, TensorInfo};

struct Host {
    state: Vec<f32>,
    bytes: Option<[u8; 16]>,
    ffn_calls: usize,
}
impl LayerProvider for Host {
    type Index = (ModelHyperparams, LayerTensors);
    fn hyperparams(index: &Self::Index) -> ModelHyperparams {
        index.0
    }
    fn data_start(_: &Self::Index) -> u64 {
        0
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

#[test]
fn missing_recurrent_state_cannot_be_hidden_by_successful_ffn() {
    let mut host = Host {
        state: vec![],
        bytes: None,
        ffn_calls: 0,
    };
    let mut hidden = [3.0];
    assert!(!HybridEngine { host: &mut host }.layer(
        &(ModelHyperparams::default(), hybrid_tensors()),
        0,
        0,
        &mut hidden,
        1,
        &mut [0.0],
        &mut [0.0]
    ));
    assert_eq!(hidden, [3.0]);
    assert_eq!(host.ffn_calls, 0);
}

fn hybrid_tensors() -> LayerTensors {
    let matrix = TensorInfo {
        dims: [1, 1, 0, 0],
        n_dims: 2,
        ggml_type: 0,
        byte_offset: 0,
    };
    let mut t = LayerTensors::default();
    t.ssm_dt_bias = Some(TensorInfo {
        byte_offset: 0,
        ..matrix
    });
    t.ssm_norm = Some(TensorInfo {
        byte_offset: 0,
        ..matrix
    });
    t.ssm_in = Some(matrix);
    t.ssm_out = Some(matrix);
    t.ssm_conv1d = Some(TensorInfo {
        dims: [4, 1, 0, 0],
        ..matrix
    });
    t
}
#[test]
fn granite_single_layer_uses_the_admitted_configured_layout() {
    use etalon_inference::ordinary::ssm_layout::{
        allocate_recurrent_state, checked_ssm_layout_for,
    };
    let mut h = ModelHyperparams::default();
    h.n_layer = 1;
    h.n_embd = 1;
    h.head_dim = 64;
    h.ssm_inner_size = 1;
    h.ssm_state_size = 128;
    h.ssm_group_count = 1;
    let t = hybrid_tensors();
    let layout = checked_ssm_layout_for(&t, &h, 1).unwrap();
    assert_eq!(layout.stride, 3 + 128 * 64);
    let mut host = Host {
        state: allocate_recurrent_state([Ok(layout.stride)], layout.stride * 4)
            .unwrap()
            .into_vec(),
        bytes: Some([0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63]),
        ffn_calls: 0,
    };
    let mut hidden = [3.0];
    assert!(HybridEngine { host: &mut host }.layer(
        &(h, t),
        0,
        0,
        &mut hidden,
        1,
        &mut [0.0; 4],
        &mut [0.0; 4]
    ));
    assert!(hidden[0] > 3.0);
    assert_eq!(host.ffn_calls, 1);
    assert_eq!(host.state[2], 3.0);
}
