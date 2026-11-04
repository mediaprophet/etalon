//! Ordinary model execution with host-owned provider and authority policy.
pub mod attention;
pub mod cpu;
pub mod expert_source;
pub mod experts;
mod ffn;
pub mod gpu_params;
mod layer;
pub mod moe;
pub mod output;
pub mod quant_support;
pub mod qwen35;
pub mod rope;
mod ssm_granite;
pub mod ssm_layout;
mod ssm_qwen;
use etalon_model::inference::{LayerTensors, ModelHyperparams, TensorInfo};
pub const RMS_NORM_EPS: f32 = 1e-5;
pub fn matmul_dims(info: &TensorInfo) -> (usize, usize) {
    (
        info.dims[0] as usize,
        if info.n_dims > 1 && info.dims[1] > 0 {
            info.dims[1] as usize
        } else {
            1
        },
    )
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerEvent {
    StateMissing(u32),
    SsmUnrecognized(u32),
    GraniteModelMissing(u32),
    GraniteInputProjection(u32),
    GraniteConvolutionFetch(u32),
    QwenModelMissing(u32),
    QwenQkvProjection(u32),
}
/// Providers retain their backend identity and must bound their own allocation and I/O.
pub trait LayerProvider {
    type Index;
    fn moe_routing(_index: &Self::Index) -> etalon_model::inference::moe_routing::MoeRoutingSettings {
        etalon_model::inference::moe_routing::MoeRoutingSettings::default()
    }
    fn hyperparams(index: &Self::Index) -> ModelHyperparams;
    fn data_start(index: &Self::Index) -> u64;
    fn tensors(index: &Self::Index, layer: u32) -> LayerTensors;
    fn model_bytes(&self) -> Option<&[u8]>;
    fn state(&self) -> &[f32];
    fn state_mut(&mut self) -> &mut [f32];
    fn gemm(
        &self,
        index: &Self::Index,
        info: &TensorInfo,
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool;
    fn norm(&self, info: &TensorInfo, out: &mut [f32]) -> usize;
    fn raw(&self, info: &TensorInfo) -> Option<&[u8]>;
    fn event(&self, event: LayerEvent);
    fn moe(
        &mut self,
        index: &Self::Index,
        emb_dim: usize,
        tensors: &LayerTensors,
        out: &mut [f32],
        input: &[f32],
    ) -> bool;
    fn record_ffn_input(&self, input: &[f32]);
    fn attention(
        &mut self,
        index: &Self::Index,
        layer: u32,
        token: u32,
        hidden: &[f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        a: &mut [f32],
        b: &mut [f32],
    ) -> Option<usize>;
    fn ffn_dispatch(
        &mut self,
        index: &Self::Index,
        hidden: &mut [f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        a: &mut [f32],
        b: &mut [f32],
    ) -> bool;
    fn profile_attention(&self, ns: u64);
    fn profile_ffn(&self, ns: u64);
    fn layer_diagnostic(
        &self,
        layer: u32,
        attn_ok: bool,
        tensors: &LayerTensors,
        attention: &[f32],
        hidden: &[f32],
    );
    fn expert_source(&self) -> Option<&dyn expert_source::ExpertSource>;
}
pub struct HybridEngine<'a, P> {
    pub host: &'a mut P,
}
impl<P: LayerProvider> HybridEngine<'_, P> {
    pub fn dispatch(
        &mut self,
        index: &P::Index,
        layer: u32,
        hidden: &mut [f32],
        emb_dim: usize,
        tensors: &LayerTensors,
        scratch_a: &mut [f32],
        scratch_b: &mut [f32],
    ) -> bool {
        if self.host.state().is_empty() {
            self.host.event(LayerEvent::StateMissing(layer));
            return false;
        }
        let h = P::hyperparams(index);
        let mut offset = 0usize;
        for i in 0..layer {
            let t = P::tensors(index, i);
            if t.is_hybrid_ssm_layer() {
                let Ok(layout) = ssm_layout::checked_ssm_layout_for(&t, &h, h.n_embd as usize)
                else {
                    return false;
                };
                let Some(next) = offset.checked_add(layout.stride) else {
                    return false;
                };
                offset = next;
            }
        }
        let Ok(layout) = ssm_layout::checked_ssm_layout_for(tensors, &h, h.n_embd as usize) else {
            return false;
        };
        if offset
            .checked_add(layout.stride)
            .is_none_or(|end| end > self.host.state().len())
        {
            return false;
        }
        if tensors.ssm_in.is_some() {
            self.dispatch_granite_ssm_layer(
                index, layer, hidden, emb_dim, tensors, scratch_a, scratch_b, offset,
            )
        } else if tensors.attn_qkv.is_some() && tensors.ssm_alpha.is_some() {
            self.dispatch_qwen_ssm_layer(
                index, layer, hidden, emb_dim, tensors, scratch_a, scratch_b, offset,
            )
        } else {
            self.host.event(LayerEvent::SsmUnrecognized(layer));
            false
        }
    }
}

pub mod cpu_attention;

pub mod async_attention;

#[cfg(target_arch = "wasm32")]
pub mod async_attention_token;

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
pub mod cuda_attention;
