/// Zero-copy byte slices for one expert's NVFP4 tensors.
pub struct FtwExpertData<'a> {
    pub gate_up_packed: &'a [u8],
    pub gate_up_scale: &'a [u8],
    pub gate_up_global: &'a [u8],
    pub down_packed: &'a [u8],
    pub down_scale: &'a [u8],
    pub down_global: &'a [u8],
}

impl<'a> FtwExpertData<'a> {
    /// Convert zero-copy slice into typed `ExpertWeightView` for zero-heap SwiGLU computation.
    pub fn to_view(
        &self,
        emb_dim: usize,
        intermediate_dim: usize,
    ) -> crate::moe::dispatch::ExpertWeightView<'a> {
        let gate_up_global = if self.gate_up_global.len() >= 4 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&self.gate_up_global[..4]);
            f32::from_le_bytes(b)
        } else {
            1.0
        };
        let down_global = if self.down_global.len() >= 4 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&self.down_global[..4]);
            f32::from_le_bytes(b)
        } else {
            1.0
        };

        crate::moe::dispatch::ExpertWeightView {
            gate_up_packed: self.gate_up_packed,
            gate_up_scale: self.gate_up_scale,
            gate_up_global,
            down_packed: self.down_packed,
            down_scale: self.down_scale,
            down_global,
            intermediate_dim,
            emb_dim,
        }
    }
}

pub trait ExpertSource {
    fn fetch_tensor_bytes(&self, name: &str) -> Option<&[u8]>;
    fn layer_tensor(&self, layer: u16, suffix: &[u8]) -> Option<&[u8]> {
        let mut name = [0u8; 128];
        let n = etalon_model::inference::write_hf_layer_tensor_name(
            u32::from(layer),
            suffix,
            &mut name,
        );
        self.fetch_tensor_bytes(core::str::from_utf8(name.get(..n)?).ok()?)
    }
    fn canonical_layer_tensor(&self, layer: u16, suffix: &[u8]) -> Option<&[u8]> {
        let mut name = [0u8; 128];
        let n = etalon_model::inference::write_blk_tensor_name(u32::from(layer), suffix, &mut name);
        self.fetch_tensor_bytes(core::str::from_utf8(name.get(..n)?).ok()?)
    }
    fn get_expert_data(&self, layer: u16, expert: u16) -> Option<FtwExpertData<'_>>;
}
