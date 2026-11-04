use super::*;
pub fn dynamic_uniform_binding(buf: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: buf,
        offset: 0,
        size: std::num::NonZeroU64::new(MC8_UNIFORM_ALIGN as u64),
    })
}
pub fn buf_slice(buf: &wgpu::Buffer, offset: u64, bytes: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: buf,
        offset,
        size: std::num::NonZeroU64::new(bytes.max(4)),
    })
}
pub fn bg_hash(parts: &[u64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &part in parts {
        h ^= part;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
pub struct EncodingResources<'a> {
    pub device: &'a wgpu::Device,
    pub role_strides: [u64; 7],
    pub attention_mask_buf: Option<&'a wgpu::Buffer>,
    pub attention_params_buf: Option<&'a wgpu::Buffer>,
    pub attention_pipeline: &'a wgpu::ComputePipeline,
    pub elem_add_residual_pipeline: &'a wgpu::ComputePipeline,
    pub elem_params_buf: Option<&'a wgpu::Buffer>,
    pub elem_rms_norm_pipeline: &'a wgpu::ComputePipeline,
    pub elem_silu_mul_pipeline: &'a wgpu::ComputePipeline,
    pub gemm_max_input_floats: usize,
    pub gemm_max_out_dim: u32,
    pub gemm_params_buf: Option<&'a wgpu::Buffer>,
    pub gemm_weight_buf: Option<&'a wgpu::Buffer>,
    pub kv_cache_gpu: Option<&'a wgpu::Buffer>,
    pub max_tensor_bytes: usize,
    pub mc8_attn_bind_layout: &'a wgpu::BindGroupLayout,
    pub mc8_bg_cache: &'a std::sync::Mutex<std::collections::HashMap<u64, wgpu::BindGroup>>,
    pub mc8_elem_bind_layout: &'a wgpu::BindGroupLayout,
    pub mc8_ffn_fused_bind_layout: &'a wgpu::BindGroupLayout,
    pub mc8_ffn_fused_pipeline: &'a wgpu::ComputePipeline,
    pub mc8_gemm_bind_layout: &'a wgpu::BindGroupLayout,
    pub mc8_weights_resident: bool,
    pub mmv_q8_0_pipeline: &'a wgpu::ComputePipeline,
    pub pipeline: &'a wgpu::ComputePipeline,
}
impl<'a, P: PrefillProvider> PrefillEngine<'a, P> {
    pub fn weight_binding(&self, role: Mc8WeightRole, layer: u32) -> wgpu::BindingResource<'a> {
        let r = self.host.encoding_resources();
        let buffer = self.host.weight_role_buffer(role);
        if r.mc8_weights_resident {
            let stride = r.role_strides[role.idx()];
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: layer as u64 * stride,
                size: std::num::NonZeroU64::new(stride.max(4)),
            })
        } else {
            buffer.as_entire_binding()
        }
    }
    pub fn write_weight_role(&self, role: Mc8WeightRole, raw: &[u8], max_bytes: usize) {
        self.host.gpu_queue().write_buffer(
            self.host.weight_role_buffer(role),
            0,
            &raw[..raw.len().min(max_bytes)],
        );
    }
}
