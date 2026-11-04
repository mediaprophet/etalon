use super::*;
use crate::native_resident::prefill_arena_slots::*;
pub(super) fn build_fusions(
    device: &wgpu::Device,
    resources: &ResidentPlanResources<'_>,
    param_arena: &wgpu::Buffer,
    lu: u64,
    use_coop: bool,
    options: ResidentPlanOptions,
    g_ty: u32,
    u_ty: u32,
    k_info: &GgufTensorInfo,
    v_info: &GgufTensorInfo,
    g_w: &wgpu::Buffer,
    u_w: &wgpu::Buffer,
    k_w: &wgpu::Buffer,
    v_w: &wgpu::Buffer,
    normed: &wgpu::Buffer,
    silu_buf: &wgpu::Buffer,
    k_proj: &wgpu::Buffer,
    v_proj: &wgpu::Buffer,
) -> (Option<wgpu::BindGroup>, Option<wgpu::BindGroup>) {
    let gp_sz = std::num::NonZeroU64::new(std::mem::size_of::<GemmGpuParams>() as u64);
    let slot_off = |l: u64, s: u64| (l * SLOTS_PER_LAYER + s) * SLOT;
    let fused_ok = options.ffn_fusion
        && g_ty == u_ty
        && matches!(
            g_ty,
            etalon_model::ggml::GGML_TYPE_Q4_0
                | etalon_model::ggml::GGML_TYPE_Q5_0
                | etalon_model::ggml::GGML_TYPE_Q8_0
                | etalon_model::ggml::GGML_TYPE_Q4_K
                | etalon_model::ggml::GGML_TYPE_Q4_K_SOA
                | etalon_model::ggml::GGML_TYPE_Q6_K
        );
    let fused_ffn = if fused_ok {
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PrefillFusedFfn"),
            layout: resources.ffn_fused_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: g_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ubind(&param_arena, slot_off(lu, S_GEMM_GATE), gp_sz),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: silu_buf.as_entire_binding(),
                },
            ],
        }))
    } else {
        None
    };
    let dual_kv = if k_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        && v_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        && use_coop
    {
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PrefillDualKv"),
            layout: resources.dual_gemv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ubind(&param_arena, slot_off(lu, S_GEMM_K), gp_sz),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: k_proj.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: v_proj.as_entire_binding(),
                },
            ],
        }))
    } else {
        None
    };

    (fused_ffn, dual_kv)
}
