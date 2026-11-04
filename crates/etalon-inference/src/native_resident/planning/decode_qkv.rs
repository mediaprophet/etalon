use super::*;
use crate::native_resident::resident_decode_slots::*;
pub(super) fn build_qkv(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &ResidentPlanResources<'_>,
    static_arena: &wgpu::Buffer,
    gbase: u64,
    n_embd: usize,
    q_dim: usize,
    kv_dim: usize,
    q_raw_len: usize,
    use_coop: bool,
    options: ResidentPlanOptions,
    q_info: &GgufTensorInfo,
    k_info: &GgufTensorInfo,
    v_info: &GgufTensorInfo,
    q_w: &wgpu::Buffer,
    k_w: &wgpu::Buffer,
    v_w: &wgpu::Buffer,
    normed: &wgpu::Buffer,
    q_proj: &wgpu::Buffer,
    k_proj: &wgpu::Buffer,
    v_proj: &wgpu::Buffer,
) -> (Option<wgpu::BindGroup>, Option<wgpu::BindGroup>) {
    let gp_sz = std::num::NonZeroU64::new(std::mem::size_of::<GemmGpuParams>() as u64);
    let want_triple = options.triple_qkv;
    let triple_qkv = if want_triple
        && q_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        && k_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        && v_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        && use_coop
    {
        let tp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ResTripleQkvParams"),
            size: SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut p = GemmGpuParams {
            n_in: n_embd as u32,
            n_out: q_dim as u32,
            weight_ggml_type: q_info.ggml_type,
            weight_row_elems: q_info.dims[0] as u32,
            weight_byte_len: q_raw_len as u32,
            n_batch: 1,
            in_row_stride: 0,
            out_row_stride: 0,
        };
        p.weight_byte_len = kv_dim as u32;
        queue.write_buffer(&tp, 0, bytemuck::bytes_of(&p));
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ResTripleQkv"),
            layout: resources.triple_gemv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: normed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: q_proj.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: k_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: k_proj.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: v_w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: v_proj.as_entire_binding(),
                },
            ],
        }))
    } else {
        None
    };
    let dual_kv = if triple_qkv.is_none()
        && k_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        && v_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
        && use_coop
    {
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ResDualKv"),
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
                    resource: ubind(&static_arena, gbase * SLOT, gp_sz),
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

    (triple_qkv, dual_kv)
}
