use super::*;
use crate::native_resident::{resident_decode_plan::*, resident_decode_slots::*};

pub(super) struct DecodeActivations<'a> {
    pub hidden_a: &'a wgpu::Buffer,
    pub hidden_b: &'a wgpu::Buffer,
    pub normed: &'a wgpu::Buffer,
    pub attn_out: &'a wgpu::Buffer,
    pub k_proj: &'a wgpu::Buffer,
    pub v_proj: &'a wgpu::Buffer,
    pub q_proj: &'a wgpu::Buffer,
    pub gate_buf: &'a wgpu::Buffer,
    pub up_buf: &'a wgpu::Buffer,
    pub silu_buf: &'a wgpu::Buffer,
}

pub(super) fn build_layers<P: ResidentPlanProvider>(
    host: &P,
    index: &P::Index,
    mmap: &[u8],
    h: &ModelHyperparams,
    layout: &P::Layout,
    options: ResidentPlanOptions,
    resources: &ResidentPlanResources<'_>,
    binds: &Bindings<'_>,
    rms_layout: &wgpu::BindGroupLayout,
    silu_layout: &wgpu::BindGroupLayout,
    norm_res: &wgpu::Buffer,
    norm_stride: u64,
    elem_base: u64,
    n_ffn: usize,
    a: DecodeActivations<'_>,
    upload_norm: impl Fn(u64, &GgufTensorInfo) -> bool,
) -> Option<(Vec<LayerBinds>, Vec<LayerProtos>, u32)> {
    let DecodeActivations {
        hidden_a,
        hidden_b,
        normed,
        attn_out,
        k_proj,
        v_proj,
        q_proj,
        gate_buf,
        up_buf,
        silu_buf,
    } = a;
    let n_embd = h.n_embd as usize;
    let n_layer = h.n_layer;
    let q_dim = h.n_head as usize * h.head_dim() as usize;
    let kv_dim = h.effective_n_kv_head() as usize * h.head_dim() as usize;
    let use_coop = options.coop_gemv;
    let device = binds.device;
    let queue = host.gpu_queue();
    let static_arena = binds.params;
    let gemm_layout = binds.gemm_layout;
    let gp_sz = std::num::NonZeroU64::new(std::mem::size_of::<GemmGpuParams>() as u64);
    let norm_bind = |slot: u64| {
        ubind(
            norm_res,
            slot * norm_stride,
            std::num::NonZeroU64::new((n_embd * 4) as u64),
        )
    };
    let mk_elem_bg =
        |label: &str,
         layout: &wgpu::BindGroupLayout,
         a: wgpu::BindingResource,
         b: wgpu::BindingResource,
         out: &wgpu::Buffer,
         off: u64| binds.elem(label, layout, a, b, out, elem_base + off);
    let mk_gemm_bg = |label: &str,
                      input: &wgpu::Buffer,
                      weight: wgpu::BindingResource,
                      slot: u64,
                      out: &wgpu::Buffer| {
        binds.gemm(
            label,
            &gemm_layout,
            input,
            weight,
            slot * SLOT,
            out.as_entire_binding(),
            input,
        )
    };
    let mk_resid_bg = |label: &str,
                       input: &wgpu::Buffer,
                       weight: wgpu::BindingResource,
                       slot: u64,
                       residual: &wgpu::Buffer,
                       out: &wgpu::Buffer| {
        binds.gemm(
            label,
            resources.coop_gemv_residual_bind_layout,
            input,
            weight,
            slot * SLOT,
            out.as_entire_binding(),
            residual,
        )
    };
    let mk_attn_bg =
        |label: &str,
         input: &wgpu::Buffer,
         weight: &wgpu::Buffer,
         slot: u64,
         layer: u32,
         out: &wgpu::Buffer| binds.attention(label, input, weight, slot * SLOT, layer, out);

    let gemm_params =
        |ggml_type: u32, n_in: usize, n_out: usize, row_elems: u32, raw_len: usize| GemmGpuParams {
            n_in: n_in as u32,
            n_out: n_out as u32,
            weight_ggml_type: ggml_type,
            weight_row_elems: row_elems,
            weight_byte_len: raw_len as u32,
            n_batch: 1,
            in_row_stride: 0,
            out_row_stride: 0,
        };

    let mut layers = Vec::with_capacity(n_layer as usize);
    let mut layer_protos = Vec::with_capacity(n_layer as usize);
    let mut layer_gate_ggml_type: u32 = 0;
    for l in 0..n_layer {
        let t = P::tensors(index, l);
        let (q_info, k_info, v_info) = (t.attn_q?, t.attn_k?, t.attn_v?);
        let o_info = t.attn_output?;
        let (gate_info, up_info, down_info) = (t.ffn_gate?, t.ffn_up?, t.ffn_down?);
        if l == 0 {
            layer_gate_ggml_type = gate_info.ggml_type;
        }
        let attn_norm = t.attn_norm?;
        let ffn_norm = t.ffn_norm?;
        for i in [&q_info, &k_info, &v_info] {
            if !ggml_gpu_attention_shader_supported(i.ggml_type)
                || !ggml_gpu_gemm_supported(i.ggml_type)
            {
                return None;
            }
        }
        for i in [&o_info, &gate_info, &up_info, &down_info] {
            if !ggml_gpu_gemm_supported(i.ggml_type) {
                return None;
            }
        }
        let (q_in, q_out) = matmul_dims(&q_info);
        let (k_in, k_out) = matmul_dims(&k_info);
        let (v_in, v_out) = matmul_dims(&v_info);
        let (o_in, o_out) = matmul_dims(&o_info);
        let (g_in, g_out) = matmul_dims(&gate_info);
        let (u_in, u_out) = matmul_dims(&up_info);
        let (d_in, d_out) = matmul_dims(&down_info);
        if q_in != n_embd
            || q_out != q_dim
            || k_in != n_embd
            || v_in != n_embd
            || k_out != kv_dim
            || v_out != kv_dim
            || o_in != q_dim
            || o_out != n_embd
            || g_in != n_embd
            || u_in != n_embd
            || g_out != n_ffn
            || u_out != n_ffn
            || d_in != n_ffn
            || d_out != n_embd
        {
            return None;
        }
        let fetch = |i: &GgufTensorInfo| P::tensor_bytes(mmap, P::data_start(index), i);
        let (q_raw, k_raw, v_raw) = (fetch(&q_info)?, fetch(&k_info)?, fetch(&v_info)?);
        let (o_raw, g_raw, u_raw, d_raw) = (
            fetch(&o_info)?,
            fetch(&gate_info)?,
            fetch(&up_info)?,
            fetch(&down_info)?,
        );
        host.prewarm_layer(
            l,
            &[
                (&q_info, q_raw, q_in, q_out),
                (&k_info, k_raw, n_embd, kv_dim),
                (&v_info, v_raw, n_embd, kv_dim),
                (&o_info, o_raw, o_in, o_out),
                (&gate_info, g_raw, g_in, g_out),
                (&up_info, u_raw, u_in, u_out),
                (&down_info, d_raw, d_in, d_out),
            ],
        );
        let res = |raw: &[u8]| host.resident_weight_buffer(raw.as_ptr() as u64, raw);
        let (q_w, k_w, v_w) = (res(q_raw)?, res(k_raw)?, res(v_raw)?);
        let o_w = res(o_raw)?;
        let ffn_bind =
            |info: &GgufTensorInfo, raw: &[u8]| -> Option<(wgpu::Buffer, u32, u32, u32)> {
                if let Some(p) = host.promote_matrix_to_f16_resident(info, raw) {
                    return Some(p);
                }
                let b = res(raw)?;
                Some((b, info.ggml_type, raw.len() as u32, info.dims[0] as u32))
            };
        let (g_w, g_ty, g_blen, g_row) = ffn_bind(&gate_info, g_raw)?;
        let (u_w, u_ty, u_blen, u_row) = ffn_bind(&up_info, u_raw)?;
        let (d_w, d_ty, d_blen, d_row) = ffn_bind(&down_info, d_raw)?;
        if l == 0
            && (g_ty == etalon_model::ggml::GGML_TYPE_F16
                || u_ty == etalon_model::ggml::GGML_TYPE_F16
                || d_ty == etalon_model::ggml::GGML_TYPE_F16)
        {
            host.event(PlanEvent::FfnF16(g_ty, u_ty, d_ty));
        }

        if !upload_norm(2 * l as u64, &attn_norm) || !upload_norm(2 * l as u64 + 1, &ffn_norm) {
            return None;
        }

        let gbase = l as u64 * GEMM_SLOTS_PER_LAYER;
        for (i, (ggml_type, n_in, n_out, row_elems, raw_len)) in [
            (
                k_info.ggml_type,
                n_embd,
                kv_dim,
                k_info.dims[0] as u32,
                k_raw.len(),
            ),
            (
                v_info.ggml_type,
                n_embd,
                kv_dim,
                v_info.dims[0] as u32,
                v_raw.len(),
            ),
            (
                q_info.ggml_type,
                n_embd,
                q_dim,
                q_info.dims[0] as u32,
                q_raw.len(),
            ),
            (
                o_info.ggml_type,
                q_dim,
                n_embd,
                o_info.dims[0] as u32,
                o_raw.len(),
            ),
            (g_ty, n_embd, n_ffn, g_row, g_blen as usize),
            (u_ty, n_embd, n_ffn, u_row, u_blen as usize),
            (d_ty, n_ffn, n_embd, d_row, d_blen as usize),
        ]
        .into_iter()
        .enumerate()
        {
            queue.write_buffer(
                &static_arena,
                (gbase + i as u64) * SLOT,
                bytemuck::bytes_of(&gemm_params(ggml_type, n_in, n_out, row_elems, raw_len)),
            );
        }

        let mut k_p =
            P::attention_gpu_params(&h, &layout, l, 0, &k_info, k_raw.len(), 1, 1, 0, 0, 0, 0);
        k_p.proj_row_stride = kv_dim as u32;
        let mut v_p =
            P::attention_gpu_params(&h, &layout, l, 0, &v_info, v_raw.len(), 2, 1, 0, 0, 0, 0);
        v_p.proj_row_stride = kv_dim as u32;
        let mut q_p =
            P::attention_gpu_params(&h, &layout, l, 0, &q_info, q_raw.len(), 0, 1, 0, 0, 0, 0);
        q_p.proj_row_stride = q_dim as u32;
        layer_protos.push(LayerProtos {
            k_write: k_p,
            v_write: v_p,
            q: q_p,
        });

        let dyn_base = l as u64 * ATTN_SLOTS_PER_LAYER;
        let (triple_qkv, dual_kv) = super::decode_qkv::build_qkv(
            device,
            queue,
            resources,
            static_arena,
            gbase,
            n_embd,
            q_dim,
            kv_dim,
            q_raw.len(),
            use_coop,
            options,
            &q_info,
            &k_info,
            &v_info,
            &q_w,
            &k_w,
            &v_w,
            normed,
            q_proj,
            k_proj,
            v_proj,
        );
        layers.push(LayerBinds {
            rms_attn: mk_elem_bg(
                "ResRmsAttn",
                &rms_layout,
                hidden_a.as_entire_binding(),
                norm_bind(2 * l as u64),
                &normed,
                ELEM_SLOT_RMS,
            ),
            triple_qkv,
            dual_kv,
            k_gemm: mk_gemm_bg("ResKGemm", &normed, k_w.as_entire_binding(), gbase, &k_proj),
            k_write: mk_attn_bg("ResKWrite", &k_proj, &k_w, dyn_base, l, &attn_out),
            v_gemm: mk_gemm_bg(
                "ResVGemm",
                &normed,
                v_w.as_entire_binding(),
                gbase + 1,
                &v_proj,
            ),
            v_write: mk_attn_bg("ResVWrite", &v_proj, &v_w, dyn_base + 1, l, &attn_out),
            q_gemm: mk_gemm_bg(
                "ResQGemm",
                &normed,
                q_w.as_entire_binding(),
                gbase + 2,
                &q_proj,
            ),
            q: mk_attn_bg("ResQSdpa", &q_proj, &q_w, dyn_base + 2, l, &attn_out),
            o_resid: mk_resid_bg(
                "ResOResid",
                &attn_out,
                o_w.as_entire_binding(),
                gbase + 3,
                &hidden_a,
                &hidden_b,
            ),
            rms_ffn: mk_elem_bg(
                "ResRmsFfn",
                &rms_layout,
                hidden_b.as_entire_binding(),
                norm_bind(2 * l as u64 + 1),
                &normed,
                ELEM_SLOT_RMS,
            ),
            fused_ffn: {
                let want = options.ffn_fusion && g_ty == u_ty && fused_ffn_quant_supported(g_ty);
                if want {
                    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ResFusedFfn"),
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
                                resource: ubind(&static_arena, (gbase + 4) * SLOT, gp_sz),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: silu_buf.as_entire_binding(),
                            },
                        ],
                    }))
                } else {
                    None
                }
            },
            gate: {
                let want = !(options.ffn_fusion && g_ty == u_ty && fused_ffn_quant_supported(g_ty));
                if want {
                    Some(mk_gemm_bg(
                        "ResGate",
                        &normed,
                        g_w.as_entire_binding(),
                        gbase + 4,
                        &gate_buf,
                    ))
                } else {
                    None
                }
            },
            up: {
                let want = !(options.ffn_fusion && g_ty == u_ty && fused_ffn_quant_supported(g_ty));
                if want {
                    Some(mk_gemm_bg(
                        "ResUp",
                        &normed,
                        u_w.as_entire_binding(),
                        gbase + 5,
                        &up_buf,
                    ))
                } else {
                    None
                }
            },
            silu: {
                let want = !(options.ffn_fusion && g_ty == u_ty && fused_ffn_quant_supported(g_ty));
                if want {
                    Some(mk_elem_bg(
                        "ResSilu",
                        &silu_layout,
                        gate_buf.as_entire_binding(),
                        up_buf.as_entire_binding(),
                        &silu_buf,
                        ELEM_SLOT_SILU,
                    ))
                } else {
                    None
                }
            },
            down_resid: mk_resid_bg(
                "ResDownResid",
                &silu_buf,
                d_w.as_entire_binding(),
                gbase + 6,
                &hidden_b,
                &hidden_a,
            ),
        });
    }

    Some((layers, layer_protos, layer_gate_ggml_type))
}
