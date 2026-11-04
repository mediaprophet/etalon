use super::*;
use crate::native_resident::prefill_arena_plan::*;
use crate::native_resident::prefill_arena_slots::*;
impl<P: ResidentPlanProvider> ResidentPlanner<'_, P> {
    pub fn build_prefill_plan(
        &self,
        index: &P::Index,
        key: (u64, u64, u32),
    ) -> Option<Box<PrefillArenaPlan>> {
        let mmap_arc = self.host.model_lease()?;
        let mmap: &[u8] = mmap_arc.as_ref();
        let h = P::hyperparams(index);
        let layout = self.host.plan_layout()?;
        let options = self.host.plan_options();
        let resources = self.host.plan_resources();
        let n_embd = h.n_embd as usize;
        let n_layer = h.n_layer;
        let n_head = h.n_head as usize;
        let n_kv = h.effective_n_kv_head() as usize;
        let head_dim = h.head_dim() as usize;
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv * head_dim;
        let bmax = PREFILL_CHUNK_SIZE;
        if n_layer == 0
            || n_embd == 0
            || n_embd > MAX_HIDDEN_DIM
            || q_dim == 0
            || kv_dim == 0
            || resources.attention_mask_buf.is_none()
            || resources.kv_cache_gpu.is_none()
        {
            return None;
        }
        let use_coop = options.coop_gemv;
        if !use_coop {
            return None; // batched non-coop GEMV grid is not wired; legacy handles it
        }

        let device = self.host.gpu_device().clone();
        let mk_storage = |label: &str, floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: ((floats * 4 + 255) & !255).max(4) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let n_ffn = {
            let t0 = P::tensors(index, 0);
            let gate = t0.ffn_gate.as_ref()?;
            matmul_dims(gate).1
        };
        if n_ffn == 0 || n_ffn > MAX_STACK_GEMM_DIM {
            return None;
        }

        // Batched activation buffers (B_max rows each).
        let hidden_a = mk_storage("PrefillHiddenA", bmax * n_embd);
        let hidden_b = mk_storage("PrefillHiddenB", bmax * n_embd);
        let normed = mk_storage("PrefillNormed", bmax * n_embd);
        let attn_out = mk_storage("PrefillAttnOut", bmax * q_dim.max(n_embd));
        // Residual is fused into O/down GEMV — no separate delta buffer.
        // Separate K/V proj slots so dual GEMV can write both in one dispatch.
        let k_proj = mk_storage("PrefillKProj", bmax * kv_dim);
        let v_proj = mk_storage("PrefillVProj", bmax * kv_dim);
        let q_proj = mk_storage("PrefillQProj", bmax * q_dim);
        let gate_buf = mk_storage("PrefillGate", bmax * n_ffn);
        let up_buf = mk_storage("PrefillUp", bmax * n_ffn);
        let silu_buf = mk_storage("PrefillSilu", bmax * n_ffn);

        // Resident norm weights: slot 2L = attn_norm, 2L+1 = ffn_norm (no output norm — prefill).
        let norm_stride = ((n_embd * 4 + 255) & !255) as wgpu::BufferAddress;
        let norm_res = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PrefillNormWeights"),
            size: norm_stride * (2 * n_layer as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let queue = self.host.gpu_queue().clone();
        let upload_norm = |slot: u64, info: &GgufTensorInfo| -> bool {
            let mut w = [0f32; MAX_HIDDEN_DIM];
            if P::norm_row(mmap, P::data_start(index), info, &mut w) < n_embd {
                return false;
            }
            queue.write_buffer(
                &norm_res,
                slot * norm_stride,
                bytemuck::cast_slice(&w[..n_embd]),
            );
            true
        };

        let param_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PrefillParamArena"),
            size: n_layer as u64 * SLOTS_PER_LAYER * SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind-group layouts + pipelines.
        let gemm_layout = self.host.native_gemm_bind_layout(use_coop).clone();
        let attn_layout = resources.attention_bind_layout.clone();
        let rms_pipe = resources.rms_pipeline?.clone();
        let rms_layout = rms_pipe.get_bind_group_layout(0);
        let silu_layout = resources.elem_silu_mul_bind_layout.clone();
        let mask_buf = resources.attention_mask_buf?.clone();
        let kv_buf = resources.kv_cache_gpu?.clone();

        let binds = Bindings {
            device: &device,
            params: &param_arena,
            attention_params: &param_arena,
            gemm_layout: &gemm_layout,
            attention_layout: &attn_layout,
            kv: &kv_buf,
            mask: &mask_buf,
            layer_f32s: P::layer_stride(&layout) as u64,
        };
        let norm_bind = |slot: u64| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &norm_res,
                offset: slot * norm_stride,
                size: std::num::NonZeroU64::new((n_embd * 4) as u64),
            })
        };
        let slot_off = |l: u64, s: u64| (l * SLOTS_PER_LAYER + s) * SLOT;
        let mk_elem_bg =
            |layout: &wgpu::BindGroupLayout,
             a: wgpu::BindingResource,
             b: wgpu::BindingResource,
             out: &wgpu::Buffer,
             off: u64| binds.elem("PrefillElemBG", layout, a, b, out, off);
        let mk_gemm_bg = |input: &wgpu::Buffer,
                          weight: wgpu::BindingResource,
                          off: u64,
                          out: &wgpu::Buffer,
                          residual: &wgpu::Buffer| {
            binds.gemm(
                "PrefillGemmBG",
                &gemm_layout,
                input,
                weight,
                off,
                out.as_entire_binding(),
                residual,
            )
        };
        let mk_attn_bg = |input: &wgpu::Buffer,
                          weight: &wgpu::Buffer,
                          off: u64,
                          layer: u32,
                          out: &wgpu::Buffer| {
            binds.attention("PrefillAttnBG", input, weight, off, layer, out)
        };

        // Prototype GEMM params (batched-tight: in/out row strides default to n_in/n_out).
        let gemm_proto =
            |ggml_type: u32, n_in: usize, n_out: usize, row_elems: u32, raw_len: usize| {
                GemmGpuParams {
                    n_in: n_in as u32,
                    n_out: n_out as u32,
                    weight_ggml_type: ggml_type,
                    weight_row_elems: row_elems,
                    weight_byte_len: raw_len as u32,
                    n_batch: 1,
                    in_row_stride: 0,
                    out_row_stride: 0,
                }
            };
        let elem_proto = |n: usize, op: u32| ElemGpuParams {
            n: n as u32,
            batch: 1,
            op,
            eps: RMS_NORM_EPS,
            a_row_stride: 0,
            b_row_stride: 0,
            out_row_stride: 0,
            a_slot: 0,
            b_slot: 0,
            out_slot: 0,
            _pad: 0,
        };

        let mut layers = Vec::with_capacity(n_layer as usize);
        let mut protos = Vec::with_capacity(n_layer as usize);
        for l in 0..n_layer {
            let t = P::tensors(index, l);
            let (q_info, k_info, v_info) = (t.attn_q?, t.attn_k?, t.attn_v?);
            let o_info = t.attn_output?;
            let (gate_info, up_info, down_info) = (t.ffn_gate?, t.ffn_up?, t.ffn_down?);
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
            let res = |raw: &[u8]| self.host.resident_weight_buffer(raw.as_ptr() as u64, raw);
            let (q_w, k_w, v_w) = (res(q_raw)?, res(k_raw)?, res(v_raw)?);
            let o_w = res(o_raw)?;
            let ffn_bind =
                |info: &GgufTensorInfo, raw: &[u8]| -> Option<(wgpu::Buffer, u32, u32, u32)> {
                    if let Some(p) = self.host.promote_matrix_to_f16_resident(info, raw) {
                        return Some(p);
                    }
                    let b = res(raw)?;
                    Some((b, info.ggml_type, raw.len() as u32, info.dims[0] as u32))
                };
            let (g_w, g_ty, g_blen, g_row) = ffn_bind(&gate_info, g_raw)?;
            let (u_w, u_ty, u_blen, u_row) = ffn_bind(&up_info, u_raw)?;
            let (d_w, d_ty, d_blen, d_row) = ffn_bind(&down_info, d_raw)?;

            if !upload_norm(2 * l as u64, &attn_norm) || !upload_norm(2 * l as u64 + 1, &ffn_norm) {
                return None;
            }

            // Prototype attention params (batch fields patched per call).
            let mut k_p =
                P::attention_gpu_params(&h, &layout, l, 0, &k_info, k_raw.len(), 1, 1, 0, 0, 0, 0);
            k_p.proj_row_stride = kv_dim as u32;
            let mut v_p =
                P::attention_gpu_params(&h, &layout, l, 0, &v_info, v_raw.len(), 2, 1, 0, 0, 0, 0);
            v_p.proj_row_stride = kv_dim as u32;
            // Q: coop GEMV preprojects into `q_proj`; SDPA reads with proj_row_stride = q_dim.
            let mut q_p = P::attention_gpu_params(
                &h,
                &layout,
                l,
                0,
                &q_info,
                q_raw.len(),
                0,
                1,
                0,
                0,
                0,
                q_dim as u32,
            );
            q_p.proj_row_stride = q_dim as u32;

            protos.push(PrefillLayerProtos {
                gemm: [
                    gemm_proto(
                        k_info.ggml_type,
                        n_embd,
                        kv_dim,
                        k_info.dims[0] as u32,
                        k_raw.len(),
                    ),
                    gemm_proto(
                        v_info.ggml_type,
                        n_embd,
                        kv_dim,
                        v_info.dims[0] as u32,
                        v_raw.len(),
                    ),
                    gemm_proto(
                        q_info.ggml_type,
                        n_embd,
                        q_dim,
                        q_info.dims[0] as u32,
                        q_raw.len(),
                    ),
                    gemm_proto(
                        o_info.ggml_type,
                        q_dim,
                        n_embd,
                        o_info.dims[0] as u32,
                        o_raw.len(),
                    ),
                    gemm_proto(g_ty, n_embd, n_ffn, g_row, g_blen as usize),
                    gemm_proto(u_ty, n_embd, n_ffn, u_row, u_blen as usize),
                    gemm_proto(d_ty, n_ffn, n_embd, d_row, d_blen as usize),
                ],
                attn: [k_p, v_p, q_p],
                elem: [
                    elem_proto(n_embd, ELEM_OP_RMS_NORM),
                    elem_proto(n_embd, ELEM_OP_ADD_RESIDUAL),
                    elem_proto(n_embd, ELEM_OP_RMS_NORM),
                    elem_proto(n_ffn, ELEM_OP_SILU_MUL),
                    elem_proto(n_embd, ELEM_OP_ADD_RESIDUAL),
                ],
            });

            let lu = l as u64;
            // Fused SwiGLU when gate/up share a supported quant (same gate as resident_decode).
            let (fused_ffn, dual_kv) = super::prefill_fusions::build_fusions(
                &device,
                &resources,
                &param_arena,
                lu,
                use_coop,
                options,
                g_ty,
                u_ty,
                &k_info,
                &v_info,
                &g_w,
                &u_w,
                &k_w,
                &v_w,
                &normed,
                &silu_buf,
                &k_proj,
                &v_proj,
            );
            layers.push(PrefillLayerBinds {
                rms_attn: mk_elem_bg(
                    &rms_layout,
                    hidden_a.as_entire_binding(),
                    norm_bind(2 * lu),
                    &normed,
                    slot_off(lu, S_ELEM_RMS_ATTN),
                ),
                dual_kv,
                k_gemm: mk_gemm_bg(
                    &normed,
                    k_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_K),
                    &k_proj,
                    &normed,
                ),
                k_write: mk_attn_bg(&k_proj, &k_w, slot_off(lu, S_ATTN_KW), l, &attn_out),
                v_gemm: mk_gemm_bg(
                    &normed,
                    v_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_V),
                    &v_proj,
                    &normed,
                ),
                v_write: mk_attn_bg(&v_proj, &v_w, slot_off(lu, S_ATTN_VW), l, &attn_out),
                q_gemm: mk_gemm_bg(
                    &normed,
                    q_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_Q),
                    &q_proj,
                    &normed,
                ),
                // hidden = precomputed Q; weight unused when proj_row_stride != 0.
                q: mk_attn_bg(&q_proj, &q_w, slot_off(lu, S_ATTN_Q), l, &attn_out),
                // O·attn + residual(hidden_a) → hidden_b
                o_resid: mk_gemm_bg(
                    &attn_out,
                    o_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_O),
                    &hidden_b,
                    &hidden_a,
                ),
                rms_ffn: mk_elem_bg(
                    &rms_layout,
                    hidden_b.as_entire_binding(),
                    norm_bind(2 * lu + 1),
                    &normed,
                    slot_off(lu, S_ELEM_RMS_FFN),
                ),
                fused_ffn,
                gate: mk_gemm_bg(
                    &normed,
                    g_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_GATE),
                    &gate_buf,
                    &normed,
                ),
                up: mk_gemm_bg(
                    &normed,
                    u_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_UP),
                    &up_buf,
                    &normed,
                ),
                silu: mk_elem_bg(
                    &silu_layout,
                    gate_buf.as_entire_binding(),
                    up_buf.as_entire_binding(),
                    &silu_buf,
                    slot_off(lu, S_ELEM_SILU),
                ),
                // Down·mid + residual(hidden_b) → hidden_a
                down_resid: mk_gemm_bg(
                    &silu_buf,
                    d_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_DOWN),
                    &hidden_a,
                    &hidden_b,
                ),
            });
        }

        let use_fused = layers.iter().any(|l| l.fused_ffn.is_some());
        let use_dual = layers.iter().any(|l| l.dual_kv.is_some());
        // 11 with fused; -1 for dual K+V vs separate.
        let passes = if use_fused {
            if use_dual {
                10
            } else {
                11
            }
        } else if use_dual {
            12
        } else {
            13
        };
        self.host.event(PlanEvent::PrefillBuilt {
            layers: n_layer,
            passes,
            fused: use_fused,
            dual: use_dual,
        });
        Some(Box::new(PrefillArenaPlan {
            map_status: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            key,
            n_embd,
            n_ffn,
            kv_dim,
            q_dim,
            n_head: n_head as u32,
            n_kv_head: n_kv as u32,
            max_context: P::max_context(&layout),
            protos,
            layers,
            param_arena,
            scratch: vec![0u8; (n_layer as u64 * SLOTS_PER_LAYER * SLOT) as usize],
            hidden_a,
            use_coop,
        }))
    }
}
