use super::*;
use crate::native_resident::verify_arena_plan::*;
use crate::native_resident::verify_arena_slots::*;
impl<P: ResidentPlanProvider> ResidentPlanner<'_, P> {
    pub fn build_verify_plan(
        &self,
        index: &P::Index,
        key: (u64, u64, u32),
    ) -> Option<Box<VerifyArenaPlan>> {
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
        let bmax = VERIFY_BATCH_MAX;
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
            return None;
        }

        // Logits projection must be resident (chunked GEMV shares one encoder).
        let logits_buf = resources.mc8_logits_resident_buf?.clone();
        let logits_row_bytes = resources.mc8_logits_row_bytes as u64;
        let logits_info = P::logits_info(index)?;
        let (logits_in, vocab) = matmul_dims(&logits_info);
        if logits_in != n_embd
            || vocab == 0
            || !ggml_gpu_gemm_supported(logits_info.ggml_type)
            || logits_row_bytes == 0
        {
            return None;
        }
        let out_norm_info = P::output_norm_info(index)?;

        let device = self.host.gpu_device().clone();
        let mk_storage = |label: &str, floats: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: ((floats * 4 + 255) & !255).max(4) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
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

        let empty = wgpu::BufferUsages::empty();
        let hidden_a = mk_storage("VerifyHiddenA", bmax * n_embd, empty);
        let hidden_b = mk_storage("VerifyHiddenB", bmax * n_embd, empty);
        let normed = mk_storage("VerifyNormed", bmax * n_embd, empty);
        let attn_out = mk_storage("VerifyAttnOut", bmax * q_dim.max(n_embd), empty);
        let delta = mk_storage("VerifyDelta", bmax * n_embd, empty);
        let kv_proj = mk_storage("VerifyKvProj", bmax * kv_dim, empty);
        let gate_buf = mk_storage("VerifyGate", bmax * n_ffn, empty);
        let up_buf = mk_storage("VerifyUp", bmax * n_ffn, empty);
        let silu_buf = mk_storage("VerifySilu", bmax * n_ffn, empty);
        let logits_out = mk_storage(
            "VerifyLogitsOut",
            bmax * vocab,
            wgpu::BufferUsages::COPY_SRC,
        );
        let logits_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VerifyLogitsStaging"),
            size: ((bmax * vocab * 4 + 255) & !255).max(4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Resident norm weights: 2L = attn_norm, 2L+1 = ffn_norm, 2·n_layer = output_norm.
        let norm_stride = ((n_embd * 4 + 255) & !255) as wgpu::BufferAddress;
        let norm_res = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VerifyNormWeights"),
            size: norm_stride * (2 * n_layer as u64 + 1),
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

        let full_chunks = vocab.div_ceil(VOCAB_CHUNK_ROWS);
        let tail_rms_slot = n_layer as u64 * SLOTS_PER_LAYER;
        let tail_gemm_base = tail_rms_slot + 1;
        let total_slots = tail_gemm_base + full_chunks as u64;
        let param_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VerifyParamArena"),
            size: total_slots * SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gemm_layout = self.host.native_gemm_bind_layout(use_coop).clone();
        let attn_layout = resources.attention_bind_layout.clone();
        let rms_pipe = resources.rms_pipeline?.clone();
        let add_pipe = resources.add_pipeline?.clone();
        let rms_layout = rms_pipe.get_bind_group_layout(0);
        let add_layout = add_pipe.get_bind_group_layout(0);
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
             off: u64| binds.elem("VerifyElemBG", layout, a, b, out, off);
        let mk_gemm_bg = |input: &wgpu::Buffer,
                          weight: wgpu::BindingResource,
                          off: u64,
                          out: wgpu::BindingResource| {
            binds.gemm("VerifyGemmBG", &gemm_layout, input, weight, off, out, input)
        };
        let mk_attn_bg = |input: &wgpu::Buffer,
                          weight: &wgpu::Buffer,
                          off: u64,
                          layer: u32,
                          out: &wgpu::Buffer| {
            binds.attention("VerifyAttnBG", input, weight, off, layer, out)
        };

        let gemm_proto =
            |info: &GgufTensorInfo, n_in: usize, n_out: usize, raw_len: usize| GemmGpuParams {
                n_in: n_in as u32,
                n_out: n_out as u32,
                weight_ggml_type: info.ggml_type,
                weight_row_elems: info.dims[0] as u32,
                weight_byte_len: raw_len as u32,
                n_batch: 1,
                in_row_stride: 0,
                out_row_stride: 0,
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
            let (k_in, k_out) = matmul_dims(&k_info);
            let (v_in, v_out) = matmul_dims(&v_info);
            let (o_in, o_out) = matmul_dims(&o_info);
            let (g_in, g_out) = matmul_dims(&gate_info);
            let (u_in, u_out) = matmul_dims(&up_info);
            let (d_in, d_out) = matmul_dims(&down_info);
            if k_in != n_embd
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
            let (o_w, g_w, u_w, d_w) = (res(o_raw)?, res(g_raw)?, res(u_raw)?, res(d_raw)?);

            if !upload_norm(2 * l as u64, &attn_norm) || !upload_norm(2 * l as u64 + 1, &ffn_norm) {
                return None;
            }

            let mut k_p =
                P::attention_gpu_params(&h, &layout, l, 0, &k_info, k_raw.len(), 1, 1, 0, 0, 0, 0);
            k_p.proj_row_stride = kv_dim as u32;
            let mut v_p =
                P::attention_gpu_params(&h, &layout, l, 0, &v_info, v_raw.len(), 2, 1, 0, 0, 0, 0);
            v_p.proj_row_stride = kv_dim as u32;
            let q_p = P::attention_gpu_params(
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

            protos.push(VerifyLayerProtos {
                gemm: [
                    gemm_proto(&k_info, n_embd, kv_dim, k_raw.len()),
                    gemm_proto(&v_info, n_embd, kv_dim, v_raw.len()),
                    gemm_proto(&o_info, q_dim, n_embd, o_raw.len()),
                    gemm_proto(&gate_info, n_embd, n_ffn, g_raw.len()),
                    gemm_proto(&up_info, n_embd, n_ffn, u_raw.len()),
                    gemm_proto(&down_info, n_ffn, n_embd, d_raw.len()),
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
            layers.push(VerifyLayerBinds {
                rms_attn: mk_elem_bg(
                    &rms_layout,
                    hidden_a.as_entire_binding(),
                    norm_bind(2 * lu),
                    &normed,
                    slot_off(lu, S_ELEM_RMS_ATTN),
                ),
                k_gemm: mk_gemm_bg(
                    &normed,
                    k_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_K),
                    kv_proj.as_entire_binding(),
                ),
                k_write: mk_attn_bg(&kv_proj, &k_w, slot_off(lu, S_ATTN_KW), l, &attn_out),
                v_gemm: mk_gemm_bg(
                    &normed,
                    v_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_V),
                    kv_proj.as_entire_binding(),
                ),
                v_write: mk_attn_bg(&kv_proj, &v_w, slot_off(lu, S_ATTN_VW), l, &attn_out),
                q: mk_attn_bg(&normed, &q_w, slot_off(lu, S_ATTN_Q), l, &attn_out),
                o: mk_gemm_bg(
                    &attn_out,
                    o_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_O),
                    delta.as_entire_binding(),
                ),
                add1: mk_elem_bg(
                    &add_layout,
                    hidden_a.as_entire_binding(),
                    delta.as_entire_binding(),
                    &hidden_b,
                    slot_off(lu, S_ELEM_ADD1),
                ),
                rms_ffn: mk_elem_bg(
                    &rms_layout,
                    hidden_b.as_entire_binding(),
                    norm_bind(2 * lu + 1),
                    &normed,
                    slot_off(lu, S_ELEM_RMS_FFN),
                ),
                gate: mk_gemm_bg(
                    &normed,
                    g_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_GATE),
                    gate_buf.as_entire_binding(),
                ),
                up: mk_gemm_bg(
                    &normed,
                    u_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_UP),
                    up_buf.as_entire_binding(),
                ),
                silu: mk_elem_bg(
                    &silu_layout,
                    gate_buf.as_entire_binding(),
                    up_buf.as_entire_binding(),
                    &silu_buf,
                    slot_off(lu, S_ELEM_SILU),
                ),
                down: mk_gemm_bg(
                    &silu_buf,
                    d_w.as_entire_binding(),
                    slot_off(lu, S_GEMM_DOWN),
                    delta.as_entire_binding(),
                ),
                add2: mk_elem_bg(
                    &add_layout,
                    hidden_b.as_entire_binding(),
                    delta.as_entire_binding(),
                    &hidden_a,
                    slot_off(lu, S_ELEM_ADD2),
                ),
            });
        }

        // Output norm weight (slot 2·n_layer) + output-RMSNorm bind group (hidden_a → normed).
        if !upload_norm(2 * n_layer as u64, &out_norm_info) {
            return None;
        }
        let rms_out = mk_elem_bg(
            &rms_layout,
            hidden_a.as_entire_binding(),
            norm_bind(2 * n_layer as u64),
            &normed,
            tail_rms_slot * SLOT,
        );

        // Per-chunk logits GEMV: normed (B × n_embd) → logits_out chunk slice. The chunk's output
        // binding is offset by `row_start` so the GEMV's `m*vocab + row` lands at the global vocab
        // position (chunk_start is a multiple of VOCAB_CHUNK_ROWS ⇒ 256-byte aligned).
        let mut chunks = Vec::with_capacity(full_chunks);
        for c in 0..full_chunks {
            let row_start = c * VOCAB_CHUNK_ROWS;
            let rows = VOCAB_CHUNK_ROWS.min(vocab - row_start);
            if rows > resources.gemm_max_out_dim as usize {
                return None;
            }
            let byte_len = rows as u64 * logits_row_bytes;
            let slot = tail_gemm_base + c as u64;
            // Params are written into `scratch` every call (the full-arena upload in run overwrites the
            // whole buffer), so no build-time write_buffer here — it would just be clobbered.
            let params = GemmGpuParams {
                n_in: n_embd as u32,
                n_out: rows as u32,
                weight_ggml_type: logits_info.ggml_type,
                weight_row_elems: logits_info.dims[0] as u32,
                weight_byte_len: byte_len as u32,
                n_batch: 1,
                in_row_stride: 0,
                out_row_stride: vocab as u32,
            };
            let weight = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &logits_buf,
                offset: row_start as u64 * logits_row_bytes,
                size: std::num::NonZeroU64::new(byte_len),
            });
            let out_res = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &logits_out,
                offset: (row_start * 4) as wgpu::BufferAddress,
                size: std::num::NonZeroU64::new(((bmax * vocab - row_start) * 4) as u64),
            });
            let gemm = mk_gemm_bg(&normed, weight, slot * SLOT, out_res);
            chunks.push(VerifyChunk {
                gemm,
                rows: rows as u32,
                slot,
                params,
            });
        }

        self.host.event(PlanEvent::VerifyBuilt {
            layers: n_layer,
            chunks: full_chunks,
            batch: bmax,
        });
        Some(Box::new(VerifyArenaPlan {
            map_status: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            key,
            n_embd,
            n_ffn,
            kv_dim,
            n_head: n_head as u32,
            n_kv_head: n_kv as u32,
            vocab,
            max_context: P::max_context(&layout),
            protos,
            layers,
            rms_out,
            tail_rms_slot,
            chunks,
            param_arena,
            scratch: vec![0u8; (total_slots * SLOT) as usize],
            embedding_scratch: vec![0f32; bmax * n_embd],
            hidden_a,
            logits_out,
            logits_staging,
            use_coop,
        }))
    }
}
