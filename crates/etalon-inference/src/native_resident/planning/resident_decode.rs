use super::*;
use crate::native_resident::resident_decode_plan::*;
use crate::native_resident::resident_decode_slots::*;
impl<P: ResidentPlanProvider> ResidentPlanner<'_, P> {
    pub fn build_resident_plan(
        &self,
        index: &P::Index,
        key: (u64, u64, u32),
    ) -> Option<Box<ResidentDecodePlan>> {
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
        if n_layer == 0
            || n_embd == 0
            || n_embd > MAX_HIDDEN_DIM
            || q_dim == 0
            || kv_dim == 0
            || resources.output_topk_pipeline.is_none()
            || resources.attention_mask_buf.is_none()
        {
            return None;
        }

        // Logits projection must be resident (per-chunk re-upload cannot share
        // one encoder), and every weight must be GPU-eligible.
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

        let use_coop = options.coop_gemv;
        let device = self.host.gpu_device().clone();
        let mk_storage = |label: &str, floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: ((floats * 4 + 255) & !255).max(4) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        // Normed hidden is also COPY_SRC for the sampler-compatible readback path.
        let mk_storage_copy_src = |label: &str, floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: ((floats * 4 + 255) & !255).max(4) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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

        let hidden_a = mk_storage("ResidentHiddenA", n_embd);
        let hidden_b = mk_storage("ResidentHiddenB", n_embd);
        let normed = mk_storage_copy_src("ResidentNormed", n_embd);
        let attn_out = mk_storage("ResidentAttnOut", q_dim.max(n_embd));
        // Residual add is fused into coop_gemv_residual (no separate delta buffer).
        // Separate K/V proj slots so dual GEMV can write both in one dispatch.
        let k_proj = mk_storage("ResidentKProj", kv_dim);
        let v_proj = mk_storage("ResidentVProj", kv_dim);
        // Q preprojection target (coop GEMV); attention reads with proj_row_stride = q_dim.
        let q_proj = mk_storage("ResidentQProj", q_dim);
        let gate_buf = mk_storage("ResidentGate", n_ffn);
        let up_buf = mk_storage("ResidentUp", n_ffn);
        let silu_buf = mk_storage("ResidentSilu", n_ffn);
        let logits_chunk = mk_storage("ResidentLogitsChunk", RESIDENT_LOGITS_CHUNK);

        // All layers' norm weights resident: slot 2L = attn_norm, 2L+1 = ffn_norm,
        // slot 2·n_layer = output_norm. 256-aligned stride.
        let norm_stride = ((n_embd * 4 + 255) & !255) as wgpu::BufferAddress;
        let norm_res = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ResidentNormWeights"),
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

        // Static uniform arena: per-layer GEMM slots + output-chunk GEMM slots
        // + shared elem slots + per-chunk topk slots.
        let full_chunks = vocab.div_ceil(RESIDENT_LOGITS_CHUNK);
        let gemm_slots = n_layer as u64 * GEMM_SLOTS_PER_LAYER + full_chunks as u64;
        let static_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ResidentStaticParams"),
            size: (gemm_slots + ELEM_SLOTS + full_chunks as u64) * SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let elem_base = gemm_slots * SLOT;
        let topk_base = (gemm_slots + ELEM_SLOTS) * SLOT;

        let attn_dyn_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ResidentAttnDynParams"),
            size: n_layer as u64 * ATTN_SLOTS_PER_LAYER * SLOT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Candidate staging (own buffer — never contends with the legacy path's).
        let block_size = options.topk_block_size;
        let total_cands = vocab.div_ceil(block_size);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ResidentTopkStaging"),
            size: ((total_cands * 8).max(8)) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hidden_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ResidentHiddenStaging"),
            size: ((n_embd * 4).max(16)) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared elem params (written once).
        let elem_p = |n: usize, op: u32| ElemGpuParams {
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
        queue.write_buffer(
            &static_arena,
            elem_base + ELEM_SLOT_RMS,
            bytemuck::bytes_of(&elem_p(n_embd, ELEM_OP_RMS_NORM)),
        );
        queue.write_buffer(
            &static_arena,
            elem_base + ELEM_SLOT_ADD,
            bytemuck::bytes_of(&elem_p(n_embd, ELEM_OP_ADD_RESIDUAL)),
        );
        queue.write_buffer(
            &static_arena,
            elem_base + ELEM_SLOT_SILU,
            bytemuck::bytes_of(&elem_p(n_ffn, ELEM_OP_SILU_MUL)),
        );

        // Bind-group layouts + helpers.
        let gemm_layout = self.host.native_gemm_bind_layout(use_coop).clone();
        let attn_layout = resources.attention_bind_layout.clone();
        let rms_pipe = resources.rms_pipeline?.clone();
        let rms_layout = rms_pipe.get_bind_group_layout(0);
        let silu_layout = resources.elem_silu_mul_bind_layout.clone();
        let topk_layout = resources.output_topk_bind_layout?.clone();
        let mask_buf = resources.attention_mask_buf?.clone();
        let kv_buf = resources.kv_cache_gpu?.clone();
        let cand_val = resources.topk_cand_val_buf?.clone();
        let cand_idx = resources.topk_cand_idx_buf?.clone();

        let binds = Bindings {
            device: &device,
            params: &static_arena,
            attention_params: &attn_dyn_arena,
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
        let tp_sz = std::num::NonZeroU64::new(16);
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
        let gemm_params =
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

        let (layers, layer_protos, layer_gate_ggml_type) = super::decode_layers::build_layers(
            self.host,
            index,
            mmap,
            &h,
            &layout,
            options,
            &resources,
            &binds,
            &rms_layout,
            &silu_layout,
            &norm_res,
            norm_stride,
            elem_base,
            n_ffn,
            super::decode_layers::DecodeActivations {
                hidden_a: &hidden_a,
                hidden_b: &hidden_b,
                normed: &normed,
                attn_out: &attn_out,
                k_proj: &k_proj,
                v_proj: &v_proj,
                q_proj: &q_proj,
                gate_buf: &gate_buf,
                up_buf: &up_buf,
                silu_buf: &silu_buf,
            },
            upload_norm,
        )?;

        // Output norm weights + bind group (reads the final hidden_a).
        if !upload_norm(2 * n_layer as u64, &out_norm_info) {
            return None;
        }
        let rms_out = mk_elem_bg(
            "ResRmsOut",
            &rms_layout,
            hidden_a.as_entire_binding(),
            norm_bind(2 * n_layer as u64),
            &normed,
            ELEM_SLOT_RMS,
        );

        // Output chunks: logits GEMV (resident weight sub-range) + top-1 reduce.
        // cand_base packs multi-chunk candidates contiguously for one mega-pass.
        let mut out_chunks = Vec::with_capacity(full_chunks);
        let mut cand_base_acc = 0u32;
        for c in 0..full_chunks {
            let row_start = c * RESIDENT_LOGITS_CHUNK;
            let rows = RESIDENT_LOGITS_CHUNK.min(vocab - row_start);
            // Resident logits buffer is sized to RESIDENT_LOGITS_CHUNK (not stack gemm max).
            if rows > RESIDENT_LOGITS_CHUNK {
                return None;
            }
            let cand_count = rows.div_ceil(block_size);
            let byte_len = rows as u64 * logits_row_bytes;
            let gemm_slot = n_layer as u64 * GEMM_SLOTS_PER_LAYER + c as u64;
            queue.write_buffer(
                &static_arena,
                gemm_slot * SLOT,
                bytemuck::bytes_of(&gemm_params(
                    logits_info.ggml_type,
                    n_embd,
                    rows,
                    logits_info.dims[0] as u32,
                    byte_len as usize,
                )),
            );
            let tparams =
                topk_params_bytes_with_base(rows as u32, 1, block_size as u32, cand_base_acc);
            queue.write_buffer(&static_arena, topk_base + c as u64 * SLOT, &tparams);
            let weight = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &logits_buf,
                offset: row_start as u64 * logits_row_bytes,
                size: std::num::NonZeroU64::new(byte_len),
            });
            let gemm = mk_gemm_bg("ResLogits", &normed, weight, gemm_slot, &logits_chunk);
            let topk = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ResTopk"),
                layout: &topk_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: logits_chunk.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ubind(&static_arena, topk_base + c as u64 * SLOT, tp_sz),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cand_val.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cand_idx.as_entire_binding(),
                    },
                ],
            });
            out_chunks.push(OutChunk {
                gemm,
                topk,
                rows: rows as u32,
                cand_count,
            });
            cand_base_acc += cand_count as u32;
        }

        let dyn_bytes = (n_layer as u64 * ATTN_SLOTS_PER_LAYER * SLOT) as usize;
        let use_fused_ffn = !layers.is_empty() && layers.iter().all(|lb| lb.fused_ffn.is_some());
        // Residual-fused O/down + triple QKV when SoA: ~9 dispatches/layer with fused FFN.
        let use_triple = layers.iter().any(|lb| lb.triple_qkv.is_some());
        let use_dual = layers.iter().any(|lb| lb.dual_kv.is_some());
        let passes_per_layer = if use_fused_ffn {
            if use_triple {
                9
            } else if use_dual {
                10
            } else {
                11
            }
        } else {
            13
        };
        let passes_token = n_layer as usize * passes_per_layer + 1 + out_chunks.len() * 2;
        self.host.set_ffn_fusion_in_resident(use_fused_ffn);
        // Layer weights may be Q4_K_SOA while logits stay Q6_K (3B .soa.p64: 2520 B/row logits).
        let is_q4_soa = layer_gate_ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
            || logits_info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA;
        // Residual multirow: opt-in only (A/B lost vs 1-row on A2000 3B).
        // Logits always use multirow geometry when n_out > 60k (device max WG dim).
        let use_multirow = use_coop && is_q4_soa && options.multirow;
        let use_warp = use_coop && is_q4_soa && !use_multirow && options.warp_gemv;
        // FFN multi-row: opt-in only (A/B lost on A2000 3B SoA).
        let use_ffn_warp = use_coop && is_q4_soa && options.ffn_warp;
        let use_ffn_mr = use_coop && is_q4_soa && !use_ffn_warp && options.ffn_mr;
        self.host.event(PlanEvent::DecodeBuilt {
            layers: n_layer,
            passes: passes_token,
            fused: use_fused_ffn,
            triple: use_triple,
            dual: use_dual,
            ffn_mr: use_ffn_mr,
            multirow: use_multirow,
            ffn_warp: use_ffn_warp,
        });
        Some(Box::new(ResidentDecodePlan {
            map_status: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            key,
            n_embd,
            n_ffn,
            kv_dim,
            q_dim,
            n_head: n_head as u32,
            n_kv_head: n_kv as u32,
            max_context: P::max_context(&layout),
            layer_protos,
            layers,
            rms_out,
            out_chunks,
            total_cands,
            attn_dyn_arena,
            dyn_scratch: vec![0u8; dyn_bytes],
            hidden_a,
            normed,
            staging,
            hidden_staging,
            use_coop,
            use_multirow,
            use_warp,
            use_ffn_mr,
            use_ffn_warp,
            use_fused_ffn,
            dispatches_per_token: passes_token as u32,
            readback_bytes_per_token: (total_cands * std::mem::size_of::<f32>()
                + total_cands * std::mem::size_of::<u32>())
                as u32,
        }))
    }
}
