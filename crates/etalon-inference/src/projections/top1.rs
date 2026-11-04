use super::*;
use crate::ordinary::{gpu_params::*, quant_support::ggml_gpu_gemm_supported};
impl<P: ProjectionGpuProvider> ProjectionGpuEngine<'_, P> {
    pub fn dispatch_output_top1_chunked(
        &self,
        index: &P::Index,
        hidden: &[f32],
        emb_dim: usize,
    ) -> Option<StreamingArgmaxResult> {
        let descriptor = P::logits(index)?;
        let info = &descriptor;
        let (n_in, vocab_size) = crate::ordinary::matmul_dims(info);
        if n_in == 0 || vocab_size == 0 || n_in > emb_dim || n_in > hidden.len() {
            return None;
        }
        let topk_pipeline = self.output_topk_pipeline.as_ref()?;
        let input_buf = self.gemm_input_buf.as_ref()?;
        let weight_buf = self.gemm_weight_buf.as_ref()?;
        let output_buf = self.gemm_output_buf.as_ref()?;
        let params_buf = self.gemm_params_buf.as_ref()?;
        let topk_params_buf = self.topk_params_buf.as_ref()?;
        let cand_val = self.topk_cand_val_buf.as_ref()?;
        let cand_idx = self.topk_cand_idx_buf.as_ref()?;
        let staging = self.topk_cand_staging.as_ref()?;
        let mmap = self.host.model_bytes()?;

        let use_coop = self.host.coop_gemv_enabled();
        let use_mr = use_coop && info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA;
        let gemm_pipeline: &wgpu::ComputePipeline = if use_mr {
            &self.coop_gemv_mr_pipeline
        } else if use_coop {
            &self.coop_gemv_pipeline
        } else {
            &self.pipeline
        };
        let gemm_layout = self.host.native_gemm_bind_layout(use_coop).clone();
        let topk_layout = self.output_topk_bind_layout.as_ref()?;
        let topk_bind = self
            .host
            .gpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Top1Bind"),
                layout: topk_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: topk_params_buf.as_entire_binding(),
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
        let block_size = super::topk::TOPK_BLOCK_SIZE;
        let full_chunks = vocab_size.div_ceil(VOCAB_CHUNK_ROWS);
        const MAX_CHUNKS: usize = 32;
        if full_chunks > MAX_CHUNKS {
            return None;
        }
        let total_cands = vocab_size.div_ceil(block_size);
        let cand_capacity = VOCAB_CHUNK_ROWS
            .div_ceil(block_size)
            .max(1)
            .saturating_mul(super::topk::TOPK_MAX_K);
        if total_cands == 0 || total_cands > cand_capacity {
            return None;
        }
        let resident_logits = self.mc8_logits_resident_buf.as_ref();
        let resident_row_bytes = self.mc8_logits_row_bytes as u64;

        self.host
            .gpu_queue()
            .write_buffer(input_buf, 0, bytemuck::cast_slice(&hidden[..n_in]));

        // Fast path: resident logits → ONE submit for all vocab chunks (no per-chunk fence).
        // Shared uniform buffers force multi-submit when weights must be re-uploaded each chunk.
        if let Some(res_buf) = resident_logits {
            if !(ggml_gpu_gemm_supported(info.ggml_type) && n_in <= MAX_STACK_GEMM_IN) {
                return None;
            }
            // 256-byte aligned slots (wgpu min uniform offset) for per-chunk params.
            const SLOT: usize = 256;
            let gemm_slot = std::mem::size_of::<GemmGpuParams>().max(32);
            let mut gemm_slab = [0u8; MAX_CHUNKS * SLOT];
            let mut topk_slab = [0u8; MAX_CHUNKS * SLOT];
            let mut chunk_meta = [(0u32, 0u32, 0u32); MAX_CHUNKS]; // rows, weight_bytes, n_blocks
            for chunk_idx in 0..full_chunks {
                let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
                let chunk_rows = VOCAB_CHUNK_ROWS.min(vocab_size - row_start);
                if chunk_rows > self.gemm_max_out_dim as usize {
                    return None;
                }
                let weight_byte_len = (chunk_rows as u64 * resident_row_bytes) as u32;
                let gparams = GemmGpuParams {
                    n_in: n_in as u32,
                    n_out: chunk_rows as u32,
                    weight_ggml_type: info.ggml_type,
                    weight_row_elems: info.dims[0] as u32,
                    weight_byte_len,
                    n_batch: 1,
                    in_row_stride: 0,
                    out_row_stride: 0,
                };
                let gp = bytemuck::bytes_of(&gparams);
                let go = chunk_idx * SLOT;
                gemm_slab[go..go + gemm_slot.min(gp.len())]
                    .copy_from_slice(&gp[..gemm_slot.min(gp.len())]);
                let tparams =
                    super::topk::topk_params_bytes(chunk_rows as u32, 1, block_size as u32);
                topk_slab[go..go + tparams.len()].copy_from_slice(&tparams);
                let num_blocks = chunk_rows.div_ceil(block_size) as u32;
                chunk_meta[chunk_idx] = (chunk_rows as u32, weight_byte_len, num_blocks);
            }
            // Two fixed aligned regions share the retained uniform allocation.
            const GEMM_REGION: u64 = (MAX_CHUNKS * SLOT) as u64;
            let gemm_multi = topk_params_buf;
            let topk_multi = topk_params_buf;
            if topk_params_buf.size() < GEMM_REGION * 2 {
                return None;
            }
            self.host.gpu_queue().write_buffer(
                gemm_multi,
                GEMM_REGION,
                &gemm_slab[..full_chunks * SLOT],
            );
            self.host
                .gpu_queue()
                .write_buffer(topk_multi, 0, &topk_slab[..full_chunks * SLOT]);

            let mut encoder =
                self.host
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Top1FusedEncoder"),
                    });
            let mut cand_offset = 0usize;
            for (chunk_idx, &(chunk_rows, _wlen, num_blocks)) in
                chunk_meta.iter().take(full_chunks).enumerate()
            {
                let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
                let byte_len = chunk_rows as u64 * resident_row_bytes;
                let weight_resource = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: res_buf,
                    offset: row_start as u64 * resident_row_bytes,
                    size: std::num::NonZeroU64::new(byte_len),
                });
                let go = (chunk_idx * SLOT) as u64;
                let gsize = std::num::NonZeroU64::new(std::mem::size_of::<GemmGpuParams>() as u64);
                let tsize = std::num::NonZeroU64::new(16);
                let gemm_entries = [
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: gemm_multi,
                            offset: GEMM_REGION + go,
                            size: gsize,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: input_buf.as_entire_binding(),
                    },
                ];
                let gemm_bind =
                    self.host
                        .gpu_device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Top1GemmBindFused"),
                            layout: &gemm_layout,
                            entries: &gemm_entries[..if use_coop { 5 } else { 4 }],
                        });
                let topk_bind_c =
                    self.host
                        .gpu_device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Top1TopkBindFused"),
                            layout: topk_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: output_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: topk_multi,
                                        offset: go,
                                        size: tsize,
                                    }),
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
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Top1GemmPass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(gemm_pipeline);
                    cpass.set_bind_group(0, &gemm_bind, &[]);
                    if use_mr {
                        cpass.dispatch_workgroups(self.host.coop_gemv_workgroups(chunk_rows), 1, 1);
                    } else if use_coop {
                        cpass.dispatch_workgroups(chunk_rows, 1, 1);
                    } else {
                        cpass.dispatch_workgroups((chunk_rows + 63) / 64, 1, 1);
                    }
                }
                {
                    let mut tpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Top1ReducePass"),
                        timestamp_writes: None,
                    });
                    tpass.set_pipeline(topk_pipeline);
                    tpass.set_bind_group(0, &topk_bind_c, &[]);
                    tpass.dispatch_workgroups(num_blocks, 1, 1);
                }
                let cand_count = num_blocks as usize;
                let cand_bytes = (cand_count * 4) as wgpu::BufferAddress;
                let val_dst = (cand_offset * 4) as wgpu::BufferAddress;
                let idx_dst = ((total_cands + cand_offset) * 4) as wgpu::BufferAddress;
                encoder.copy_buffer_to_buffer(cand_val, 0, staging, val_dst, cand_bytes);
                encoder.copy_buffer_to_buffer(cand_idx, 0, staging, idx_dst, cand_bytes);
                cand_offset += cand_count;
            }
            self.host.resolve(&mut encoder);
            self.host.submit(encoder.finish());
            self.host.accumulate(Phase::OutputTopk);
        } else {
            // Legacy: per-chunk upload + submit (shared weight staging buffer).
            let mut cand_offset = 0usize;
            for chunk_idx in 0..full_chunks {
                let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
                let chunk_rows = VOCAB_CHUNK_ROWS.min(vocab_size - row_start);
                let raw = etalon_model::ggml::fetch_tensor_row_range_bytes(
                    mmap,
                    P::data_start(index),
                    info,
                    row_start,
                    chunk_rows,
                )
                .ok()?;
                if !(ggml_gpu_gemm_supported(info.ggml_type)
                    && n_in <= MAX_STACK_GEMM_IN
                    && chunk_rows <= self.gemm_max_out_dim as usize
                    && raw.len() <= self.max_tensor_bytes)
                {
                    return None;
                }
                let byte_len = raw.len() as u32;
                let resident = if self.host.resident_weights_enabled() {
                    self.host.resident_weight_buffer(raw.as_ptr() as u64, raw)
                } else {
                    None
                };
                let weight_binding: &wgpu::Buffer = match resident.as_ref() {
                    Some(r) => r,
                    None => {
                        self.host.write_weight_words(raw, self.max_tensor_bytes);
                        weight_buf
                    }
                };

                let gparams = GemmGpuParams {
                    n_in: n_in as u32,
                    n_out: chunk_rows as u32,
                    weight_ggml_type: info.ggml_type,
                    weight_row_elems: info.dims[0] as u32,
                    weight_byte_len: byte_len,
                    n_batch: 1,
                    in_row_stride: 0,
                    out_row_stride: 0,
                };
                self.host
                    .gpu_queue()
                    .write_buffer(params_buf, 0, bytemuck::bytes_of(&gparams));
                let tparams =
                    super::topk::topk_params_bytes(chunk_rows as u32, 1, block_size as u32);
                self.host
                    .gpu_queue()
                    .write_buffer(topk_params_buf, 0, &tparams);

                let num_blocks = chunk_rows.div_ceil(block_size);
                let cand_count = num_blocks;

                let gemm_entries = [
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight_binding.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: input_buf.as_entire_binding(),
                    },
                ];
                let gemm_bind =
                    self.host
                        .gpu_device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Top1GemmBind"),
                            layout: &gemm_layout,
                            entries: &gemm_entries[..if use_coop { 5 } else { 4 }],
                        });
                let mut encoder =
                    self.host
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Top1Encoder"),
                        });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Top1GemmPass"),
                        timestamp_writes: self.host.pass_writes(Profile::Begin),
                    });
                    cpass.set_pipeline(gemm_pipeline);
                    cpass.set_bind_group(0, &gemm_bind, &[]);
                    if use_mr {
                        cpass.dispatch_workgroups(
                            self.host.coop_gemv_workgroups(chunk_rows as u32),
                            1,
                            1,
                        );
                    } else if use_coop {
                        cpass.dispatch_workgroups(chunk_rows as u32, 1, 1);
                    } else {
                        cpass.dispatch_workgroups((chunk_rows as u32 + 63) / 64, 1, 1);
                    }
                }
                {
                    let mut tpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Top1ReducePass"),
                        timestamp_writes: self.host.pass_writes(Profile::End),
                    });
                    tpass.set_pipeline(topk_pipeline);
                    tpass.set_bind_group(0, &topk_bind, &[]);
                    tpass.dispatch_workgroups(num_blocks as u32, 1, 1);
                }
                let cand_bytes = (cand_count * 4) as wgpu::BufferAddress;
                let val_dst = (cand_offset * 4) as wgpu::BufferAddress;
                let idx_dst = ((total_cands + cand_offset) * 4) as wgpu::BufferAddress;
                encoder.copy_buffer_to_buffer(cand_val, 0, staging, val_dst, cand_bytes);
                encoder.copy_buffer_to_buffer(cand_idx, 0, staging, idx_dst, cand_bytes);
                self.host.resolve(&mut encoder);
                self.host.submit(encoder.finish());
                self.host.accumulate(Phase::OutputTopk);
                cand_offset += cand_count;
            }
        }

        let map_bytes = (total_cands * 8) as wgpu::BufferAddress;
        let mapped_ok = self.host.map_read(staging, map_bytes);
        if !mapped_ok {
            let _ = staging.unmap();
            return None;
        }

        let mut best_token_id = 0u32;
        let mut max_logit = f32::NEG_INFINITY;
        {
            let data = staging
                .slice(..map_bytes)
                .get_mapped_range()
                .expect("wgpu buffer map_range failed");
            let val_bytes = total_cands * 4;
            let vals: &[f32] = bytemuck::cast_slice(&data[..val_bytes]);
            let idxs: &[u32] = bytemuck::cast_slice(&data[val_bytes..val_bytes * 2]);
            let mut offset = 0usize;
            for chunk_idx in 0..full_chunks {
                let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
                let chunk_rows = VOCAB_CHUNK_ROWS.min(vocab_size - row_start);
                let cand_count = chunk_rows.div_ceil(block_size);
                for i in 0..cand_count {
                    let pos = offset + i;
                    let v = vals[pos];
                    let token_id = row_start as u32 + idxs[pos];
                    if v > f32::NEG_INFINITY
                        && (v > max_logit || (v == max_logit && token_id < best_token_id))
                    {
                        max_logit = v;
                        best_token_id = token_id;
                    }
                }
                offset += cand_count;
            }
        }
        staging.unmap();

        if max_logit == f32::NEG_INFINITY {
            None
        } else {
            Some(StreamingArgmaxResult {
                best_token_id,
                max_logit,
            })
        }
    }
}
