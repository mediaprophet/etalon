use super::*;
use crate::ordinary::{gpu_params::*, quant_support::ggml_gpu_gemm_supported};
impl<P: ProjectionGpuProvider> ProjectionGpuEngine<'_, P> {
    pub fn dispatch_output_topk_chunked_into(
        &self,
        index: &P::Index,
        hidden: &[f32],
        emb_dim: usize,
        k: usize,
        out: &mut [super::topk::TopKItem],
    ) -> Option<usize> {
        let descriptor = P::logits(index)?;
        let info = &descriptor;
        let (n_in, vocab_size) = crate::ordinary::matmul_dims(info);
        if n_in == 0 || vocab_size == 0 || n_in > emb_dim || n_in > hidden.len() {
            return None;
        }
        let pipeline = self.output_topk_pipeline.as_ref()?;
        let topk_layout = self.output_topk_bind_layout.as_ref()?;
        let input_buf = self.gemm_input_buf.as_ref()?;
        let weight_buf = self.gemm_weight_buf.as_ref()?;
        let output_buf = self.gemm_output_buf.as_ref()?;
        let params_buf = self.gemm_params_buf.as_ref()?;
        let topk_params_buf = self.topk_params_buf.as_ref()?;
        let cand_val = self.topk_cand_val_buf.as_ref()?;
        let cand_idx = self.topk_cand_idx_buf.as_ref()?;
        let staging = self.topk_cand_staging.as_ref()?;
        let mmap = self.host.model_bytes()?;

        let k = k.clamp(1, super::topk::TOPK_MAX_K);
        if out.len() < k {
            return None;
        }
        let block_size = super::topk::TOPK_BLOCK_SIZE;
        let full_chunks = vocab_size.div_ceil(VOCAB_CHUNK_ROWS);
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
        let topk_bind = self
            .host
            .gpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TopkBind"),
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

        let mut selected = 0usize;

        // A1a step-2: when the output projection is resident (uploaded once at init), bind the
        // per-chunk sub-range - zero per-token upload. `VOCAB_CHUNK_ROWS` is a multiple of 256, so
        // every chunk offset is storage-binding aligned. The bound bytes, quant, shader and params
        // are identical to the per-chunk-upload fallback, so logits are byte-for-byte equal.
        let resident_logits = self.mc8_logits_resident_buf.as_ref();
        let resident_row_bytes = self.mc8_logits_row_bytes as u64;

        for chunk_idx in 0..full_chunks {
            let row_start = chunk_idx * VOCAB_CHUNK_ROWS;
            let chunk_rows = VOCAB_CHUNK_ROWS.min(vocab_size - row_start);

            let (weight_resource, weight_byte_len) = if let Some(buf) = resident_logits {
                // Only the GPU-quant fast path is supported; otherwise signal fallback to argmax.
                if !(ggml_gpu_gemm_supported(info.ggml_type)
                    && n_in <= MAX_STACK_GEMM_IN
                    && chunk_rows <= self.gemm_max_out_dim as usize)
                {
                    return None;
                }
                let byte_len = chunk_rows as u64 * resident_row_bytes;
                let res = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: row_start as u64 * resident_row_bytes,
                    size: std::num::NonZeroU64::new(byte_len),
                });
                (res, byte_len as u32)
            } else {
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
                self.host.write_weight_words(raw, self.max_tensor_bytes);
                (weight_buf.as_entire_binding(), byte_len)
            };

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
            self.host
                .gpu_queue()
                .write_buffer(input_buf, 0, bytemuck::cast_slice(&hidden[..n_in]));
            self.host
                .gpu_queue()
                .write_buffer(params_buf, 0, bytemuck::bytes_of(&gparams));
            let tparams =
                super::topk::topk_params_bytes(chunk_rows as u32, k as u32, block_size as u32);
            self.host
                .gpu_queue()
                .write_buffer(topk_params_buf, 0, &tparams);

            let num_blocks = chunk_rows.div_ceil(block_size);
            let cand_count = num_blocks * k;

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
            let gemm_bind = self
                .host
                .gpu_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("TopkGemmBind"),
                    layout: &gemm_layout,
                    entries: &gemm_entries[..if use_coop { 5 } else { 4 }],
                });
            let mut encoder =
                self.host
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("TopkEncoder"),
                    });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("TopkGemmPass"),
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
                    label: Some("TopkReducePass"),
                    timestamp_writes: self.host.pass_writes(Profile::End),
                });
                tpass.set_pipeline(pipeline);
                tpass.set_bind_group(0, &topk_bind, &[]);
                tpass.dispatch_workgroups(num_blocks as u32, 1, 1);
            }
            let cand_bytes = (cand_count * 4) as wgpu::BufferAddress;
            encoder.copy_buffer_to_buffer(cand_val, 0, staging, 0, cand_bytes);
            encoder.copy_buffer_to_buffer(cand_idx, 0, staging, cand_bytes, cand_bytes);
            self.host.resolve(&mut encoder);
            self.host.submit(encoder.finish());
            self.host.accumulate(Phase::OutputTopk);

            let map_bytes = cand_bytes * 2;
            let mapped_ok = self.host.map_read(staging, map_bytes);
            if !mapped_ok {
                let _ = staging.unmap();
                return None;
            }
            {
                let data = staging
                    .slice(..map_bytes)
                    .get_mapped_range()
                    .expect("wgpu buffer map_range failed");
                let vals: &[f32] = bytemuck::cast_slice(&data[..cand_count * 4]);
                let idxs: &[u32] = bytemuck::cast_slice(&data[cand_count * 4..cand_count * 8]);
                for i in 0..cand_count {
                    let v = vals[i];
                    if v > f32::NEG_INFINITY {
                        selected = super::topk::insert_candidate(
                            out,
                            selected,
                            k,
                            row_start as u32 + idxs[i],
                            v,
                        );
                    }
                }
            }
            staging.unmap();
        }

        if selected == 0 {
            None
        } else {
            Some(selected)
        }
    }
}
