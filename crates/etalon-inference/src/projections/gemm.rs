use super::*;
use crate::ordinary::{gpu_params::*, quant_support::ggml_gpu_gemm_supported};
impl<P: ProjectionGpuProvider> ProjectionGpuEngine<'_, P> {
    pub fn dispatch_gemm_raw_into(
        &self,
        info: &GgufTensorInfo,
        raw: &[u8],
        input: &[f32],
        out: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> bool {
        if n_in > input.len() || n_out > out.len() {
            return false;
        }

        if self.host.cuda_gemm(info, raw, input, out, n_in, n_out) {
            return true;
        }

        let weight_bytes = raw.len();
        // GEMM shader supports a wider quant set than the legacy `ggml_gpu_quant_supported` (Q4_K/Q6_K)
        // — notably Q8_0, which was silently falling back to the CPU `stack_gemm_quant` below (the FFN
        // bottleneck for Q8_0 models). The guards (size/buffer caps) still fail closed → CPU fallback.
        if ggml_gpu_gemm_supported(info.ggml_type)
            && n_in <= MAX_STACK_GEMM_IN
            && n_out <= self.gemm_max_out_dim as usize
            && weight_bytes <= self.max_tensor_bytes
            && self.gemm_input_buf.is_some()
        {
            let params = GemmGpuParams {
                n_in: n_in as u32,
                n_out: n_out as u32,
                weight_ggml_type: info.ggml_type,
                weight_row_elems: info.dims[0] as u32,
                weight_byte_len: raw.len() as u32,
                n_batch: 1,
                in_row_stride: 0,
                out_row_stride: 0,
            };
            let input_buf = self.gemm_input_buf.as_ref().unwrap();
            let weight_buf = self.gemm_weight_buf.as_ref().unwrap();
            let output_buf = self.gemm_output_buf.as_ref().unwrap();
            let params_buf = self.gemm_params_buf.as_ref().unwrap();
            let staging = self.gemm_output_staging.as_ref().unwrap();

            self.host
                .gpu_queue()
                .write_buffer(input_buf, 0, bytemuck::cast_slice(&input[..n_in]));
            // Phase 2 (native): bind this tensor's resident VRAM buffer (uploaded once, keyed by
            // byte_offset) instead of re-uploading the weight into the shared gemm_weight_buf every
            // token. On wasm the resident path is the MC8 arena, so this stays the per-token upload.
            #[cfg(not(target_arch = "wasm32"))]
            let resident = if self.host.resident_weights_enabled() {
                // Distinguish chunks by range start (the output chunks share tensor byte_offset).
                // The host retains the immutable source owner, invalidates on replacement, and
                // the canonical cache validates length/device. This address is never a standalone
                // model identity and remains warm across requests using the same lease. Buffers
                // hold original bytes (layout-neutral); transformed f16 blobs use a separate tag.
                self.host.resident_weight_buffer(raw.as_ptr() as u64, raw)
            } else {
                None
            };
            #[cfg(target_arch = "wasm32")]
            let resident: Option<wgpu::Buffer> = None;
            let weight_binding: &wgpu::Buffer = match resident.as_ref() {
                Some(r) => r,
                None => {
                    self.host.write_weight_words(raw, self.max_tensor_bytes);
                    weight_buf
                }
            };
            self.host
                .gpu_queue()
                .write_buffer(params_buf, 0, bytemuck::bytes_of(&params));

            // 0.0.21: select the cooperative GEMV kernel (one workgroup/row, coalesced + shared-mem
            // reduction) when enabled, else the naive 1-thread/row kernel. Same group-0 bindings, so
            // only the pipeline + dispatch geometry differ. The bind group must be built from the
            // ACTIVE pipeline's auto-layout.
            #[cfg(not(target_arch = "wasm32"))]
            let use_coop = self.host.coop_gemv_enabled();
            #[cfg(target_arch = "wasm32")]
            let use_coop = false;
            #[cfg(not(target_arch = "wasm32"))]
            let use_mr = use_coop
                && info.ggml_type == etalon_model::ggml::GGML_TYPE_Q4_K_SOA
                && n_out >= 512;
            #[cfg(not(target_arch = "wasm32"))]
            let active_pipeline: &wgpu::ComputePipeline = if use_mr {
                &self.coop_gemv_mr_pipeline
            } else if use_coop {
                &self.coop_gemv_pipeline
            } else {
                &self.pipeline
            };
            #[cfg(target_arch = "wasm32")]
            let use_mmv_q8_0 =
                info.ggml_type == etalon_model::ggml::GGML_TYPE_Q8_0 && (n_in % 32 == 0);
            #[cfg(target_arch = "wasm32")]
            let active_pipeline: &wgpu::ComputePipeline = if use_mmv_q8_0 {
                &self.mmv_q8_0_pipeline
            } else {
                &self.pipeline
            };
            #[cfg(target_arch = "wasm32")]
            let use_mr = false;

            #[cfg(not(target_arch = "wasm32"))]
            let bind_layout = self.host.native_gemm_bind_layout(use_coop).clone();
            #[cfg(target_arch = "wasm32")]
            let bind_layout = active_pipeline.get_bind_group_layout(0);
            // CoopGemvBGL is 5-slot (binding 4 = residual). Dummy residual = input.
            let bind_group = if use_coop {
                self.host
                    .gpu_device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("LayerGemmBindGroup"),
                        layout: &bind_layout,
                        entries: &[
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
                        ],
                    })
            } else {
                self.host
                    .gpu_device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("LayerGemmBindGroup"),
                        layout: &bind_layout,
                        entries: &[
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
                        ],
                    })
            };

            let mut encoder =
                self.host
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("LayerGemmEncoder"),
                    });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: self.host.pass_writes(Profile::Both),
                });
                cpass.set_pipeline(active_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                if use_mr {
                    cpass.dispatch_workgroups(self.host.coop_gemv_workgroups(n_out as u32), 1, 1);
                } else if use_coop {
                    cpass.dispatch_workgroups(n_out as u32, 1, 1);
                } else {
                    #[cfg(target_arch = "wasm32")]
                    if use_mmv_q8_0 {
                        cpass.dispatch_workgroups((n_out as u32 + 3) / 4, 1, 1);
                    } else {
                        cpass.dispatch_workgroups((n_out as u32 + 63) / 64, 1, 1);
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    cpass.dispatch_workgroups((n_out as u32 + 63) / 64, 1, 1);
                }
            }
            let out_bytes = (n_out * 4) as wgpu::BufferAddress;
            encoder.copy_buffer_to_buffer(output_buf, 0, staging, 0, out_bytes);
            self.host.resolve(&mut encoder);
            self.host.submit(encoder.finish());
            self.host.accumulate(Phase::Gemm);

            if self.host.readback(staging, out_bytes, &mut out[..n_out]) {
                return true;
            }
            let _ = staging.unmap();
        }

        self.host.cpu_gemm(raw, info, input, out, n_in, n_out)
    }
}
