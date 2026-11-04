use super::encoding::*;
use super::*;
use etalon_model::inference::TensorInfo as GgufTensorInfo;
impl<P: PrefillProvider> PrefillEngine<'_, P> {
    pub fn encode_gemm_bufs_offset(
        &self,
        pipeline: &mut P::Pipeline,
        info: &GgufTensorInfo,
        raw: &[u8],
        n_in: usize,
        n_out: usize,
        input: &wgpu::Buffer,
        in_off: wgpu::BufferAddress,
        in_bytes: wgpu::BufferAddress,
        output: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        n_batch: u32,
        in_row_stride: u32,
        out_row_stride: u32,
        gemm_dyn_offset: u32,
        layer: u32,
        weight_role: Mc8WeightRole,
    ) -> bool {
        let r = self.host.encoding_resources();
        if !P::shader_supported(info.ggml_type)
            || n_in > r.gemm_max_input_floats as usize
            || n_out > r.gemm_max_out_dim as usize
            || raw.len() > r.max_tensor_bytes
        {
            return false;
        }
        let batch = n_batch.max(1);
        if !r.mc8_weights_resident {
            self.write_weight_role(weight_role, raw, r.max_tensor_bytes);
        }
        let params_buf = r.gemm_params_buf.as_ref().unwrap();
        let key = bg_hash(&[
            1,
            layer as u64,
            weight_role.idx() as u64,
            in_off,
            in_bytes,
            out_off,
            out_bytes,
            r.mc8_weights_resident as u64,
            input as *const _ as u64,
            output as *const _ as u64,
            in_row_stride as u64,
            out_row_stride as u64,
        ]);
        let bind = {
            let cached = r
                .mc8_bg_cache
                .lock()
                .ok()
                .and_then(|c| c.get(&key).cloned());
            match cached {
                Some(bg) => bg,
                None => {
                    let bg = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("MC8GemmBindOff"),
                        layout: &r.mc8_gemm_bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf_slice(input, in_off, in_bytes),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.weight_binding(weight_role, layer),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: dynamic_uniform_binding(params_buf),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: buf_slice(output, out_off, out_bytes),
                            },
                        ],
                    });
                    if let Ok(mut c) = r.mc8_bg_cache.lock() {
                        c.insert(key, bg.clone());
                    }
                    bg
                }
            }
        };
        let mut cpass = P::encoder(pipeline).begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        let use_mmv =
            info.ggml_type == etalon_model::ggml::GGML_TYPE_Q8_0 && batch == 1 && (n_in % 32 == 0);
        if use_mmv {
            cpass.set_pipeline(&r.mmv_q8_0_pipeline);
        } else {
            cpass.set_pipeline(&r.pipeline);
        }
        cpass.set_bind_group(0, &bind, &[gemm_dyn_offset]);
        if use_mmv {
            cpass.dispatch_workgroups((n_out as u32 + 3) / 4, 1, 1);
        } else {
            cpass.dispatch_workgroups((n_out as u32 + 63) / 64, batch, 1);
        }
        true
    }
    pub fn encode_gemm_bufs(
        &self,
        pipeline: &mut P::Pipeline,
        info: &GgufTensorInfo,
        raw: &[u8],
        n_in: usize,
        n_out: usize,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> bool {
        let r = self.host.encoding_resources();
        if !P::shader_supported(info.ggml_type)
            || n_in > r.gemm_max_input_floats as usize
            || n_out > r.gemm_max_out_dim as usize
            || raw.len() > r.max_tensor_bytes
        {
            return false;
        }
        let params = super::params::mc8_gemm_params(info, raw.len(), n_in, n_out, 1, 0, 0);
        let mut arena = Mc8UniformArena {
            bytes: [0u8; MC8_MAX_GEMM_UNIFORM_SLOTS * MC8_UNIFORM_ALIGN],
            slots: 0,
        };
        let gemm_dyn_offset = arena.push(&params);
        let params_buf = r.gemm_params_buf.as_ref().unwrap();
        arena.upload(self.host.gpu_queue(), params_buf);
        self.host.write_weight_words(raw, r.max_tensor_bytes);
        let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC8GemmBind"),
            layout: &r.mc8_gemm_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r.gemm_weight_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dynamic_uniform_binding(params_buf),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut cpass = P::encoder(pipeline).begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        let use_mmv = info.ggml_type == etalon_model::ggml::GGML_TYPE_Q8_0 && (n_in % 32 == 0);
        if use_mmv {
            cpass.set_pipeline(&r.mmv_q8_0_pipeline);
        } else {
            cpass.set_pipeline(&r.pipeline);
        }
        cpass.set_bind_group(0, &bind, &[gemm_dyn_offset]);
        if use_mmv {
            cpass.dispatch_workgroups((n_out as u32 + 3) / 4, 1, 1);
        } else {
            cpass.dispatch_workgroups((n_out as u32 + 63) / 64, 1, 1);
        }
        true
    }
}
