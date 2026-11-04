use super::encoding::*;
use super::*;
use etalon_model::inference::{ModelHyperparams, TensorInfo as GgufTensorInfo};
impl<P: PrefillProvider> PrefillEngine<'_, P> {
    pub fn encode_attention_pass_gpu(
        &self,
        pipeline: &mut P::Pipeline,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n_embd: usize,
        num_tokens_in_batch: u32,
        batch_start_token_idx: u32,
        layout: &P::Layout,
        layer: u32,
        token_idx: u32,
        h: &ModelHyperparams,
        info: &GgufTensorInfo,
        raw_weights: &[u8],
        proj_kind: u32,
        n_workgroups: u32,
        attn_dyn_offset: u32,
        weight_role: Mc8WeightRole,
    ) -> bool {
        let r = self.host.encoding_resources();
        debug_assert_eq!(n_embd, h.n_embd as usize);
        debug_assert!(
            batch_start_token_idx < P::max_context(layout) || P::max_context(layout) == 0
        );
        if !P::shader_supported(info.ggml_type)
            || r.kv_cache_gpu.is_none()
            || r.attention_params_buf.is_none()
        {
            return false;
        }
        let (mask_words, mask_active, mask_word_count) =
            P::attention_kv_mask_for_dispatch(layout, token_idx, proj_kind);
        debug_assert!(mask_word_count as usize <= mask_words.len());
        if !r.mc8_weights_resident {
            self.write_weight_role(weight_role, raw_weights, r.max_tensor_bytes);
        }
        if mask_active != 0 {
            self.host.gpu_queue().write_buffer(
                r.attention_mask_buf.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&mask_words),
            );
        }
        let layer_f32s = P::layer_stride(layout) as usize;
        let layer_bytes = (layer_f32s * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let layer_offset =
            (layer as usize * layer_f32s * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let kv_binding = wgpu::BufferBinding {
            buffer: r.kv_cache_gpu.as_ref().unwrap(),
            offset: layer_offset,
            size: std::num::NonZeroU64::new(layer_bytes.max(4)),
        };
        let params_buf = r.attention_params_buf.as_ref().unwrap();
        let key = bg_hash(&[
            5,
            layer as u64,
            proj_kind as u64,
            layer_offset,
            layer_bytes,
            r.mc8_weights_resident as u64,
            weight_role.idx() as u64,
            input as *const _ as u64,
            output as *const _ as u64,
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
                        label: Some("MC8AttnBind"),
                        layout: &r.mc8_attn_bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input.as_entire_binding(),
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
                                resource: wgpu::BindingResource::Buffer(kv_binding),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: output.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: r
                                    .attention_mask_buf
                                    .as_ref()
                                    .unwrap()
                                    .as_entire_binding(),
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
        let (wg_x, wg_y) = if proj_kind == 0 && num_tokens_in_batch > 1 {
            (h.n_head, num_tokens_in_batch)
        } else {
            (n_workgroups.max(1), 1)
        };
        let mut cpass = P::encoder(pipeline).begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&r.attention_pipeline);
        cpass.set_bind_group(0, &bind, &[attn_dyn_offset]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
        true
    }
    pub fn encode_attention_batched_q_prefill(
        &self,
        pipeline: &mut P::Pipeline,
        input: &wgpu::Buffer,
        in_off: wgpu::BufferAddress,
        in_bytes: wgpu::BufferAddress,
        output: &wgpu::Buffer,
        out_off: wgpu::BufferAddress,
        out_bytes: wgpu::BufferAddress,
        n_embd: usize,
        n_tokens: u32,
        layout: &P::Layout,
        layer: u32,
        h: &ModelHyperparams,
        raw_weights: &[u8],
        attn_dyn_offset: u32,
    ) -> bool {
        let r = self.host.encoding_resources();
        debug_assert_eq!(n_embd, h.n_embd as usize);
        if r.kv_cache_gpu.is_none()
            || r.attention_params_buf.is_none()
            || r.attention_mask_buf.is_none()
            || n_tokens == 0
        {
            return false;
        }
        if !r.mc8_weights_resident {
            self.write_weight_role(Mc8WeightRole::AttnQ, raw_weights, r.max_tensor_bytes);
        }
        let params_buf = r.attention_params_buf.as_ref().unwrap();
        let layer_f32s = P::layer_stride(layout) as usize;
        let layer_bytes = (layer_f32s * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let layer_offset =
            (layer as usize * layer_f32s * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let kv_binding = wgpu::BufferBinding {
            buffer: r.kv_cache_gpu.as_ref().unwrap(),
            offset: layer_offset,
            size: std::num::NonZeroU64::new(layer_bytes.max(4)),
        };
        let bind = {
            let key = bg_hash(&[
                6,
                layer as u64,
                layer_offset,
                layer_bytes,
                in_off,
                in_bytes,
                out_off,
                out_bytes,
                input as *const _ as u64,
                output as *const _ as u64,
            ]);
            let cached = r
                .mc8_bg_cache
                .lock()
                .ok()
                .and_then(|c| c.get(&key).cloned());
            match cached {
                Some(bg) => bg,
                None => {
                    let bg = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("MC8AttnBatchedQ"),
                        layout: &r.mc8_attn_bind_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: buf_slice(input, in_off, in_bytes),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.weight_binding(Mc8WeightRole::AttnQ, layer),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: dynamic_uniform_binding(params_buf),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Buffer(kv_binding),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: buf_slice(output, out_off, out_bytes),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: r
                                    .attention_mask_buf
                                    .as_ref()
                                    .unwrap()
                                    .as_entire_binding(),
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
        cpass.set_pipeline(&r.attention_pipeline);
        cpass.set_bind_group(0, &bind, &[attn_dyn_offset]);
        cpass.dispatch_workgroups(h.n_head, n_tokens, 1);
        true
    }
}
