//! Fixed 3-pass Stable GPU LSD Radix Sort CPU oracle and controller.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct RadixParams {
    pub n: u32,
    pub shift: u32,
    pub num_wgs: u32,
    pub _pad1: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadixSortOutcome {
    Sorted { passes: u32 },
    Empty,
}

/// CPU reference oracle for 64-bit LSD radix sort.
pub fn radix_sort_u64_cpu(keys: &mut [u64], indices: &mut [u32]) -> RadixSortOutcome {
    if keys.is_empty() {
        return RadixSortOutcome::Empty;
    }
    let n = keys.len();
    assert_eq!(n, indices.len());

    let mut temp_keys = vec![0u64; n];
    let mut temp_indices = vec![0u32; n];

    for pass in 0..8 {
        let shift = pass * 8;
        let mut counts = [0usize; 256];

        for &k in keys.iter() {
            let digit = ((k >> shift) & 0xFF) as usize;
            counts[digit] += 1;
        }

        let mut offsets = [0usize; 256];
        let mut sum = 0usize;
        for d in 0..256 {
            offsets[d] = sum;
            sum += counts[d];
        }

        for i in 0..n {
            let k = keys[i];
            let idx = indices[i];
            let digit = ((k >> shift) & 0xFF) as usize;
            let pos = offsets[digit];
            offsets[digit] += 1;
            temp_keys[pos] = k;
            temp_indices[pos] = idx;
        }

        keys.copy_from_slice(&temp_keys);
        indices.copy_from_slice(&temp_indices);
    }

    RadixSortOutcome::Sorted { passes: 8 }
}

#[cfg(feature = "gpu")]
pub struct GpuRadixSorter {
    pub hist_pipeline: etalon::ComputePipeline,
    pub scatter_pipeline: etalon::ComputePipeline,
}

#[cfg(feature = "gpu")]
impl GpuRadixSorter {
    pub fn new(device: &etalon::Device) -> Self {
        let hist_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-hist-shader"),
            source: etalon::ShaderSource::Wgsl(include_str!("shaders/graph_radix_hist.wgsl").into()),
        });
        let scatter_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-scatter-shader"),
            source: etalon::ShaderSource::Wgsl(include_str!("shaders/graph_radix_scatter.wgsl").into()),
        });

        let hist_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-radix-hist-pipeline"),
            layout: None,
            module: &hist_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-radix-scatter-pipeline"),
            layout: None,
            module: &scatter_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            hist_pipeline,
            scatter_pipeline,
        }
    }

    /// Sort `keys` on the GPU and return a vector of row indices representing the sorted order.
    pub fn sort_indices(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        keys: &[u64],
    ) -> Option<Vec<u32>> {
        let n = keys.len();
        if n == 0 {
            return Some(Vec::new());
        }

        let mut packed = vec![0u32; n * 2];
        for (i, &x) in keys.iter().enumerate() {
            packed[i * 2] = x as u32;
            packed[i * 2 + 1] = (x >> 32) as u32;
        }
        let idx: Vec<u32> = (0..n as u32).collect();
        let key_bytes = (n * 8) as u64;
        let idx_bytes = (n * 4) as u64;

        let usage_st = etalon::BufferUsages::STORAGE
            | etalon::BufferUsages::COPY_SRC
            | etalon::BufferUsages::COPY_DST;

        let k0 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-k0"),
            size: key_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let k1 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-k1"),
            size: key_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let i0 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-i0"),
            size: idx_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let i1 = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-i1"),
            size: idx_bytes,
            usage: usage_st,
            mapped_at_creation: false,
        });

        queue.write_buffer(&k0, 0, bytemuck::cast_slice(&packed));
        queue.write_buffer(&i0, 0, bytemuck::cast_slice(&idx));

        let wg = (n as u32).div_ceil(256).max(1);
        let hist_size = (256 * wg * 4) as u64;

        let hist_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-hist"),
            size: hist_size,
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_SRC
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hist_stage = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-hist-stage"),
            size: hist_size,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let off_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-off"),
            size: hist_size,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let param_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-params"),
            size: core::mem::size_of::<RadixParams>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut src_k = &k0;
        let mut src_i = &i0;
        let mut dst_k = &k1;
        let mut dst_i = &i1;

        for shift in (0..64).step_by(8) {
            let params = RadixParams {
                n: n as u32,
                shift: shift as u32,
                num_wgs: wg,
                _pad1: 0,
            };
            queue.write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));

            let hist_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
                label: Some("etalon-radix-hist-bg"),
                layout: &self.hist_pipeline.get_bind_group_layout(0),
                entries: &[
                    etalon::BindGroupEntry {
                        binding: 0,
                        resource: src_k.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 1,
                        resource: hist_buf.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 2,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let mut enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
                label: Some("etalon-radix-hist-enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&etalon::ComputePassDescriptor {
                    label: Some("etalon-radix-hist"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.hist_pipeline);
                pass.set_bind_group(0, &hist_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            enc.copy_buffer_to_buffer(&hist_buf, 0, &hist_stage, 0, hist_size);
            queue.submit(Some(enc.finish()));

            // Synchronous map-read for device prefix sum
            let slice = hist_stage.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(etalon::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            let _ = device.poll(etalon::PollType::wait_or_timeout(core::time::Duration::from_millis(5000)));
            rx.recv().ok()?.ok()?;

            let raw = slice.get_mapped_range().ok()?;
            let hist: &[u32] = bytemuck::cast_slice(&raw);
            let mut offsets = vec![0u32; 256 * wg as usize];
            let mut sum = 0u32;
            for d in 0..256 {
                for w in 0..wg as usize {
                    offsets[d * wg as usize + w] = sum;
                    sum = sum.saturating_add(hist[d * wg as usize + w]);
                }
            }
            drop(raw);
            hist_stage.unmap();

            if sum as usize != n {
                return None;
            }
            queue.write_buffer(&off_buf, 0, bytemuck::cast_slice(&offsets));

            let sc_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
                label: Some("etalon-radix-scatter-bg"),
                layout: &self.scatter_pipeline.get_bind_group_layout(0),
                entries: &[
                    etalon::BindGroupEntry {
                        binding: 0,
                        resource: src_k.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 1,
                        resource: src_i.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 2,
                        resource: dst_k.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 3,
                        resource: dst_i.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 4,
                        resource: off_buf.as_entire_binding(),
                    },
                    etalon::BindGroupEntry {
                        binding: 5,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let mut enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
                label: Some("etalon-radix-scatter-enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&etalon::ComputePassDescriptor {
                    label: Some("etalon-radix-scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, &sc_bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            queue.submit(Some(enc.finish()));

            std::mem::swap(&mut src_k, &mut dst_k);
            std::mem::swap(&mut src_i, &mut dst_i);
        }

        // Read back final indices
        let out_stage = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-out-stage"),
            size: idx_bytes,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("etalon-radix-out-enc"),
        });
        enc.copy_buffer_to_buffer(src_i, 0, &out_stage, 0, idx_bytes);
        queue.submit(Some(enc.finish()));

        let slice = out_stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(etalon::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = device.poll(etalon::PollType::wait_or_timeout(core::time::Duration::from_millis(5000)));
        rx.recv().ok()?.ok()?;

        let raw = slice.get_mapped_range().ok()?;
        let result: Vec<u32> = bytemuck::cast_slice(&raw).to_vec();
        drop(raw);
        out_stage.unmap();

        Some(result)
    }
}
