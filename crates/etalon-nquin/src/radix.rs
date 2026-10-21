//! Fixed 3-pass Stable GPU LSD Radix Sort CPU oracle and controller.

use bytemuck::{Pod, Zeroable};

use super::layout::NQuinWire;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct RadixParams {
    pub n: u32,
    pub shift: u32,
    pub num_wgs: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct PrefixParams {
    pub total_bins: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadixSortOutcome {
    Sorted { passes: u32 },
    Empty,
}

/// Exclusive prefix sum over a 256-bin histogram (kept for CPU unit test parity).
#[inline]
pub fn cpu_prefix_sum_256(counts: &[u32; 256]) -> [u32; 256] {
    let mut offsets = [0u32; 256];
    let mut sum = 0u32;
    for d in 0..256 {
        offsets[d] = sum;
        sum = sum.saturating_add(counts[d]);
    }
    offsets
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
    pub prefix_pipeline: etalon::ComputePipeline,
    pub scatter_pipeline: etalon::ComputePipeline,
    pub hist_layout: etalon::BindGroupLayout,
    pub prefix_layout: etalon::BindGroupLayout,
    pub scatter_layout: etalon::BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuRadixSorter {
    pub fn new(device: &etalon::Device) -> Self {
        let hist_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-hist-shader"),
            source: etalon::ShaderSource::Wgsl(include_str!("shaders/graph_radix_hist.wgsl").into()),
        });
        let prefix_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-prefix-shader"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("shaders/graph_radix_prefix.wgsl").into(),
            ),
        });
        let scatter_shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-radix-scatter-shader"),
            source: etalon::ShaderSource::Wgsl(
                include_str!("shaders/graph_radix_scatter.wgsl").into(),
            ),
        });

        let hist_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-radix-hist-pipeline"),
            layout: None,
            module: &hist_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let prefix_pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-radix-prefix-pipeline"),
            layout: None,
            module: &prefix_shader,
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

        let hist_layout = hist_pipeline.get_bind_group_layout(0);
        let prefix_layout = prefix_pipeline.get_bind_group_layout(0);
        let scatter_layout = scatter_pipeline.get_bind_group_layout(0);

        Self {
            hist_pipeline,
            prefix_pipeline,
            scatter_pipeline,
            hist_layout,
            prefix_layout,
            scatter_layout,
        }
    }

    /// Sort `quins` by their 64-bit `predicate` field on the GPU and return the sorted row indices.
    pub fn sort_by_predicate(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        quins: &[NQuinWire],
    ) -> Option<Vec<u32>> {
        let keys: Vec<u64> = quins.iter().map(|q| q.predicate).collect();
        self.sort_indices(device, queue, &keys)
    }

    /// Sort `keys` on the GPU in a single command submission (zero intermediate CPU readbacks)
    /// and return a vector of row indices representing the stable sorted order.
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
        let total_bins = 256 * wg;
        let hist_size = (total_bins * 4) as u64;

        let hist_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-hist"),
            size: hist_size,
            usage: usage_st,
            mapped_at_creation: false,
        });
        let off_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-off"),
            size: hist_size,
            usage: usage_st,
            mapped_at_creation: false,
        });

        let prefix_params = PrefixParams {
            total_bins,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let prefix_param_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-prefix-params"),
            size: core::mem::size_of::<PrefixParams>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&prefix_param_buf, 0, bytemuck::bytes_of(&prefix_params));

        let prefix_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-radix-prefix-bg"),
            layout: &self.prefix_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: hist_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: off_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: prefix_param_buf.as_entire_binding(),
                },
            ],
        });

        // Pre-create per-pass uniform buffers for shifts 0, 8, 16, 24, 32, 40, 48, 56
        // so all 8 LSD passes are recorded into a single CommandEncoder.
        let param_bufs: [etalon::Buffer; 8] = core::array::from_fn(|pass| {
            let params = RadixParams {
                n: n as u32,
                shift: (pass * 8) as u32,
                num_wgs: wg,
                _pad1: 0,
            };
            let buf = device.create_buffer(&etalon::BufferDescriptor {
                label: Some("etalon-radix-pass-params"),
                size: core::mem::size_of::<RadixParams>() as u64,
                usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
            buf
        });

        let out_stage = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-radix-out-stage"),
            size: idx_bytes,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut src_k = &k0;
        let mut src_i = &i0;
        let mut dst_k = &k1;
        let mut dst_i = &i1;

        let mut enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("etalon-radix-single-submission-enc"),
        });

        for param_buf in &param_bufs {
            let hist_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
                label: Some("etalon-radix-hist-bg"),
                layout: &self.hist_layout,
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

            let sc_bg = device.create_bind_group(&etalon::BindGroupDescriptor {
                label: Some("etalon-radix-scatter-bg"),
                layout: &self.scatter_layout,
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

            let mut pass = enc.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: Some("etalon-radix-lsd-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hist_pipeline);
            pass.set_bind_group(0, &hist_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);

            pass.set_pipeline(&self.prefix_pipeline);
            pass.set_bind_group(0, &prefix_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &sc_bg, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
            drop(pass);

            std::mem::swap(&mut src_k, &mut dst_k);
            std::mem::swap(&mut src_i, &mut dst_i);
        }

        enc.copy_buffer_to_buffer(src_i, 0, &out_stage, 0, idx_bytes);
        queue.submit(Some(enc.finish()));

        #[cfg(target_arch = "wasm32")]
        {
            let _ = out_stage;
            return None;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let slice = out_stage.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(etalon::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            let _ = device.poll(etalon::PollType::wait_or_timeout(
                core::time::Duration::from_millis(5000),
            ));
            rx.recv().ok()?.ok()?;

            let raw = slice.get_mapped_range().ok()?;
            let result: Vec<u32> = bytemuck::cast_slice(&raw).to_vec();
            drop(raw);
            out_stage.unmap();

            if result.len() != n {
                return None;
            }
            let mut seen = vec![false; n];
            let mut prev = 0u64;
            for (step, &row_idx) in result.iter().enumerate() {
                let row = row_idx as usize;
                if row >= n || seen[row] {
                    return None;
                }
                seen[row] = true;
                let k = keys[row];
                if step > 0 && k < prev {
                    return None;
                }
                prev = k;
            }

            Some(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_prefix_sum_256_exact() {
        let mut counts = [0u32; 256];
        counts[0] = 3;
        counts[1] = 5;
        counts[255] = 7;
        let offsets = cpu_prefix_sum_256(&counts);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 3);
        assert_eq!(offsets[2], 8);
        assert_eq!(offsets[255], 8);
    }

    #[test]
    fn radix_sort_cpu_adversarial_lengths_and_high_bits() {
        for &len in &[0usize, 1, 255, 256, 257, 1021, 1024, 1025] {
            let mut keys: Vec<u64> = (0..len)
                .rev()
                .map(|i| {
                    let hi = if i % 2 == 0 { 1u64 << 63 } else { 0 };
                    hi | ((i / 3) as u64)
                })
                .collect();
            let orig_keys = keys.clone();
            let mut indices: Vec<u32> = (0..len as u32).collect();
            let outcome = radix_sort_u64_cpu(&mut keys, &mut indices);
            if len == 0 {
                assert_eq!(outcome, RadixSortOutcome::Empty);
                continue;
            }
            assert_eq!(outcome, RadixSortOutcome::Sorted { passes: 8 });
            for w in keys.windows(2) {
                assert!(w[0] <= w[1]);
            }
            // Verify stability for duplicate keys
            for pair in indices.windows(2) {
                let a = pair[0] as usize;
                let b = pair[1] as usize;
                if orig_keys[a] == orig_keys[b] {
                    assert!(a < b, "LSD radix sort must be stable on duplicate keys");
                }
            }
        }
    }

    #[test]
    fn radix_wgsl_shaders_parse_and_validate() {
        for (name, src) in [
            ("graph_radix_hist.wgsl", include_str!("shaders/graph_radix_hist.wgsl")),
            ("graph_radix_prefix.wgsl", include_str!("shaders/graph_radix_prefix.wgsl")),
            ("graph_radix_scatter.wgsl", include_str!("shaders/graph_radix_scatter.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name} failed WGSL parse: {e}"));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name} failed Naga validation: {e}"));
        }
    }
}
