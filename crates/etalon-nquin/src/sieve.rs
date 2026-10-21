//! Equality sieve, Warp-Transposed `uvec4` GPU scan pipeline, and CPU verification oracle.

use bytemuck::{Pod, Zeroable};

use super::layout::NQuinWire;

/// Raw WGSL source for the bank-conflict-free warp-transposed `uvec4` NQuin scan shader.
pub const NQUIN_WARP_SCAN_WGSL: &str = include_str!("shaders/nquin_warp_scan.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum QuinField {
    Subject = 0,
    Predicate = 1,
    Object = 2,
    Context = 3,
}

impl QuinField {
    #[inline(always)]
    pub fn extract(&self, quin: &NQuinWire) -> u64 {
        match self {
            Self::Subject => quin.subject,
            Self::Predicate => quin.predicate,
            Self::Object => quin.object,
            Self::Context => quin.context,
        }
    }

    #[inline(always)]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct SieveParams {
    pub n: u32,
    pub field: u32,
    pub needle_lo: u32,
    pub needle_hi: u32,
}

/// Compact Quins whose `field == needle` into `out`. Returns count written.
pub fn sieve_eq_cpu(
    quins: &[NQuinWire],
    field: QuinField,
    needle: u64,
    out: &mut [NQuinWire],
) -> usize {
    let mut written = 0;
    for q in quins {
        if field.extract(q) == needle {
            if written < out.len() {
                out[written] = *q;
                written += 1;
            } else {
                break;
            }
        }
    }
    written
}

/// GPU warp-transposed `uvec4` equality sieve over 48-byte `NQuinWire` buffers.
#[cfg(feature = "gpu")]
pub struct GpuQuinSieve {
    pub pipeline: etalon::ComputePipeline,
    pub bind_group_layout: etalon::BindGroupLayout,
}

/// Alias matching Runbook 05 naming (`GpuNQuinSieve`).
#[cfg(feature = "gpu")]
pub type GpuNQuinSieve = GpuQuinSieve;

#[cfg(feature = "gpu")]
impl GpuQuinSieve {
    pub fn new(device: &etalon::Device) -> Self {
        let shader = device.create_shader_module(etalon::ShaderModuleDescriptor {
            label: Some("etalon-nquin-warp-scan-shader"),
            source: etalon::ShaderSource::Wgsl(NQUIN_WARP_SCAN_WGSL.into()),
        });

        let pipeline = device.create_compute_pipeline(&etalon::ComputePipelineDescriptor {
            label: Some("etalon-nquin-warp-scan-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Record a warp-transposed `uvec4` equality scan dispatch into `cpass`.
    pub fn record_dispatch<'a>(
        &'a self,
        cpass: &mut etalon::ComputePass<'a>,
        bind_group: &'a etalon::BindGroup,
        quin_count: u32,
    ) {
        if quin_count == 0 {
            return;
        }
        let workgroups = quin_count.div_ceil(32);
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Execute the GPU warp-transposed equality scan over `quins` and return sorted matching row indices.
    pub fn sieve_eq_indices(
        &self,
        device: &etalon::Device,
        queue: &etalon::Queue,
        quins: &[NQuinWire],
        field: QuinField,
        needle: u64,
    ) -> Option<Vec<u32>> {
        let n = quins.len();
        if n == 0 {
            return Some(Vec::new());
        }

        // Pad input buffer up to a multiple of 32 Quins (1536 bytes) so the 3-wave vec4<u32>
        // tile loader has full 16-byte aligned backing storage.
        let padded_quins = n.div_ceil(32) * 32;
        let quin_bytes = (padded_quins * 48) as u64;
        let match_bytes = (n * 4) as u64;

        let quin_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-quins"),
            size: quin_bytes,
            usage: etalon::BufferUsages::STORAGE | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&quin_buf, 0, bytemuck::cast_slice(quins));

        let matches_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-matches"),
            size: match_bytes,
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_SRC
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-count"),
            size: 4,
            usage: etalon::BufferUsages::STORAGE
                | etalon::BufferUsages::COPY_SRC
                | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&count_buf, 0, &[0u8; 4]);

        let params = SieveParams {
            n: n as u32,
            field: field.as_u32(),
            needle_lo: needle as u32,
            needle_hi: (needle >> 32) as u32,
        };
        let param_buf = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-params"),
            size: core::mem::size_of::<SieveParams>() as u64,
            usage: etalon::BufferUsages::UNIFORM | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));

        let bg = device.create_bind_group(&etalon::BindGroupDescriptor {
            label: Some("etalon-nquin-sieve-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                etalon::BindGroupEntry {
                    binding: 0,
                    resource: quin_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 1,
                    resource: matches_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 2,
                    resource: count_buf.as_entire_binding(),
                },
                etalon::BindGroupEntry {
                    binding: 3,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let count_stage = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-count-stage"),
            size: 4,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let matches_stage = device.create_buffer(&etalon::BufferDescriptor {
            label: Some("etalon-nquin-sieve-matches-stage"),
            size: match_bytes,
            usage: etalon::BufferUsages::MAP_READ | etalon::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&etalon::CommandEncoderDescriptor {
            label: Some("etalon-nquin-sieve-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&etalon::ComputePassDescriptor {
                label: Some("etalon-nquin-sieve-pass"),
                timestamp_writes: None,
            });
            self.record_dispatch(&mut pass, &bg, n as u32);
        }
        enc.copy_buffer_to_buffer(&count_buf, 0, &count_stage, 0, 4);
        enc.copy_buffer_to_buffer(&matches_buf, 0, &matches_stage, 0, match_bytes);
        queue.submit(Some(enc.finish()));

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (count_stage, matches_stage);
            return None;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let count_slice = count_stage.slice(..);
            let (tx_c, rx_c) = std::sync::mpsc::channel();
            count_slice.map_async(etalon::MapMode::Read, move |res| {
                let _ = tx_c.send(res);
            });
            let matches_slice = matches_stage.slice(..);
            let (tx_m, rx_m) = std::sync::mpsc::channel();
            matches_slice.map_async(etalon::MapMode::Read, move |res| {
                let _ = tx_m.send(res);
            });
            let _ = device.poll(etalon::PollType::wait_or_timeout(
                core::time::Duration::from_millis(5000),
            ));
            rx_c.recv().ok()?.ok()?;
            rx_m.recv().ok()?.ok()?;

            let count_raw = count_slice.get_mapped_range().ok()?;
            let match_count = u32::from_le_bytes([
                count_raw[0],
                count_raw[1],
                count_raw[2],
                count_raw[3],
            ]) as usize;
            drop(count_raw);
            count_stage.unmap();

            if match_count > n {
                matches_stage.unmap();
                return None;
            }

            let matches_raw = matches_slice.get_mapped_range().ok()?;
            let all_indices: &[u32] = bytemuck::cast_slice(&matches_raw);
            let mut out = all_indices[..match_count].to_vec();
            drop(matches_raw);
            matches_stage.unmap();

            out.sort_unstable();
            Some(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nquin_warp_scan_wgsl_parses_and_validates_in_naga() {
        let module = naga::front::wgsl::parse_str(NQUIN_WARP_SCAN_WGSL)
            .expect("nquin_warp_scan.wgsl must parse cleanly");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("nquin_warp_scan.wgsl must validate cleanly through naga");
    }

    #[test]
    fn sieve_eq_cpu_filters_exact_field() {
        let q0 = NQuinWire::new(10, 0x100, 20, 1, 0, 10 ^ 0x100 ^ 20 ^ 1);
        let q1 = NQuinWire::new(11, 0x200, 21, 1, 0, 11 ^ 0x200 ^ 21 ^ 1);
        let q2 = NQuinWire::new(12, 0x100, 22, 2, 0, 12 ^ 0x100 ^ 22 ^ 2);
        let quins = [q0, q1, q2];
        let mut out = [NQuinWire::new(0, 0, 0, 0, 0, 0); 3];
        let count = sieve_eq_cpu(&quins, QuinField::Predicate, 0x100, &mut out);
        assert_eq!(count, 2);
        assert_eq!(out[0], q0);
        assert_eq!(out[1], q2);
    }
}
