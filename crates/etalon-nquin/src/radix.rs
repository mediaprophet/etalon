//! Eight-pass Stable GPU LSD Radix Sort CPU oracle and controller.

use bytemuck::{Pod, Zeroable};

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
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::GpuRadixSorter;

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
        for &len in &[0usize, 1, 63, 64, 65, 255, 256, 257, 1021, 1024, 1025] {
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
            (
                "graph_radix_hist.wgsl",
                include_str!("shaders/graph_radix_hist.wgsl"),
            ),
            (
                "graph_radix_prefix.wgsl",
                include_str!("shaders/graph_radix_prefix.wgsl"),
            ),
            (
                "graph_radix_scatter.wgsl",
                include_str!("shaders/graph_radix_scatter.wgsl"),
            ),
        ] {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name} failed WGSL parse: {e}"));
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
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
