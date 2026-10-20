//! Fixed 3-pass Stable GPU LSD Radix Sort CPU oracle and controller.

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
