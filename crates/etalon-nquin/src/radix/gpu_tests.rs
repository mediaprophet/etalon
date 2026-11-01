use super::*;

#[test]
fn sizes_use_64_lanes_and_all_256_bins() {
    let limits = etalon::Limits::default();
    for n in [1usize, 63, 64, 65, 255, 256, 257] {
        let (entries, groups, bins, hist) = sort_sizes(n, &limits).unwrap();
        assert_eq!(groups, n.div_ceil(64) as u32);
        assert_eq!(bins, groups * 256);
        assert_eq!(hist, u64::from(bins) * 4);
        assert_eq!(entries, n as u64 * 12);
    }
}

#[test]
fn rejects_device_limits_before_allocation() {
    let limits = etalon::Limits {
        max_compute_workgroups_per_dimension: 1,
        ..Default::default()
    };
    assert_eq!(
        sort_sizes(65, &limits),
        Err(GpuQueryError::UnsupportedLimits)
    );
    let limits = etalon::Limits {
        max_storage_buffer_binding_size: 512,
        ..Default::default()
    };
    assert_eq!(
        sort_sizes(1, &limits),
        Err(GpuQueryError::UnsupportedLimits)
    );
    if let Ok(too_many) = usize::try_from(u64::from(u32::MAX) + 1) {
        assert_eq!(
            sort_sizes(too_many, &limits),
            Err(GpuQueryError::InputTooLarge)
        );
    }
}

#[test]
fn sorted_readback_rejects_duplicates_disorder_and_unstable_ties() {
    let keys = [9, 1, 9, u64::MAX];
    let mut out = [u32::MAX; 4];
    let raw = packed_bytes(&keys, &[1, 0, 2, 3]);
    assert_eq!(copy_sorted_rows(&raw, &keys, &mut out), Ok(4));
    assert_eq!(out, [1, 0, 2, 3]);
    for rows in [[1u32, 2, 0, 3], [1, 0, 0, 3], [0, 1, 2, 3], [1, 0, 2, 4]] {
        let raw = packed_bytes(&keys, &rows);
        assert_eq!(
            copy_sorted_rows(&raw, &keys, &mut out),
            Err(GpuQueryError::InvalidOutput)
        );
    }
    assert_eq!(
        copy_sorted_rows(&raw[..12], &keys, &mut out),
        Err(GpuQueryError::InvalidOutput)
    );
    assert!(matches!(
        copy_sorted_rows(&raw, &keys, &mut out[..3]),
        Err(GpuQueryError::OutputTooSmall { .. })
    ));
}

fn packed_bytes(keys: &[u64], rows: &[u32]) -> Vec<u8> {
    rows.iter()
        .flat_map(|&row| {
            let entry = RadixEntry::new(keys.get(row as usize).copied().unwrap_or(0), row);
            [entry.key_lo, entry.key_hi, entry.index]
                .into_iter()
                .flat_map(u32::to_le_bytes)
        })
        .collect()
}

#[test]
fn packed_readback_preserves_all_key_bits_and_rejects_corruption() {
    for bit in 0..64 {
        let keys = [0, 1u64 << bit, u64::MAX];
        let mut raw = packed_bytes(&keys, &[0, 1, 2]);
        let mut out = [u32::MAX; 4];
        assert_eq!(copy_sorted_rows(&raw, &keys, &mut out), Ok(3));
        assert_eq!(out, [0, 1, 2, u32::MAX]);
        raw[12] ^= 1;
        assert_eq!(
            copy_sorted_rows(&raw, &keys, &mut out),
            Err(GpuQueryError::InvalidOutput)
        );
    }
}

#[test]
fn packed_sort_fits_four_storage_workload_limits() {
    let limits = etalon::Limits {
        max_storage_buffers_per_shader_stage: 4,
        max_uniform_buffers_per_shader_stage: 1,
        max_bindings_per_bind_group: 4,
        max_buffers_and_acceleration_structures_per_shader_stage: 4,
        max_buffer_size: 32768,
        max_storage_buffer_binding_size: 32768,
        ..Default::default()
    };
    assert_eq!(sort_sizes(1025, &limits), Ok((12300, 17, 4352, 17408)));
    let limits = etalon::Limits {
        max_storage_buffers_per_shader_stage: 2,
        ..limits
    };
    assert_eq!(
        sort_sizes(1025, &limits),
        Err(GpuQueryError::UnsupportedLimits)
    );
    let too_many = (u32::MAX / 3 + 1) as usize;
    assert_eq!(
        sort_sizes(too_many, &etalon::Limits::default()),
        Err(GpuQueryError::InputTooLarge)
    );
}
