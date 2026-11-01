use super::*;

#[test]
fn sizes_use_64_lanes_and_all_256_bins() {
    let limits = etalon::Limits::default();
    for n in [1usize, 63, 64, 65, 255, 256, 257] {
        let (keys, indices, groups, bins, hist) = sort_sizes(n, &limits).unwrap();
        assert_eq!(groups, n.div_ceil(64) as u32);
        assert_eq!(bins, groups * 256);
        assert_eq!(hist, u64::from(bins) * 4);
        assert_eq!((keys, indices), (n as u64 * 8, n as u64 * 4));
    }
}

#[test]
fn rejects_device_limits_before_allocation() {
    let mut limits = etalon::Limits::default();
    limits.max_compute_workgroups_per_dimension = 1;
    assert_eq!(
        sort_sizes(65, &limits),
        Err(GpuQueryError::UnsupportedLimits)
    );
    limits = etalon::Limits::default();
    limits.max_storage_buffer_binding_size = 512;
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
    let raw: Vec<u8> = [1u32, 0, 2, 3]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    assert_eq!(copy_sorted_rows(&raw, &keys, &mut out), Ok(4));
    assert_eq!(out, [1, 0, 2, 3]);
    for rows in [[1u32, 2, 0, 3], [1, 0, 0, 3], [0, 1, 2, 3], [1, 0, 2, 4]] {
        let raw: Vec<u8> = rows.into_iter().flat_map(u32::to_le_bytes).collect();
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
