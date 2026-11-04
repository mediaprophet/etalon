use super::{bloom_vram_bytes, hdr_composite_available, BloomUniformBlock, CompositeParamsGpu};

#[test]
fn bloom_output_uniform_includes_white_balance_gains_with_wgsl_alignment() {
    assert_eq!(std::mem::size_of::<CompositeParamsGpu>(), 32);
    assert_eq!(std::mem::size_of::<BloomUniformBlock>(), 48);
}

#[test]
fn exposure_capability_requires_the_active_composite_path() {
    assert!(hdr_composite_available(true, true, true, true));
    assert!(!hdr_composite_available(false, true, true, true));
    assert!(!hdr_composite_available(true, false, true, true));
    assert!(!hdr_composite_available(true, true, false, true));
    assert!(!hdr_composite_available(true, true, true, false));
}

#[test]
fn bloom_budget_counts_hdr_ping_pong_dummy_and_uniform_buffer() {
    assert_eq!(bloom_vram_bytes(1920, 1080), Some(24_883_256));
    assert_eq!(bloom_vram_bytes(1, 1), Some(80));
}

#[test]
fn bloom_target_budget_rejects_size_overflow() {
    assert_eq!(bloom_vram_bytes(u32::MAX, u32::MAX), None);
}
