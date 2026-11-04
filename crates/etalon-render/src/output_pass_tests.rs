use super::{output_target_vram_bytes, OutputParams};

#[test]
fn sdr_output_budget_counts_the_full_scene_target_and_uniform() {
    assert_eq!(output_target_vram_bytes(1920, 1080, 4), Some(8_294_432));
    assert_eq!(output_target_vram_bytes(1920, 1080, 8), Some(16_588_832));
    assert_eq!(std::mem::size_of::<OutputParams>(), 32);
    assert_eq!(output_target_vram_bytes(u32::MAX, u32::MAX, 8), None);
}
