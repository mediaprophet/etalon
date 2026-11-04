use crate::moe::nvfp4::{fp8_e4m3_to_f32, E2M1_TABLE};

/// Output projection keeps the original block accumulation order.
pub fn nvfp4_output_row(
    packed: &[u8],
    scales: &[u8],
    global: f32,
    hidden: &[f32],
    width: usize,
) -> f32 {
    let mut row_dot = 0.0f32;
    for b in 0..width / 16 {
        let eff_scale = fp8_e4m3_to_f32(scales[b]) * global;
        let bytes = &packed[b * 8..(b + 1) * 8];
        let offset = b * 16;
        let mut b_dot = 0.0f32;
        for i in 0..8 {
            let low = (bytes[i] & 0x0f) as usize;
            let high = ((bytes[i] >> 4) & 0x0f) as usize;
            b_dot += E2M1_TABLE[low] * hidden[offset + i * 2]
                + E2M1_TABLE[high] * hidden[offset + i * 2 + 1];
        }
        row_dot += b_dot * eff_scale;
    }
    row_dot
}
