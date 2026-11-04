// ─── Q4_K dequantization ──────────────────────────────────────────────────────

/// Bytes per Q4_K block (2 × f16 scales + 16 data bytes = 20 bytes, 32 weights).
pub const Q4_K_BLOCK_BYTES: usize = 20;
/// Number of weights encoded per Q4_K block.
pub const Q4_K_BLOCK_SIZE: usize = 32;

/// Dequantize a single Q4_K block.
///
/// Layout: [scale_f16(2)] [min_f16(2)] [nibbles(16)]
#[inline]
pub fn dequantize_q4_k_block(block: &[u8; Q4_K_BLOCK_BYTES], out: &mut [f32; Q4_K_BLOCK_SIZE]) {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let min = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    for i in 0..16 {
        let byte = block[4 + i];
        out[i * 2] = (byte & 0x0F) as f32 * scale + min;
        out[i * 2 + 1] = (byte >> 4) as f32 * scale + min;
    }
}

/// Dequantize an entire Q4_K tensor slice to a flat `Vec<f32>`.
pub fn dequantize_q4_k_tensor(q4k_bytes: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = q4k_bytes.len() / Q4_K_BLOCK_BYTES;
    let mut out = vec![0f32; num_blocks * Q4_K_BLOCK_SIZE];
    for b in 0..num_blocks {
        let block: &[u8; Q4_K_BLOCK_BYTES] = q4k_bytes[b * Q4_K_BLOCK_BYTES..][..Q4_K_BLOCK_BYTES]
            .try_into()
            .unwrap();
        let mut deq = [0f32; Q4_K_BLOCK_SIZE];
        dequantize_q4_k_block(block, &mut deq);
        out[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE].copy_from_slice(&deq);
    }
    out.truncate(num_elements);
    out
}

