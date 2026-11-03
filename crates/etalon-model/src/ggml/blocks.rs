use super::*;

pub(super) fn dequant_q4_0(
    raw: &[u8],
    n_elems: usize,
    out: &mut [f32],
) -> Result<usize, GgmlDequantError> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let n_blocks = n_elems.div_ceil(BLOCK_ELEMS);
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(GgmlDequantError::TruncatedInput);
    }
    for b in 0..n_blocks {
        let bs = b * BLOCK_BYTES;
        let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();
        let half = BLOCK_ELEMS / 2;
        for j in 0..half {
            if b * BLOCK_ELEMS + j >= n_elems {
                break;
            }
            let byte = raw[bs + 2 + j];
            let x0 = (byte & 0x0F) as i32 - 8;
            let x1 = ((byte >> 4) & 0x0F) as i32 - 8;
            out[b * BLOCK_ELEMS + j] = x0 as f32 * scale;
            let hi = b * BLOCK_ELEMS + j + half;
            if hi < n_elems {
                out[hi] = x1 as f32 * scale;
            }
        }
    }
    Ok(n_elems)
}

/// `dequantize_row_q5_0` from ggml-quants.c — 5-bit weights, 32 elems per 22-byte block.
#[allow(clippy::needless_range_loop)] // Preserve the original indexed decoder.
pub(super) fn dequant_q5_0(
    raw: &[u8],
    n_elems: usize,
    out: &mut [f32],
) -> Result<usize, GgmlDequantError> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 22;
    let n_blocks = n_elems.div_ceil(BLOCK_ELEMS);
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(GgmlDequantError::TruncatedInput);
    }
    for b in 0..n_blocks {
        let bs = b * BLOCK_BYTES;
        let d = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();
        let qh = u32::from_le_bytes([raw[bs + 2], raw[bs + 3], raw[bs + 4], raw[bs + 5]]);
        let qs = &raw[bs + 6..bs + 22];
        let half = BLOCK_ELEMS / 2;
        for j in 0..half {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0F) as u32 | xh_0) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | xh_1) as i32 - 16;
            let lo = b * BLOCK_ELEMS + j;
            if lo < n_elems {
                out[lo] = x0 as f32 * d;
            }
            let hi = lo + half;
            if hi < n_elems {
                out[hi] = x1 as f32 * d;
            }
        }
    }
    Ok(n_elems)
}

pub(super) fn dequant_q8_0(
    raw: &[u8],
    n_elems: usize,
    out: &mut [f32],
) -> Result<usize, GgmlDequantError> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let n_blocks = n_elems.div_ceil(BLOCK_ELEMS);
    if raw.len() < n_blocks * BLOCK_BYTES {
        return Err(GgmlDequantError::TruncatedInput);
    }
    for b in 0..n_blocks {
        let bs = b * BLOCK_BYTES;
        let scale = half::f16::from_le_bytes([raw[bs], raw[bs + 1]]).to_f32();
        let elems = BLOCK_ELEMS.min(n_elems - b * BLOCK_ELEMS);
        for j in 0..elems {
            out[b * BLOCK_ELEMS + j] = raw[bs + 2 + j] as i8 as f32 * scale;
        }
    }
    Ok(n_elems)
}
