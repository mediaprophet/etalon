//! Scalar image, perceptual hash, preprocessing, and biosense quality kernels
//! over caller-owned buffers (`#![no_std]`, zero-heap).
//!
//! A byte is an uncalibrated digital code, not physical temperature or radiance.
use crate::filter::sqrt_f64;
use crate::SignalError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// Compute the mean digital code in a rectangular, single-channel u8 frame
/// region. Modality and calibration remain external metadata and are neither
/// accepted nor validated here. No RGB, IR radiometry, object recognition or
/// physical temperature is inferred.
pub fn mean_u8_region(
    frame: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    region: ImageRegion,
) -> Result<f64, SignalError> {
    if width == 0 || height == 0 || region.width == 0 || region.height == 0 || stride < width {
        return Err(SignalError::InvalidParameter);
    }
    let x_end = region
        .x
        .checked_add(region.width)
        .ok_or(SignalError::NumericRange)?;
    let y_end = region
        .y
        .checked_add(region.height)
        .ok_or(SignalError::NumericRange)?;
    if x_end > width || y_end > height {
        return Err(SignalError::InvalidParameter);
    }
    let span = (height - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(width))
        .ok_or(SignalError::NumericRange)?;
    if frame.len() < span {
        return Err(SignalError::LengthMismatch);
    }
    let count = region
        .width
        .checked_mul(region.height)
        .ok_or(SignalError::NumericRange)?;
    let mut sum = 0.0f64;
    for y in region.y..y_end {
        let start = y * stride + region.x;
        for &sample in &frame[start..start + region.width] {
            sum += f64::from(sample);
        }
    }
    Ok(sum / count as f64)
}

/// Compute the 256-bin intensity histogram of a single-channel `u8` buffer.
pub fn histogram_u8(gray: &[u8]) -> [u32; 256] {
    let mut bins = [0u32; 256];
    for &px in gray {
        let idx = usize::from(px);
        bins[idx] = bins[idx].saturating_add(1);
    }
    bins
}

/// Compute Otsu's inter-class variance maximizing threshold for a 256-bin histogram.
///
/// If `total == 0`, the total sample count is summed from `hist`. Returns `0` for an
/// empty or uniform-zero histogram.
pub fn otsu_threshold_from_hist(hist: &[u32; 256], total: u32) -> u8 {
    let total_u64 = if total > 0 {
        u64::from(total)
    } else {
        hist.iter().map(|&c| u64::from(c)).sum()
    };
    if total_u64 == 0 {
        return 0;
    }
    let mut sum_all = 0u64;
    for (i, &c) in hist.iter().enumerate() {
        sum_all += (i as u64) * u64::from(c);
    }
    let mut sum_b = 0u64;
    let mut w_b = 0u64;
    let mut max_var = -1.0f64;
    let mut thresh = 0u8;
    for (t, &count) in hist.iter().enumerate() {
        w_b += u64::from(count);
        if w_b == 0 {
            continue;
        }
        if w_b >= total_u64 {
            break;
        }
        let w_f = total_u64 - w_b;
        sum_b += (t as u64) * u64::from(count);
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum_all.saturating_sub(sum_b)) as f64 / w_f as f64;
        let diff = m_b - m_f;
        let var = (w_b as f64) * (w_f as f64) * diff * diff;
        if var > max_var {
            max_var = var;
            thresh = t as u8;
        }
    }
    thresh
}

/// Compute a 64-bit average perceptual hash (aHash) from an 8×8 luma patch.
pub fn ahash_from_luma_64(luma64: &[u8; 64]) -> u64 {
    let mut sum = 0u32;
    for &p in luma64 {
        sum += u32::from(p);
    }
    let mean = (sum / 64) as u8;
    let mut hash = 0u64;
    for (i, &p) in luma64.iter().enumerate() {
        if p >= mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Compute a 64-bit horizontal difference hash (dHash) from a 9×8 luma patch.
pub fn dhash_from_luma_72(luma9x8: &[u8; 72]) -> u64 {
    let mut hash = 0u64;
    let mut bit = 0u32;
    for y in 0..8usize {
        let row = y * 9;
        for x in 0..8usize {
            if luma9x8[row + x + 1] >= luma9x8[row + x] {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    hash
}

/// Popcount of differing bits between two 64-bit hashes.
#[inline]
pub const fn hamming_distance_u64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Normalized Hamming distance in `[0.0, 1.0]` between two 64-bit hashes.
#[inline]
pub fn normalized_hamming_u64(a: u64, b: u64) -> f32 {
    hamming_distance_u64(a, b) as f32 / 64.0
}

/// Cosine similarity `dot(a, b) / (|a| |b|)` over equal-length `f32` slices.
///
/// Returns `0.0` if slices are empty, differ in length, or have near-zero norm.
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let ai = f64::from(a[i]);
        let bi = f64::from(b[i]);
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = sqrt_f64(na) * sqrt_f64(nb);
    if !denom.is_finite() || denom <= 1e-12 {
        return 0.0;
    }
    ((dot / denom) as f32).clamp(-1.0, 1.0)
}

/// Cosine distance `1.0 - cosine_similarity_f32(a, b)`.
#[inline]
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 1.0;
    }
    1.0 - cosine_similarity_f32(a, b)
}

/// Nearest-neighbour RGB8 resize into caller-supplied `out` (`dst_w * dst_h * 3` bytes).
///
/// If `src_stride == 0`, a tightly packed stride of `src_w * 3` is assumed.
pub fn resize_nearest_rgb8_into(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
    out: &mut [u8],
) -> Result<usize, SignalError> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err(SignalError::InvalidParameter);
    }
    let min_row_bytes = (src_w as usize)
        .checked_mul(3)
        .ok_or(SignalError::NumericRange)?;
    let stride = if src_stride == 0 {
        min_row_bytes
    } else {
        src_stride as usize
    };
    if stride < min_row_bytes {
        return Err(SignalError::InvalidParameter);
    }
    let required_src = (src_h as usize - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(min_row_bytes))
        .ok_or(SignalError::NumericRange)?;
    if src.len() < required_src {
        return Err(SignalError::LengthMismatch);
    }
    let written = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .and_then(|v| v.checked_mul(3))
        .ok_or(SignalError::NumericRange)?;
    if out.len() < written {
        return Err(SignalError::LengthMismatch);
    }
    for oy in 0..dst_h {
        let sy = ((u64::from(oy) * u64::from(src_h)) / u64::from(dst_h)) as usize;
        let row_off = sy * stride;
        for ox in 0..dst_w {
            let sx = ((u64::from(ox) * u64::from(src_w)) / u64::from(dst_w)) as usize;
            let soff = row_off + sx * 3;
            let doff = ((oy as usize) * (dst_w as usize) + (ox as usize)) * 3;
            out[doff] = src[soff];
            out[doff + 1] = src[soff + 1];
            out[doff + 2] = src[soff + 2];
        }
    }
    Ok(written)
}

/// Normalize packed RGB8 (`w * h * 3` bytes) into planar `f32` CHW (`3 * w * h` elements).
pub fn normalize_rgb8_to_f32_chw_into(
    rgb: &[u8],
    w: u32,
    h: u32,
    mean: [f32; 3],
    std: [f32; 3],
    out: &mut [f32],
) -> Result<usize, SignalError> {
    if w == 0 || h == 0 {
        return Err(SignalError::InvalidParameter);
    }
    if mean.iter().any(|v| !v.is_finite()) || std.iter().any(|v| !v.is_finite() || *v == 0.0) {
        return Err(SignalError::InvalidParameter);
    }
    let pixels = (w as usize)
        .checked_mul(h as usize)
        .ok_or(SignalError::NumericRange)?;
    let total = pixels.checked_mul(3).ok_or(SignalError::NumericRange)?;
    if rgb.len() < total || out.len() < total {
        return Err(SignalError::LengthMismatch);
    }
    for i in 0..pixels {
        let r = f32::from(rgb[i * 3]) / 255.0;
        let g = f32::from(rgb[i * 3 + 1]) / 255.0;
        let b = f32::from(rgb[i * 3 + 2]) / 255.0;
        out[i] = (r - mean[0]) / std[0];
        out[pixels + i] = (g - mean[1]) / std[1];
        out[2 * pixels + i] = (b - mean[2]) / std[2];
    }
    Ok(total)
}

/// Intersection-over-Union (IoU) of two `[x, y, w, h]` boxes in `u16` coordinates.
#[inline]
pub fn iou_xywh_u16(a: [u16; 4], b: [u16; 4]) -> f32 {
    let ax0 = f32::from(a[0]);
    let ay0 = f32::from(a[1]);
    let aw = f32::from(a[2]);
    let ah = f32::from(a[3]);
    let bx0 = f32::from(b[0]);
    let by0 = f32::from(b[1]);
    let bw = f32::from(b[2]);
    let bh = f32::from(b[3]);

    let ax1 = ax0 + aw;
    let ay1 = ay0 + ah;
    let bx1 = bx0 + bw;
    let by1 = by0 + bh;

    let ix0 = ax0.max(bx0);
    let iy0 = ay0.max(by0);
    let ix1 = ax1.min(bx1);
    let iy1 = ay1.min(by1);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let inter = iw * ih;
    let union = aw * ah + bw * bh - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Variance of the 4-neighbour discrete Laplacian over interior pixels of a `width × height`
/// single-channel `u8` frame (higher = sharper).
pub fn laplacian_blur_variance_u8(luma: &[u8], width: usize, height: usize) -> f32 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let Some(total) = width.checked_mul(height) else {
        return 0.0;
    };
    if luma.len() < total {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut n = 0.0f32;
    for y in 1..(height - 1) {
        let row = y * width;
        for x in 1..(width - 1) {
            let c = f32::from(luma[row + x]);
            let lap = f32::from(luma[row + x + 1])
                + f32::from(luma[row + x - 1])
                + f32::from(luma[row + width + x])
                + f32::from(luma[row - width + x])
                - 4.0 * c;
            sum += lap;
            sum2 += lap * lap;
            n += 1.0;
        }
    }
    if n < 1.0 {
        return 0.0;
    }
    let mean = sum / n;
    (sum2 / n - mean * mean).max(0.0)
}

/// Mean absolute difference (MAD) between two single-channel `u8` frames.
pub fn frame_motion_mad_u8(prev: &[u8], curr: &[u8]) -> f32 {
    let n = prev.len().min(curr.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0u64;
    for i in 0..n {
        sum += u64::from(prev[i].abs_diff(curr[i]));
    }
    (sum as f64 / n as f64) as f32
}

/// Mean RGB values `[R, G, B]` inside a clamped rectangular ROI `(x0, y0, rw, rh)` of a
/// packed `width × height` RGB8 frame.
pub fn roi_mean_rgb_u8(
    rgb: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
) -> [f32; 3] {
    if width == 0 || height == 0 || rw == 0 || rh == 0 {
        return [0.0; 3];
    }
    let Some(need) = width.checked_mul(height).and_then(|p| p.checked_mul(3)) else {
        return [0.0; 3];
    };
    if rgb.len() < need {
        return [0.0; 3];
    }
    let x1 = x0.min(width.saturating_sub(1));
    let y1 = y0.min(height.saturating_sub(1));
    let x2 = x0.saturating_add(rw).min(width);
    let y2 = y0.saturating_add(rh).min(height);
    if x1 >= x2 || y1 >= y2 {
        return [0.0; 3];
    }
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let mut count = 0u64;
    for y in y1..y2 {
        let row_off = y * width * 3;
        for x in x1..x2 {
            let idx = row_off + x * 3;
            r_sum += u64::from(rgb[idx]);
            g_sum += u64::from(rgb[idx + 1]);
            b_sum += u64::from(rgb[idx + 2]);
            count += 1;
        }
    }
    if count == 0 {
        return [0.0; 3];
    }
    let n = count as f32;
    [r_sum as f32 / n, g_sum as f32 / n, b_sum as f32 / n]
}

#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;

#[cfg(test)]
mod vision_kernel_tests {
    use super::*;

    #[test]
    fn histogram_and_otsu_bimodal() {
        let mut samples = [20u8; 64];
        for s in &mut samples[32..] {
            *s = 220;
        }
        let hist = histogram_u8(&samples);
        assert_eq!(hist[20], 32);
        assert_eq!(hist[220], 32);
        let thr = otsu_threshold_from_hist(&hist, 64);
        assert!((20..=220).contains(&thr));
    }

    #[test]
    fn perceptual_hashes_and_distances() {
        let mut luma64 = [0u8; 64];
        for (i, px) in luma64.iter_mut().enumerate() {
            *px = (i as u8) * 3;
        }
        let h1 = ahash_from_luma_64(&luma64);
        let h2 = ahash_from_luma_64(&luma64);
        assert_eq!(hamming_distance_u64(h1, h2), 0);
        assert!((normalized_hamming_u64(h1, !h1) - 1.0).abs() < 1e-6);

        let mut luma72 = [0u8; 72];
        for (i, px) in luma72.iter_mut().enumerate() {
            *px = (i % 9) as u8 * 20;
        }
        assert_eq!(dhash_from_luma_72(&luma72), u64::MAX);

        let a = [0.6f32, 0.8];
        let b = [0.8f32, -0.6];
        assert!((cosine_similarity_f32(&a, &a) - 1.0).abs() < 1e-6);
        assert!(cosine_distance_f32(&a, &a).abs() < 1e-6);
        assert!(cosine_similarity_f32(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn preprocess_resize_normalize_and_iou() {
        let src = [255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let mut out = [0u8; 3];
        let written = resize_nearest_rgb8_into(&src, 2, 2, 6, 1, 1, &mut out).unwrap();
        assert_eq!(written, 3);
        assert_eq!(out, [255, 0, 0]);

        let mut chw = [0.0f32; 3];
        let n = normalize_rgb8_to_f32_chw_into(&out, 1, 1, [0.0; 3], [1.0; 3], &mut chw).unwrap();
        assert_eq!(n, 3);
        assert!((chw[0] - 1.0).abs() < 1e-6);
        assert!(chw[1].abs() < 1e-6);

        assert!((iou_xywh_u16([10, 10, 20, 20], [10, 10, 20, 20]) - 1.0).abs() < 1e-6);
        assert_eq!(iou_xywh_u16([0, 0, 10, 10], [20, 20, 10, 10]), 0.0);
    }

    #[test]
    fn biosense_blur_motion_and_roi_mean() {
        let flat = [100u8; 25];
        assert_eq!(laplacian_blur_variance_u8(&flat, 5, 5), 0.0);
        let mut impulse = [0u8; 25];
        impulse[12] = 255;
        assert!(laplacian_blur_variance_u8(&impulse, 5, 5) > 100.0);

        assert!((frame_motion_mad_u8(&[10, 20, 30], &[20, 20, 40]) - (20.0 / 3.0)).abs() < 1e-5);

        let rgb = [
            10u8, 20, 30, 40, 50, 60, // row 0
            70, 80, 90, 100, 110, 120, // row 1
        ];
        let mean = roi_mean_rgb_u8(&rgb, 2, 2, 0, 0, 2, 2);
        assert!((mean[0] - 55.0).abs() < 1e-5);
        assert!((mean[1] - 65.0).abs() < 1e-5);
        assert!((mean[2] - 75.0).abs() < 1e-5);
    }
}
