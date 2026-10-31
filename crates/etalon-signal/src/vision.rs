//! Scalar image operations for caller-described single-channel frames.
//! A byte is an uncalibrated digital code, not temperature or radiance.
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
pub fn mean_u8_region(frame: &[u8], width: usize, height: usize, stride: usize, region: ImageRegion) -> Result<f64, SignalError> {
    if width == 0 || height == 0 || region.width == 0 || region.height == 0 || stride < width {
        return Err(SignalError::InvalidParameter);
    }
    let x_end = region.x.checked_add(region.width).ok_or(SignalError::NumericRange)?;
    let y_end = region.y.checked_add(region.height).ok_or(SignalError::NumericRange)?;
    if x_end > width || y_end > height { return Err(SignalError::InvalidParameter); }
    let span = (height - 1).checked_mul(stride).and_then(|v| v.checked_add(width)).ok_or(SignalError::NumericRange)?;
    if frame.len() < span { return Err(SignalError::LengthMismatch); }
    let count = region.width.checked_mul(region.height).ok_or(SignalError::NumericRange)?;
    let mut sum = 0.0f64;
    for y in region.y..y_end {
        let start = y * stride + region.x;
        for &sample in &frame[start..start + region.width] { sum += f64::from(sample); }
    }
    Ok(sum / count as f64)
}

#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;
