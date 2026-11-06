//! Bounded WebGPU capability reports, offscreen readback layouts, and deterministic pixel receipts.

/// Maximum dimension admitted by the bounded offscreen acceptance path.
pub const MAX_OFFSCREEN_DIMENSION: u32 = 4096;

/// Maximum staging allocation admitted for one synchronous pixel readback.
pub const MAX_PIXEL_READBACK_BYTES: u64 = 16 * 1024 * 1024;

/// Maximum UTF-8 byte length retained for adapter/driver capability strings.
pub const MAX_CAPABILITY_TEXT_BYTES: usize = 128;

/// Standard WebGPU `COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes).
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// The only pixel format owned by this acceptance slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
    Rgba8Unorm,
}

/// Stable status for a renderer capability probe or pixel receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebGpuStatus {
    AdapterUnavailable,
    DeviceUnavailable,
    AdapterOnly,
    Ready,
}

/// Stable, bounded feature facts exposed to native and browser callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WebGpuFeatureReport {
    pub shader_f16: bool,
    pub subgroup: bool,
    pub timestamp_query: bool,
    pub texture_compression_bc: bool,
}

/// Stable device limits needed to reason about the offscreen acceptance slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WebGpuLimitReport {
    pub max_texture_dimension_2d: u32,
    pub max_buffer_size: u64,
}

/// Adapter identity and the small capability subset relevant to this renderer.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WebGpuAdapterReport {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub driver_info: String,
    pub features: WebGpuFeatureReport,
    pub limits: WebGpuLimitReport,
}

/// Capability evidence that does not contain GPU handles or unbounded data.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WebGpuCapabilityReport {
    pub status: WebGpuStatus,
    pub adapter_available: bool,
    pub device_available: bool,
    pub adapter: Option<WebGpuAdapterReport>,
    pub offscreen_format: PixelFormat,
    pub max_offscreen_dimension: u32,
    pub max_pixel_readback_bytes: u64,
}

impl WebGpuCapabilityReport {
    /// Unavailable variant.
    pub fn unavailable() -> Self {
        Self {
            status: WebGpuStatus::AdapterUnavailable,
            adapter_available: false,
            device_available: false,
            adapter: None,
            offscreen_format: PixelFormat::Rgba8Unorm,
            max_offscreen_dimension: MAX_OFFSCREEN_DIMENSION,
            max_pixel_readback_bytes: MAX_PIXEL_READBACK_BYTES,
        }
    }
}

/// A fixed-size, deterministic record of one pixel acceptance attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelReceiptStatus {
    Observed,
    AdapterUnavailable,
    NotOffscreen,
    Refused,
    MapFailed,
    InvalidExtent,
}

/// Bounded evidence for a pixel readback. Pixel bytes are deliberately omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PixelReceipt {
    pub version: u16,
    pub status: PixelReceiptStatus,
    pub adapter_backed: bool,
    pub observed: bool,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub byte_len: u64,
    pub content_hash: u64,
    pub nonzero_bytes: u32,
}

impl PixelReceipt {
    /// Constructs a refusal receipt with zero observed bytes.
    pub fn refusal(
        status: PixelReceiptStatus,
        width: u32,
        height: u32,
        adapter_backed: bool,
    ) -> Self {
        Self {
            version: 1,
            status,
            adapter_backed,
            observed: false,
            width,
            height,
            format: PixelFormat::Rgba8Unorm,
            byte_len: 0,
            content_hash: 0,
            nonzero_bytes: 0,
        }
    }
}

/// Readback result with inspectable refusal/recovery evidence.
#[derive(Debug)]
pub struct PixelReadback {
    pub pixels: Option<Vec<u8>>,
    pub receipt: PixelReceipt,
}

/// Runtime evidence returned by a renderer after a pixel acceptance attempt.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WebGpuRuntimeReceipt {
    pub capabilities: WebGpuCapabilityReport,
    pub requested_extent: (u32, u32),
    pub effective_extent: Option<(u32, u32)>,
    pub pixels: PixelReceipt,
}

/// Truncates a capability string on a valid UTF-8 character boundary to at most
/// [`MAX_CAPABILITY_TEXT_BYTES`] bytes.
pub fn truncate_text(mut text: String) -> String {
    if text.len() > MAX_CAPABILITY_TEXT_BYTES {
        let mut end = MAX_CAPABILITY_TEXT_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// Alias for [`truncate_text`].
#[inline]
pub fn bounded_capability_text(text: String) -> String {
    truncate_text(text)
}

/// Deterministic 64-bit FNV-1a digest over `bytes`.
#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Alias for [`fnv1a64`] used for pixel readback content digests.
#[inline]
pub fn pixel_content_hash(pixels: &[u8]) -> u64 {
    fnv1a64(pixels)
}

/// Checked tightly-packed RGBA8 byte size for an offscreen target.
pub fn offscreen_rgba8_bytes(width: u32, height: u32) -> Option<u64> {
    u64::from(width.max(1))
        .checked_mul(u64::from(height.max(1)))?
        .checked_mul(4)
}

/// Checked padded staging buffer byte size for a 256-byte row-aligned readback.
pub fn raw_readback_bytes(width: u32, height: u32) -> Option<u64> {
    let row_bytes = u64::from(width).checked_mul(4)?;
    let align = u64::from(COPY_BYTES_PER_ROW_ALIGNMENT);
    let padded_row_bytes = row_bytes.checked_add(align - 1)? / align * align;
    padded_row_bytes.checked_mul(u64::from(height))
}

/// Returns `(padded_bytes_per_row, unpadded_bytes_per_row, readback_bytes, output_bytes)` when
/// `(width, height)` fits within [`MAX_OFFSCREEN_DIMENSION`] and [`MAX_PIXEL_READBACK_BYTES`].
pub fn readback_layout(width: u32, height: u32) -> Option<(u32, u32, u64, usize)> {
    if width == 0
        || height == 0
        || width > MAX_OFFSCREEN_DIMENSION
        || height > MAX_OFFSCREEN_DIMENSION
    {
        return None;
    }
    let row_bytes = u64::from(width).checked_mul(4)?;
    let align = u64::from(COPY_BYTES_PER_ROW_ALIGNMENT);
    let padded_row_bytes = row_bytes.checked_add(align - 1)? / align * align;
    let bytes_per_row = u32::try_from(padded_row_bytes).ok()?;
    let readback_bytes = padded_row_bytes.checked_mul(u64::from(height))?;
    if readback_bytes > MAX_PIXEL_READBACK_BYTES {
        return None;
    }
    let output_bytes = usize::try_from(row_bytes.checked_mul(u64::from(height))?).ok()?;
    Some((
        bytes_per_row,
        u32::try_from(row_bytes).ok()?,
        readback_bytes,
        output_bytes,
    ))
}

/// Clamps `(requested_width, requested_height)` and halves dimensions until the row-padded
/// readback staging footprint fits within [`MAX_PIXEL_READBACK_BYTES`].
pub fn bounded_offscreen_extent(requested_width: u32, requested_height: u32) -> (u32, u32) {
    let mut width = requested_width.clamp(1, MAX_OFFSCREEN_DIMENSION);
    let mut height = requested_height.clamp(1, MAX_OFFSCREEN_DIMENSION);

    while raw_readback_bytes(width, height)
        .map(|bytes| bytes > MAX_PIXEL_READBACK_BYTES)
        .unwrap_or(true)
    {
        if width == 1 && height == 1 {
            break;
        }
        width = width.div_ceil(2).max(1);
        height = height.div_ceil(2).max(1);
    }

    (width, height)
}

/// Build deterministic pixel evidence without requiring a GPU.
pub fn pixel_receipt_from_rgba8(width: u32, height: u32, pixels: &[u8]) -> PixelReceipt {
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4));
    let valid = width > 0
        && height > 0
        && width <= MAX_OFFSCREEN_DIMENSION
        && height <= MAX_OFFSCREEN_DIMENSION
        && expected == Some(pixels.len() as u64)
        && expected.is_some_and(|bytes| bytes <= MAX_PIXEL_READBACK_BYTES);
    if !valid {
        return PixelReceipt::refusal(PixelReceiptStatus::InvalidExtent, width, height, false);
    }

    PixelReceipt {
        version: 1,
        status: PixelReceiptStatus::Observed,
        adapter_backed: false,
        observed: true,
        width,
        height,
        format: PixelFormat::Rgba8Unorm,
        byte_len: pixels.len() as u64,
        content_hash: pixel_content_hash(pixels),
        nonzero_bytes: pixels
            .iter()
            .filter(|byte| **byte != 0)
            .count()
            .min(u32::MAX as usize) as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_layout_and_bounded_extent_respect_limits() {
        assert_eq!(offscreen_rgba8_bytes(1920, 1080), Some(8_294_400));
        assert_eq!(
            readback_layout(1920, 1080),
            Some((7680, 7680, 8_294_400, 8_294_400))
        );
        assert_eq!(readback_layout(1, 1), Some((256, 4, 256, 4)));
        assert_eq!(readback_layout(u32::MAX, 1), None);
        assert_eq!(bounded_offscreen_extent(0, 0), (1, 1));
        assert_eq!(bounded_offscreen_extent(u32::MAX, u32::MAX), (2048, 2048));
        assert_eq!(
            readback_layout(2048, 2048).unwrap().2,
            MAX_PIXEL_READBACK_BYTES
        );
        assert_eq!(readback_layout(4096, 4096), None);
    }

    #[test]
    fn pixel_receipt_is_deterministic_and_bounded() {
        let r1 = pixel_receipt_from_rgba8(1, 1, &[10, 20, 30, 255]);
        let r2 = pixel_receipt_from_rgba8(1, 1, &[10, 20, 30, 255]);
        assert_eq!(r1, r2);
        assert_eq!(r1.status, PixelReceiptStatus::Observed);
        assert_eq!(r1.nonzero_bytes, 4);
        assert_eq!(
            pixel_receipt_from_rgba8(1, 1, &[1, 2, 3]).status,
            PixelReceiptStatus::InvalidExtent
        );
        let long = "a".repeat(200);
        assert_eq!(truncate_text(long).len(), MAX_CAPABILITY_TEXT_BYTES);
    }
}
