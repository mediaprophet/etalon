//! Content-addressed texture contracts and checked mip-chain layout.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TextureColorSpace {
    Srgb,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureIdentity {
    pub source_digest: [u8; 32],
    pub color_space: TextureColorSpace,
    pub mip_semantic: crate::texture_mips::TextureMipSemantic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureUploadError {
    InvalidDimensions,
    InvalidRgba8Length,
    TextureDimensionLimit,
    ByteSizeOverflow,
    GpuBudgetRefused,
    UploadBudgetRefused,
    InvalidMipSemantic,
    InvalidMipLevelCount,
    InvalidMipLevelLength { level: usize },
    TooManyMipLevels,
}

impl std::fmt::Display for TextureUploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "texture upload: {self:?}")
    }
}

impl std::error::Error for TextureUploadError {}

/// Validates `upload`.
pub fn validate_upload(
    width: u32,
    height: u32,
    adapter_limit: u32,
    rgba8_len: usize,
) -> Result<(u64, u32), TextureUploadError> {
    if width == 0 || height == 0 {
        return Err(TextureUploadError::InvalidDimensions);
    }
    if width > adapter_limit || height > adapter_limit {
        return Err(TextureUploadError::TextureDimensionLimit);
    }
    let byte_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(TextureUploadError::ByteSizeOverflow)?;
    let bytes_per_row = width
        .checked_mul(4)
        .ok_or(TextureUploadError::ByteSizeOverflow)?;
    if rgba8_len != byte_len {
        return Err(TextureUploadError::InvalidRgba8Length);
    }
    Ok((
        u64::try_from(byte_len).map_err(|_| TextureUploadError::ByteSizeOverflow)?,
        bytes_per_row,
    ))
}

/// `mip_chain_layout`.
pub fn mip_chain_layout(width: u32, height: u32) -> Result<(u32, u64), TextureUploadError> {
    if width == 0 || height == 0 {
        return Err(TextureUploadError::InvalidDimensions);
    }
    let mut level_width = width;
    let mut level_height = height;
    let mut levels = 1u32;
    let mut total = 0u64;
    loop {
        let level_bytes = u64::from(level_width)
            .checked_mul(u64::from(level_height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(TextureUploadError::ByteSizeOverflow)?;
        total = total
            .checked_add(level_bytes)
            .ok_or(TextureUploadError::ByteSizeOverflow)?;
        if level_width == 1 && level_height == 1 {
            break;
        }
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
        levels = levels
            .checked_add(1)
            .ok_or(TextureUploadError::ByteSizeOverflow)?;
    }
    Ok((levels, total))
}

/// `mip_semantic_matches`.
pub fn mip_semantic_matches(
    color_space: TextureColorSpace,
    semantic: crate::texture_mips::TextureMipSemantic,
) -> bool {
    use crate::texture_mips::TextureMipSemantic::{AlphaMask, Color, LinearData, Normal};
    match (color_space, semantic) {
        (TextureColorSpace::Srgb, Color) => true,
        (TextureColorSpace::Srgb, AlphaMask { .. }) => semantic.alpha_cutoff().is_some(),
        (TextureColorSpace::Linear, LinearData | Normal) => true,
        _ => false,
    }
}

pub fn validate_precomputed_chain(
    width: u32,
    height: u32,
    adapter_limit: u32,
    levels: &[&[u8]],
) -> Result<(u32, u64), TextureUploadError> {
    let Some(base_level) = levels.first() else {
        return Err(TextureUploadError::InvalidMipLevelCount);
    };
    validate_upload(width, height, adapter_limit, base_level.len())?;
    let (expected_count, resident_bytes) = mip_chain_layout(width, height)?;
    if levels.len() != expected_count as usize {
        return Err(TextureUploadError::InvalidMipLevelCount);
    }
    if levels.len() > 32 {
        return Err(TextureUploadError::TooManyMipLevels);
    }

    let mut level_width = width;
    let mut level_height = height;
    let mut summed_bytes = 0u64;
    for (level, pixels) in levels.iter().enumerate() {
        let expected_bytes = u64::from(level_width)
            .checked_mul(u64::from(level_height))
            .and_then(|count| count.checked_mul(4))
            .ok_or(TextureUploadError::ByteSizeOverflow)?;
        if u64::try_from(pixels.len()).map_err(|_| TextureUploadError::ByteSizeOverflow)?
            != expected_bytes
        {
            return Err(TextureUploadError::InvalidMipLevelLength { level });
        }
        summed_bytes = summed_bytes
            .checked_add(expected_bytes)
            .ok_or(TextureUploadError::ByteSizeOverflow)?;
        if level + 1 < levels.len() {
            level_width = (level_width / 2).max(1);
            level_height = (level_height / 2).max(1);
        }
    }
    if summed_bytes != resident_bytes {
        return Err(TextureUploadError::ByteSizeOverflow);
    }
    Ok((expected_count, resident_bytes))
}
