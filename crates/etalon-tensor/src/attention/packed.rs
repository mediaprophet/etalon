use super::{FusedAttentionParams, FUSED_ATTENTION_WGSL};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PackedAttentionError {
    #[error("weight byte length differs from the uniform contract")]
    WeightLength,
    #[error("active attention mask does not contain all declared batch pages")]
    MaskTooSmall,
    #[error("packed attention size exceeds u32 shader addressing")]
    SizeOverflow,
    #[error("packed attention payload exceeds the caller byte limit")]
    ByteLimit,
    #[error("packed attention output buffer is too small")]
    OutputTooSmall,
}

/// Pack weights followed by aligned mask pages without allocating or changing the uniform ABI.
/// `byte_limit` must fit both device buffer/storage-binding limits and the caller's budget.
/// Failure leaves `out` untouched. The returned count includes one dummy word for empty inputs.
pub fn pack_attention_weights_and_mask(
    params: &FusedAttentionParams,
    weights: &[u8],
    mask: &[u32],
    out: &mut [u32],
    byte_limit: u64,
) -> Result<usize, PackedAttentionError> {
    if u64::try_from(weights.len()).ok() != Some(u64::from(params.weight_byte_len)) {
        return Err(PackedAttentionError::WeightLength);
    }
    let mask_required = if params.mask_active != 0 {
        params
            .mask_word_count
            .checked_mul(params.num_tokens_in_batch.max(1))
            .ok_or(PackedAttentionError::SizeOverflow)?
    } else {
        0
    };
    if u64::try_from(mask.len()).map_err(|_| PackedAttentionError::SizeOverflow)?
        < u64::from(mask_required)
    {
        return Err(PackedAttentionError::MaskTooSmall);
    }
    let weight_words = weights.len() / 4 + usize::from(!weights.len().is_multiple_of(4));
    let required = weight_words
        .checked_add(mask.len())
        .ok_or(PackedAttentionError::SizeOverflow)?
        .max(1);
    let words = u32::try_from(required).map_err(|_| PackedAttentionError::SizeOverflow)?;
    if u64::from(words) * 4 > byte_limit {
        return Err(PackedAttentionError::ByteLimit);
    }
    if out.len() < required {
        return Err(PackedAttentionError::OutputTooSmall);
    }
    out[..required].fill(0);
    for (word, bytes) in out[..weight_words].iter_mut().zip(weights.chunks(4)) {
        let mut padded = [0u8; 4];
        padded[..bytes.len()].copy_from_slice(bytes);
        *word = u32::from_le_bytes(padded);
    }
    out[weight_words..weight_words + mask.len()].copy_from_slice(mask);
    Ok(required)
}

/// Shader for the additive four-storage-buffer pipeline (weights + mask share binding 1).
pub fn packed_fused_attention_wgsl() -> String {
    FUSED_ATTENTION_WGSL.replace(
        "@group(0) @binding(5) var<storage, read> kv_mask_words: array<u32>;",
        "",
    ).replace(
        "return (kv_mask_words[mask_base + word] & (1u << offset)) != 0u;",
        "let packed_mask_base = params.weight_byte_len / 4u + select(0u, 1u, (params.weight_byte_len % 4u) != 0u);\n    return (weight_words[packed_mask_base + mask_base + word] & (1u << offset)) != 0u;",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn pads_odd_weights_and_preserves_every_batch_mask_page_and_output_guard() {
        let params = FusedAttentionParams {
            weight_byte_len: 5,
            mask_active: 1,
            mask_word_count: 2,
            num_tokens_in_batch: 2,
            ..FusedAttentionParams::zeroed()
        };
        let mut out = [0xaaaa_aaaa; 7];
        assert_eq!(
            pack_attention_weights_and_mask(&params, &[1, 2, 3, 4, 5], &[1, 2, 4, 8], &mut out, 24),
            Ok(6)
        );
        assert_eq!(out, [0x0403_0201, 5, 1, 2, 4, 8, 0xaaaa_aaaa]);
    }

    #[test]
    fn failures_leave_output_untouched() {
        let params = FusedAttentionParams {
            weight_byte_len: 1,
            mask_active: 1,
            mask_word_count: 1,
            num_tokens_in_batch: 2,
            ..FusedAttentionParams::zeroed()
        };
        let mut out = [99; 4];
        assert_eq!(
            pack_attention_weights_and_mask(&params, &[], &[1, 2], &mut out, 16),
            Err(PackedAttentionError::WeightLength)
        );
        assert_eq!(
            pack_attention_weights_and_mask(&params, &[0], &[1], &mut out, 16),
            Err(PackedAttentionError::MaskTooSmall)
        );
        assert_eq!(
            pack_attention_weights_and_mask(&params, &[0], &[1, 2], &mut out, 11),
            Err(PackedAttentionError::ByteLimit)
        );
        assert_eq!(
            pack_attention_weights_and_mask(&params, &[0], &[1, 2], &mut out[..2], 16),
            Err(PackedAttentionError::OutputTooSmall)
        );
        assert_eq!(out, [99; 4]);
    }

    #[test]
    fn rejects_mask_address_overflow_before_writing() {
        let params = FusedAttentionParams {
            mask_active: 1,
            mask_word_count: u32::MAX,
            num_tokens_in_batch: 2,
            ..FusedAttentionParams::zeroed()
        };
        let mut out = [99; 1];
        assert_eq!(
            pack_attention_weights_and_mask(&params, &[], &[], &mut out, u64::MAX),
            Err(PackedAttentionError::SizeOverflow)
        );
        assert_eq!(out, [99]);
    }

    #[test]
    fn dense_empty_payload_has_a_valid_dummy_binding_word() {
        let params = FusedAttentionParams {
            mask_word_count: 99,
            ..FusedAttentionParams::zeroed()
        };
        let mut out = [99; 2];
        assert_eq!(
            pack_attention_weights_and_mask(&params, &[], &[], &mut out, 4),
            Ok(1)
        );
        assert_eq!(out, [0, 99]);
    }
}
