//! STELLAR §A A1a — GPU top-K reduction: CPU oracle, host merge, and the WGSL kernel.
//!
//! Decode is memory-bandwidth-bound; the sentinel/sampler only needs the high-probability
//! mass, not the 49 k near-zero tail. Instead of reading back the full logit vector
//! (~196 KB/token) and doing a CPU argmax, the GPU reduces each block of the vocabulary to
//! its top-K candidates; the host merges those into the global top-K. The CPU oracle here is
//! the byte-for-byte reference the on-device kernel is verified against (see `topk_gpu.rs`),
//! exactly as `ternary.rs` anchors `ternary_gpu.rs`.
//!
//! Contract (must match `shaders/topk_reduction.wgsl`): NaN → −∞ (never selected); ties broken
//! toward the LOWER token id (deterministic); K=1 == argmax.

/// One top-K entry: a token id and its raw logit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKItem {
    pub token_id: u32,
    pub logit: f32,
}

/// The block-reduction kernel (auto bind-group layout; entry `topk_block`).
pub const TOPK_REDUCTION_WGSL: &str = include_str!("topk_reduction.wgsl");

/// Block size each workgroup reduces — must equal `MAX_BLOCK` in the WGSL (`var<workgroup>` cap).
pub const TOPK_BLOCK_SIZE: usize = 1024;

/// Largest K the host paths support (kept generous; the kernel itself is K-agnostic per round).
pub const TOPK_MAX_K: usize = 64;

/// 16-byte `Params` uniform: `n, k, block_size, cand_base` (cand_base=0 for single-chunk).
pub fn topk_params_bytes(n: u32, k: u32, block_size: u32) -> [u8; 16] {
    topk_params_bytes_with_base(n, k, block_size, 0)
}

/// Like [`topk_params_bytes`] with a non-zero candidate write base for multi-chunk mega-pass.
pub fn topk_params_bytes_with_base(n: u32, k: u32, block_size: u32, cand_base: u32) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&n.to_le_bytes());
    b[4..8].copy_from_slice(&k.to_le_bytes());
    b[8..12].copy_from_slice(&block_size.to_le_bytes());
    b[12..16].copy_from_slice(&cand_base.to_le_bytes());
    b
}

/// Normalize: NaN → −∞ so it can never win a comparison.
#[inline]
fn clean(x: f32) -> f32 {
    if x.is_nan() {
        f32::NEG_INFINITY
    } else {
        x
    }
}

/// Order two `(id, logit)` candidates: higher logit first, lower id on ties.
#[inline]
fn cmp_desc(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

/// CPU reference top-K over a full logit vector. Drops −∞ (e.g. masked) entries.
pub fn topk_cpu(logits: &[f32], k: usize) -> Vec<TopKItem> {
    let mut v: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, clean(x)))
        .collect();
    v.sort_by(cmp_desc);
    v.into_iter()
        .filter(|(_, val)| *val > f32::NEG_INFINITY)
        .take(k)
        .map(|(token_id, logit)| TopKItem { token_id, logit })
        .collect()
}

/// Merge per-block GPU candidates (`num_blocks × k` pairs) into the global top-K.
/// Blocks cover disjoint index ranges, so candidate ids are unique. Drops −∞ entries
/// and any id in `masked` (the governance/sieve veto — "a masked token never returned").
pub fn merge_block_candidates(
    cand_val: &[f32],
    cand_idx: &[u32],
    k: usize,
    masked: Option<&dyn Fn(u32) -> bool>,
) -> Vec<TopKItem> {
    let mut v: Vec<(u32, f32)> = cand_val
        .iter()
        .zip(cand_idx.iter())
        .map(|(val, idx)| (*idx, clean(*val)))
        .filter(|(idx, val)| *val > f32::NEG_INFINITY && masked.map(|m| !m(*idx)).unwrap_or(true))
        .collect();
    v.sort_by(cmp_desc);
    v.truncate(k);
    v.into_iter()
        .map(|(token_id, logit)| TopKItem { token_id, logit })
        .collect()
}

/// Bounded merge with the same score/id ordering as the reference merge.
pub fn insert_candidate(
    out: &mut [TopKItem],
    count: usize,
    k: usize,
    token_id: u32,
    logit: f32,
) -> usize {
    if logit.is_nan() || logit <= f32::NEG_INFINITY {
        return count;
    }
    let cap = k.min(out.len());
    let count = count.min(cap);
    let position = out[..count]
        .iter()
        .position(|old| logit > old.logit || (logit == old.logit && token_id < old.token_id))
        .unwrap_or(count);
    if position >= cap {
        return count;
    }
    let next = (count + 1).min(cap);
    out.copy_within(position..next - 1, position + 1);
    out[position] = TopKItem { token_id, logit };
    next
}
