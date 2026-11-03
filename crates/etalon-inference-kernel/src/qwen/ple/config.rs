/// Borrowed fixed-capacity numerical metadata; no model/serde dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleNgramConfig<'a> {
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub eos_token_id: u32,
    pub embedding_row_width: u32,
    pub multiplier_count: u32,
    pub layer_multipliers: &'a [i64; 8],
    pub head_count: u32,
    pub head_offsets: &'a [i64; 32],
    pub head_vocab_sizes: &'a [i64; 32],
}
/// The host owns this projection. Each public operation snapshots it once;
/// no allocation guarantee is made for arbitrary downstream implementations.
pub trait PleNgramConfigSource {
    fn ple_ngram_config(&self) -> PleNgramConfig<'_>;
}
impl PleNgramConfigSource for PleNgramConfig<'_> {
    fn ple_ngram_config(&self) -> PleNgramConfig<'_> {
        *self
    }
}
impl PleNgramConfig<'_> {
    /// True only when every value required to reproduce trained row IDs is present.
    pub fn is_complete(&self) -> bool {
        let required_heads = self
            .ngram_size
            .saturating_sub(1)
            .saturating_mul(self.heads_per_ngram);
        self.ngram_size >= 2
            && self.multiplier_count >= self.ngram_size
            && required_heads != 0
            && self.head_count >= required_heads
            && required_heads <= self.head_offsets.len() as u32
            && self.embedding_row_width != 0
            && self.head_vocab_sizes[..required_heads as usize]
                .iter()
                .all(|&size| size > 0)
    }
}
