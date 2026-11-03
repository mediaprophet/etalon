use super::{PleNgramConfig, PleNgramConfigSource, PleTokenHistory, MAX_PLE_HEADS, MAX_PLE_NGRAM};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PleNgramError {
    IncompleteConfig,
    UnsupportedNgramSize,
    OutputTooSmall,
    RowOutOfRange,
}

impl core::fmt::Display for PleNgramError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IncompleteConfig => write!(f, "Qwen4Exp PLE metadata is incomplete"),
            Self::UnsupportedNgramSize => write!(f, "Qwen4Exp PLE n-gram size is unsupported"),
            Self::OutputTooSmall => write!(f, "PLE row-ID output buffer is too small"),
            Self::RowOutOfRange => write!(f, "PLE row ID exceeds the extracted table"),
        }
    }
}

impl core::error::Error for PleNgramError {}

/// Select rows without advancing history. A late row error can leave a written
/// output prefix; remaining output and history retain their previous values.
pub fn select_rows<C: PleNgramConfigSource + ?Sized>(
    config: &C,
    token: u32,
    history: &PleTokenHistory,
    table_rows: u64,
    out: &mut [u64],
) -> Result<usize, PleNgramError> {
    select_rows_view(config.ple_ngram_config(), token, history, table_rows, out)
}

/// Generate the model's global PLE row IDs for one token.
///
/// For a 3-gram model this emits eight bigram and eight trigram row IDs.  The
/// signed Euclidean remainder matches tensor-runtime modulo for a negative
/// i64 XOR result.
// Preserve the original indexed history and trained-multiplier evaluation order.
#[allow(clippy::needless_range_loop)]
pub(super) fn select_rows_view(
    config: PleNgramConfig<'_>,
    token: u32,
    history: &PleTokenHistory,
    table_rows: u64,
    out: &mut [u64],
) -> Result<usize, PleNgramError> {
    if !config.is_complete() {
        return Err(PleNgramError::IncompleteConfig);
    }
    let ngram_size = config.ngram_size as usize;
    if !(2..=MAX_PLE_NGRAM).contains(&ngram_size) {
        return Err(PleNgramError::UnsupportedNgramSize);
    }
    let required = (ngram_size - 1)
        .checked_mul(config.heads_per_ngram as usize)
        .ok_or(PleNgramError::OutputTooSmall)?;
    if required > MAX_PLE_HEADS || out.len() < required {
        return Err(PleNgramError::OutputTooSmall);
    }

    let mut shifted = [config.eos_token_id; MAX_PLE_NGRAM];
    shifted[0] = token;
    for shift in 1..ngram_size {
        if shift <= history.count {
            shifted[shift] = history.previous[shift - 1];
        }
    }

    let heads_per_ngram = config.heads_per_ngram as usize;
    let mut written = 0usize;
    for ngram in 2..=ngram_size {
        let mut mixed = (shifted[0] as i64).wrapping_mul(config.layer_multipliers[0]);
        for position in 1..ngram {
            mixed ^= (shifted[position] as i64).wrapping_mul(config.layer_multipliers[position]);
        }
        let first_head = (ngram - 2) * heads_per_ngram;
        for head in first_head..first_head + heads_per_ngram {
            let size = config.head_vocab_sizes[head];
            let offset = config.head_offsets[head];
            let row = mixed
                .rem_euclid(size)
                .checked_add(offset)
                .ok_or(PleNgramError::RowOutOfRange)?;
            if row < 0 || row as u64 >= table_rows {
                return Err(PleNgramError::RowOutOfRange);
            }
            out[written] = row as u64;
            written += 1;
        }
    }
    Ok(written)
}

/// Sort and deduplicate a selected row list in caller storage.
pub fn sort_dedup_rows(rows: &mut [u64]) -> usize {
    rows.sort_unstable();
    let mut kept = 0usize;
    for read in 0..rows.len() {
        if kept == 0 || rows[read] != rows[kept - 1] {
            rows[kept] = rows[read];
            kept += 1;
        }
    }
    kept
}
