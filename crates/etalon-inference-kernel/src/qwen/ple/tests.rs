use super::*;
struct Fixture {
    ngram_size: u32,
    heads_per_ngram: u32,
    eos_token_id: u32,
    embedding_row_width: u32,
    multiplier_count: u32,
    layer_multipliers: [i64; 8],
    head_count: u32,
    head_offsets: [i64; 32],
    head_vocab_sizes: [i64; 32],
}
impl PleNgramConfigSource for Fixture {
    fn ple_ngram_config(&self) -> PleNgramConfig<'_> {
        PleNgramConfig {
            ngram_size: self.ngram_size,
            heads_per_ngram: self.heads_per_ngram,
            eos_token_id: self.eos_token_id,
            embedding_row_width: self.embedding_row_width,
            multiplier_count: self.multiplier_count,
            layer_multipliers: &self.layer_multipliers,
            head_count: self.head_count,
            head_offsets: &self.head_offsets,
            head_vocab_sizes: &self.head_vocab_sizes,
        }
    }
}
fn config() -> Fixture {
    let mut config = Fixture {
        ngram_size: 3,
        heads_per_ngram: 2,
        eos_token_id: 99,
        embedding_row_width: 8,
        multiplier_count: 3,
        layer_multipliers: [3, 5, 7, 0, 0, 0, 0, 0],
        head_count: 4,
        head_offsets: [0; 32],
        head_vocab_sizes: [0; 32],
    };
    config.head_offsets[..4].copy_from_slice(&[0, 11, 24, 41]);
    config.head_vocab_sizes[..4].copy_from_slice(&[11, 13, 17, 19]);
    config
}

#[test]
fn rows_are_stable_and_stay_in_their_head_ranges() {
    let cfg = config();
    let mut history = PleTokenHistory::default();
    let mut rows = [0u64; 4];
    assert_eq!(history.select_and_push(&cfg, 7, 60, &mut rows).unwrap(), 4);
    assert!(rows[0] < 11 && (11..24).contains(&rows[1]));
    assert!((24..41).contains(&rows[2]) && (41..60).contains(&rows[3]));
    let first = rows;
    let mut fresh = PleTokenHistory::default();
    assert_eq!(fresh.select_and_push(&cfg, 7, 60, &mut rows).unwrap(), 4);
    assert_eq!(rows, first);
}

#[test]
fn eos_resets_history_before_the_following_token() {
    let cfg = config();
    let mut history = PleTokenHistory::default();
    let mut rows = [0u64; 4];
    history.select_and_push(&cfg, 7, 60, &mut rows).unwrap();
    history
        .select_and_push(&cfg, cfg.eos_token_id, 60, &mut rows)
        .unwrap();
    let after_eos = {
        history.select_and_push(&cfg, 8, 60, &mut rows).unwrap();
        rows
    };
    let mut fresh = PleTokenHistory::default();
    fresh.select_and_push(&cfg, 8, 60, &mut rows).unwrap();
    assert_eq!(after_eos, rows);
}

#[test]
fn dedup_is_in_place() {
    let mut rows = [9, 2, 9, 7, 2];
    assert_eq!(sort_dedup_rows(&mut rows), 3);
    assert_eq!(&rows[..3], &[2, 7, 9]);
}
