use super::*;

#[test]
fn rope_is_neox_partial_64() {
    // NeoX pairing: dims (i, i+32) rotate together; dims >= 64 untouched.
    let mut v = [0.0f32; 128];
    v[0] = 1.0;
    v[32] = 2.0;
    v[64] = 7.0;
    rope_head(&mut v, 1, 10_000.0);
    let (sin, cos) = 1.0f32.sin_cos();
    assert!((v[0] - (cos - 2.0 * sin)).abs() < 1.0e-5);
    assert!((v[32] - (sin + 2.0 * cos)).abs() < 1.0e-5);
    assert_eq!(v[64], 7.0);
    // Position zero is the identity.
    let mut w = [3.0f32; 128];
    rope_head(&mut w, 0, 10_000.0);
    assert_eq!(w[0], 3.0);
}

#[test]
fn indexer_budget_is_token_level() {
    // `attention.indexer.top_k = 2048` counts TOKENS: 512 complete
    // four-token blocks plus the open group's causal tail.
    assert_eq!(QWEN4EXP_QSA_TOP_BLOCKS, 512);
    assert_eq!(QWEN4EXP_QSA_MAX_SELECTED, 2048 + 3);
}

#[test]
fn ranked_selection_keeps_lowest_index_on_ties() {
    let mut selected = [usize::MAX; 3];
    let mut scores = [f32::NEG_INFINITY; 3];
    let mut used = 0;
    insert_ranked(4, 1.0, &mut selected, &mut scores, &mut used);
    insert_ranked(2, 1.0, &mut selected, &mut scores, &mut used);
    insert_ranked(3, 2.0, &mut selected, &mut scores, &mut used);
    assert_eq!(selected, [3, 2, 4]);
}
