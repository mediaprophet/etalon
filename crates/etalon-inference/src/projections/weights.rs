use etalon_model::{ggml::*, inference::TensorInfo};
/// Cold conversion is bounded by the caller's allocation budget.
pub fn dequant_dense_into(info: &TensorInfo, raw: &[u8], n_in: usize, out: &mut [f32]) -> bool {
    if n_in == 0 || out.len() % n_in != 0 {
        return false;
    }
    #[cfg(feature = "native-weights")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(n_in)
            .enumerate()
            .map(|(r, row)| dequant_matrix_row_into(raw, info, r, row).is_ok())
            .all(|ok| ok)
    }
    #[cfg(not(feature = "native-weights"))]
    out.chunks_mut(n_in)
        .enumerate()
        .all(|(r, row)| dequant_matrix_row_into(raw, info, r, row).is_ok())
}
pub fn dense_f32(
    info: &TensorInfo,
    raw: &[u8],
    n_in: usize,
    n_out: usize,
    max_bytes: usize,
) -> Option<Vec<f32>> {
    let total = n_in.checked_mul(n_out)?;
    if n_in == 0 || total.checked_mul(4)? > max_bytes {
        return None;
    }
    let mut dense = vec![0.0; total];
    dequant_dense_into(info, raw, n_in, &mut dense).then_some(dense)
}
pub fn promote_f16(info: &TensorInfo, raw: &[u8], max_bytes: usize) -> Option<Vec<u8>> {
    if !matches!(
        info.ggml_type,
        GGML_TYPE_Q4_K | GGML_TYPE_Q4_K_SOA | GGML_TYPE_Q6_K | GGML_TYPE_Q8_0
    ) || info.n_dims < 2
        || info.dims[0] == 0
        || info.dims[1] == 0
    {
        return None;
    }
    let width = usize::try_from(info.dims[0]).ok()?;
    let rows = usize::try_from(info.dims[1]).ok()?;
    let bytes = width.checked_mul(rows)?.checked_mul(2)?;
    let scratch = width.checked_mul(4)?;
    if bytes.checked_add(scratch)? > max_bytes {
        return None;
    }
    let mut packed = vec![0u8; bytes];
    let mut row = vec![0.0; width];
    for r in 0..rows {
        dequant_matrix_row_into(raw, info, r, &mut row).ok()?;
        for (c, &value) in row.iter().enumerate() {
            let offset = (r * width + c) * 2;
            packed[offset..offset + 2].copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
        }
    }
    Some(packed)
}
