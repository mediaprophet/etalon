use super::super::{finite, Error};

pub(super) fn norm_inplace(
    x: &mut [f32],
    weight: &[f32],
    raw: bool,
    epsilon: f32,
) -> Result<(), Error> {
    let mut sum = 0.0f32;
    for &v in x.iter() {
        sum += v * v;
    }
    finite(&[sum])?;
    let denominator = sum / x.len() as f32 + epsilon;
    finite(&[denominator])?;
    let scale = denominator.sqrt().recip();
    finite(&[scale])?;
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = (*v * scale) * if raw { 1.0 + w } else { w };
    }
    finite(x)
}

pub(super) fn rotate(
    x: &mut [f32],
    dim: usize,
    inv_freq: &[f32],
    position: usize,
) -> Result<(), Error> {
    let half = inv_freq.len();
    for head in x.chunks_exact_mut(dim) {
        for (i, &frequency) in inv_freq.iter().enumerate() {
            let angle = position as f32 * frequency;
            finite(&[angle])?;
            let (sin, cos) = angle.sin_cos();
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = b * cos + a * sin;
        }
    }
    finite(x)
}

// Contiguous append-only cache needs no page-table materialization. The final
// token is staged separately, so attention never exposes a partial cache update.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    keys: &[f32],
    values: &[f32],
    cursor: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    scores: &mut [f32],
    out: &mut [f32],
) -> Result<(), Error> {
    let width = kv_heads * dim;
    let scale = (dim as f32).sqrt().recip();
    finite(&[scale])?;
    for h in 0..heads {
        let kvh = h / (heads / kv_heads);
        let query = &q[h * dim..(h + 1) * dim];
        for (t, score) in scores[..=cursor].iter_mut().enumerate() {
            let key = if t == cursor {
                &k[kvh * dim..(kvh + 1) * dim]
            } else {
                &keys[t * width + kvh * dim..t * width + (kvh + 1) * dim]
            };
            let mut dot = 0.0;
            for d in 0..dim {
                dot += query[d] * key[d];
            }
            *score = dot * scale;
        }
        finite(&scores[..=cursor])?;
        let maximum = scores[..=cursor]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for score in &mut scores[..=cursor] {
            *score = (*score - maximum).exp();
            sum += *score;
        }
        finite(&[sum])?;
        if sum <= 0.0 {
            return Err(Error::Shape);
        }
        for score in &mut scores[..=cursor] {
            *score /= sum;
        }
        let dst = &mut out[h * dim..(h + 1) * dim];
        dst.fill(0.0);
        for (t, &score) in scores[..=cursor].iter().enumerate() {
            let value = if t == cursor {
                &v[kvh * dim..(kvh + 1) * dim]
            } else {
                &values[t * width + kvh * dim..t * width + (kvh + 1) * dim]
            };
            for d in 0..dim {
                dst[d] += score * value[d];
            }
        }
        finite(dst)?;
    }
    Ok(())
}
