pub(super) const RMS_NORM_EPS: f32 = 1e-5;

pub(super) fn rms_norm_inplace(x: &mut [f32], weight: &[f32]) {
    let n = x.len().min(weight.len());
    if n == 0 {
        return;
    }
    let mut ss = 0.0f32;
    for value in &x[..n] {
        ss += *value * *value;
    }
    ss /= n as f32;
    let inv_rms = 1.0 / (ss + RMS_NORM_EPS).sqrt();
    for (value, weight) in x[..n].iter_mut().zip(&weight[..n]) {
        *value = *value * inv_rms * *weight;
    }
}

/// The preserved CPU path uses adjacent pairs, despite its former NEOX comment.
pub(super) fn rope_inplace(
    vec: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: u32,
    base: f32,
    scale: f32,
) {
    let half = head_dim / 2;
    if half == 0 {
        return;
    }
    let scale = if scale > 0.0 && scale.is_finite() {
        scale
    } else {
        1.0
    };
    let scaled_pos = pos as f32 / scale;
    for head in 0..n_heads {
        let off = head * head_dim;
        if off + head_dim > vec.len() {
            return;
        }
        for i in 0..half {
            let theta = scaled_pos * base.powf(-2.0 * i as f32 / head_dim as f32);
            let (s, c) = theta.sin_cos();
            let x0 = vec[off + 2 * i];
            let x1 = vec[off + 2 * i + 1];
            vec[off + 2 * i] = x0 * c - x1 * s;
            vec[off + 2 * i + 1] = x0 * s + x1 * c;
        }
    }
}
