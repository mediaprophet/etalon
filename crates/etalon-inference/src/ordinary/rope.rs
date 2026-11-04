pub fn rope_inplace(
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
            // Interleaved ("normal"/llama) rope: pair adjacent dims (2i, 2i+1). GGUF llama-arch
            // (SmolLM2) is is_neox=false — weights are permuted for interleaved, NOT split-half.
            let theta = scaled_pos * base.powf(-2.0 * i as f32 / head_dim as f32);
            let (s, c) = theta.sin_cos();
            let x0 = vec[off + 2 * i];
            let x1 = vec[off + 2 * i + 1];
            vec[off + 2 * i] = x0 * c - x1 * s;
            vec[off + 2 * i + 1] = x0 * s + x1 * c;
        }
    }
}
