use super::*;
pub(super) fn tensor(value: Option<TensorInfo>) -> Result<TensorInfo, StreamedQsaError> {
    value.ok_or(StreamedQsaError::MissingTensor)
}

pub(super) fn is_vector(info: TensorInfo, width: usize) -> bool {
    (info.n_dims == 1 && info.dims[0] as usize == width)
        || (info.n_dims == 2 && info.dims[0] as usize == width && info.dims[1] == 1)
}

pub(super) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(super) fn rms_norm_heads(vector: &mut [f32], weights: &[f32], heads: usize, head_dim: usize) {
    for head in 0..heads {
        let offset = head * head_dim;
        let part = &mut vector[offset..offset + head_dim];
        let mut sum = 0.0f32;
        for &value in part.iter() {
            sum += value * value;
        }
        let scale = 1.0 / (sum / head_dim as f32 + 1.0e-6).sqrt();
        // Qwen4Exp q/k/indexer norms are GemmaPlusOneRMSNorm in the source
        // checkpoint, but the GGUF converter pre-folds the +1 into the stored
        // gamma (llama.cpp build_norm): multiply the weight as-is.
        for index in 0..head_dim {
            part[index] *= scale * weights[index % weights.len()];
        }
    }
}

/// NeoX-style partial rotary: only the first `ROTARY_DIM` dims rotate, paired
/// as `(i, i + ROTARY_DIM/2)` with `freq = theta^(-2i/ROTARY_DIM)`.  The rest of
/// each head passes through unchanged, matching `freetoken.layers.rotary`.
pub(super) fn rope_head(vector: &mut [f32], position: usize, theta: f32) {
    const HALF: usize = QWEN4EXP_QSA_ROTARY_DIM / 2;
    for pair in 0..HALF {
        let frequency = theta.powf(-(2.0 * pair as f32) / QWEN4EXP_QSA_ROTARY_DIM as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let first = vector[pair];
        let second = vector[pair + HALF];
        vector[pair] = first * cos - second * sin;
        vector[pair + HALF] = first * sin + second * cos;
    }
}

pub(super) fn rope_heads(
    vector: &mut [f32],
    heads: usize,
    position: usize,
    theta: f32,
    head_dim: usize,
) {
    for head in 0..heads {
        let start = head * head_dim;
        rope_head(&mut vector[start..start + head_dim], position, theta);
    }
}

pub(super) fn insert_ranked(
    index: usize,
    score: f32,
    selected: &mut [usize],
    scores: &mut [f32],
    used: &mut usize,
) {
    let capacity = selected.len().min(scores.len());
    if capacity == 0 {
        return;
    }
    let mut slot = *used;
    if *used < capacity {
        *used += 1;
    } else if score < scores[capacity - 1]
        || (score == scores[capacity - 1] && index > selected[capacity - 1])
    {
        return;
    } else {
        slot = capacity - 1;
    }
    while slot > 0
        && (score > scores[slot - 1] || (score == scores[slot - 1] && index < selected[slot - 1]))
    {
        if slot < capacity {
            scores[slot] = scores[slot - 1];
            selected[slot] = selected[slot - 1];
        }
        slot -= 1;
    }
    scores[slot] = score;
    selected[slot] = index;
}
