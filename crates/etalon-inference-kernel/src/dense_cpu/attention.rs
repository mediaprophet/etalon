use super::*;

fn kv_index(
    c: DenseConfig,
    plane: usize,
    value: bool,
    layer: usize,
    position: usize,
    head: usize,
    dim: usize,
) -> usize {
    let plane = if value { plane } else { 0 };
    plane + ((((layer * c.max_context + position) * c.kv_heads + head) * c.head_dim) + dim)
}

pub(super) fn attention<E>(
    c: DenseConfig,
    plane: usize,
    layer: usize,
    position: usize,
    w: &mut DenseWorkspace<'_>,
) -> Result<(), DenseError<E>> {
    for head in 0..c.kv_heads {
        for dim in 0..c.head_dim {
            let src = head * c.head_dim + dim;
            let ki = kv_index(c, plane, false, layer, position, head, dim);
            let vi = kv_index(c, plane, true, layer, position, head, dim);
            w.kv[ki] = w.k[src];
            w.kv[vi] = w.v[src];
        }
    }
    w.attention.fill(0.0);
    let q_per_kv = c.heads / c.kv_heads;
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    for q_head in 0..c.heads {
        let kv_head = q_head / q_per_kv;
        let q_off = q_head * c.head_dim;
        let mut max_score = f32::NEG_INFINITY;
        for past in 0..=position {
            let mut dot = 0.0f32;
            for dim in 0..c.head_dim {
                dot +=
                    w.q[q_off + dim] * w.kv[kv_index(c, plane, false, layer, past, kv_head, dim)];
            }
            let score = dot * scale;
            w.scores[past] = score;
            max_score = max_score.max(score);
        }
        let mut sum = 0.0f32;
        for past in 0..=position {
            let value = (w.scores[past] - max_score).exp();
            w.scores[past] = value;
            sum += value;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(DenseError::Kernel {
                layer: Some(layer),
                role: DenseRole::AttentionSoftmax,
            });
        }
        for past in 0..=position {
            let probability = w.scores[past] / sum;
            for dim in 0..c.head_dim {
                w.attention[q_off + dim] +=
                    probability * w.kv[kv_index(c, plane, true, layer, past, kv_head, dim)];
            }
        }
    }
    Ok(())
}
