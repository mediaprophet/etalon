#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamedRoutingError {
    BufferTooSmall,
}

/// Fixed top-k selection without a heap or an unstable sort.  Equal logits
/// prefer the lower expert number, keeping decode receipts reproducible.
pub fn select_top_experts<const K: usize>(
    logits: &[f32],
    indices: &mut [usize; K],
    selected_logits: &mut [f32; K],
) -> Result<(), StreamedRoutingError> {
    if logits.len() < K {
        return Err(StreamedRoutingError::BufferTooSmall);
    }
    for slot in 0..K {
        indices[slot] = usize::MAX;
        selected_logits[slot] = f32::NEG_INFINITY;
    }
    for (expert, &logit) in logits.iter().enumerate() {
        for slot in 0..K {
            let better = logit > selected_logits[slot]
                || (logit == selected_logits[slot] && expert < indices[slot]);
            if better {
                for shift in (slot + 1..K).rev() {
                    indices[shift] = indices[shift - 1];
                    selected_logits[shift] = selected_logits[shift - 1];
                }
                indices[slot] = expert;
                selected_logits[slot] = logit;
                break;
            }
        }
    }
    Ok(())
}

/// Softmax only the selected top-k router scores.  The model routes through
/// these normalized coefficients, not through a sparse unnormalised gate.
pub fn normalize_top_experts<const K: usize>(logits: &[f32; K], out: &mut [f32; K]) {
    let mut max = f32::NEG_INFINITY;
    for &value in logits {
        if value > max {
            max = value;
        }
    }
    let mut sum = 0.0f32;
    for index in 0..K {
        let value = (logits[index] - max).exp();
        out[index] = value;
        sum += value;
    }
    if sum > 0.0 && sum.is_finite() {
        for value in out {
            *value /= sum;
        }
    } else {
        for value in out.iter_mut() {
            *value = 0.0;
        }
    }
}

pub fn swiglu_into(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<(), StreamedRoutingError> {
    if gate.len() != up.len() || out.len() < gate.len() {
        return Err(StreamedRoutingError::BufferTooSmall);
    }
    for index in 0..gate.len() {
        let value = gate[index];
        out[index] = (value / (1.0 + (-value).exp())) * up[index];
    }
    Ok(())
}
